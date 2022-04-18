import os
from PIL import Image

import pandas as pd
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from sklearn.model_selection import train_test_split


root = "Fabio/task3/"
img_dir = root + "food"
train_triplets_path = root + "train_triplets.txt"
test_path = root + "test_triplets.txt"
triplets_path = root + "triplets.txt"
train_split_path = root + "train_split.txt"
val_split_path = root + "val_split.txt"


AVAIL_GPUS = min(1, torch.cuda.device_count())
learning_rate = 0.063
batch_size = 64
num_workers = 8
tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)  # could also try augmentation


def prepare_data():
    # get triplets
    with open(train_triplets_path) as f:
        triplets = f.readlines()
    triplets = [triplet.strip().split(" ") for triplet in triplets]

    # for each triplet we create two samples and write to a new txt file:
    #  - one for the original triplet; with label 1
    #  - one with the second and third image swapped; with label 0
    with open(triplets_path, "w") as f:
        for triplet in triplets:
            f.writelines(" ".join(triplet) + " " + str(1) + "\n")
            triplet[1], triplet[2] = triplet[2], triplet[1]
            f.writelines(" ".join(triplet) + " " + str(0) + "\n")
    with open(triplets_path, "r") as f:
        triplets = f.readlines()
    triplets = [triplet.strip().split(" ") for triplet in triplets]

    # split into training and validation set
    train_triplets, val_triplets = train_test_split(
        triplets, test_size=0.1, random_state=489, shuffle=True
    )

    # write training and validation set to txt (so that DataSet can access it)
    with open(train_split_path, "w") as f:
        for item in train_triplets:
            f.writelines(" ".join(item) + "\n")
    with open(val_split_path, "w") as f:
        for item in val_triplets:
            f.writelines(" ".join(item) + "\n")


class TripletsDataset(Dataset):
    def __init__(self, triplets_path, img_dir, transform=None):
        self.triplets_path = triplets_path
        self.img_dir = img_dir
        self.transform = transform

        with open(self.triplets_path) as f:
            triplets = f.readlines()
        triplets = [triplet.strip().split(" ") for triplet in triplets]
        self.labels = [int(triplet[-1]) for triplet in triplets]
        self.triplets = [triplet[:-1] for triplet in triplets]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        label = self.labels[idx]
        label = torch.tensor(label)

        paths = [os.path.join(self.img_dir, f"{i}.jpg") for i in triplet]
        images = [Image.open(path) for path in paths]

        if self.transform:
            images = [self.transform(image) for image in images]

        return images, label


class TripletsTestset(Dataset):
    def __init__(self, triplets_path, img_dir, transform=None):
        self.triplets_path = triplets_path
        self.img_dir = img_dir
        self.transform = transform

        with open(self.triplets_path) as f:
            triplets = f.readlines()
        self.triplets = [triplet.strip().split(" ") for triplet in triplets]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        paths = [os.path.join(self.img_dir, f"{i}.jpg") for i in triplet]
        images = [Image.open(path) for path in paths]

        if self.transform:
            images = [self.transform(image) for image in images]

        return images


class Classifier(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, lr=1e-3, batch_size=64):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = lr

        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.requires_grad_(False)  # no fine-tuning
        self.embedding = nn.Sequential(
            *list(self.efficientnet.children())[:-1]
        )  # this is efficientnet without its classification layers

        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 1280, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 2),
        )  # the classifier we are training

        self.loss = nn.CrossEntropyLoss()

    def forward(self, images):
        img_a, img_b, img_c = images

        phi_a = self.embedding(img_a).squeeze()
        phi_b = self.embedding(img_b).squeeze()
        phi_c = self.embedding(img_c).squeeze()
        phi = torch.concat((phi_a, phi_b, phi_c), dim=1)

        y_hat = self.classifier(phi)

        return y_hat

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)

    def training_step(self, batch):
        x, y = batch
        # x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
        )
        return loader


if __name__ == "__main__":
    prepare_data()

    train_dataset = TripletsDataset(train_split_path, img_dir, tfms)
    val_dataset = TripletsDataset(val_split_path, img_dir, tfms)

    model = Classifier(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=learning_rate,
        batch_size=batch_size,
    )

    bar = TQDMProgressBar(refresh_rate=1)
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=5, verbose=True
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        min_epochs=1,
        max_epochs=250,
        callbacks=[bar, early_stop_callback],
        auto_lr_find=False,
        auto_scale_batch_size=False,
    )
    # trainer.tune(model)
    trainer.fit(model)

    # predictions
    test_dataset = TripletsTestset(test_path, img_dir, tfms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    output = trainer.predict(model, dataloaders=test_loader)[0]
    logits_list = F.softmax(output, dim=1)
    y_hat = []
    for logits in logits_list:
        y_hat.append(1 if logits[0] > 0.5 else 0)

    df_pred = pd.DataFrame(y_hat)
    df_pred.to_csv("submission.txt", index=False, header=None)
