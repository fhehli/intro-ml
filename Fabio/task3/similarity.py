from functools import partial
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


base = "Fabio/task3/"  # where the data is located
img_dir = base + "food"  # where you unzipped food.zip
train_path = base + "train_triplets.txt"
test_path = base + "test_triplets.txt"
train_split_path = base + "train_split.txt"
val_split_path = base + "val_split.txt"


AVAIL_GPUS = min(1, torch.cuda.device_count())
learning_rate = 1e-3
batch_size = 128
num_workers = 8
tfms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)  # could also try augmentation


def prepare_data():
    # get triplets
    with open(train_path) as f:
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
        self.triplets = [triplet.strip().split(" ") for triplet in triplets]

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]

        paths = (os.path.join(self.img_dir, f"{i}.jpg") for i in triplet)
        images = (Image.open(path) for path in paths)

        if self.transform:
            images = (self.transform(image) for image in images)

        img_a, img_b, img_c = images

        return img_a, img_b, img_c


class SimilarityNet(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, lr=1e-3, batch_size=64):
        super().__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = lr

        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.features = nn.Sequential(
            *list(self.efficientnet.children())[:-1]
        )  # this is efficientnet without its classification layers
        self.features.requires_grad_(False)  # no fine-tuning
        self.embedding = nn.Linear(1280, 1024)

        self.loss = nn.TripletMarginLoss(margin=3.0)
        self.val_loss = partial(F.triplet_margin_loss, margin=0)

    def forward(self, img_a, img_b, img_c):

        phi_a = self.features(img_a)
        phi_b = self.features(img_b)
        phi_c = self.features(img_c)

        phi_a = phi_a.view(phi_a.size(0), -1)
        phi_b = phi_b.view(phi_b.size(0), -1)
        phi_c = phi_c.view(phi_c.size(0), -1)

        embedded_a = self.embedding(phi_a)
        embedded_b = self.embedding(phi_b)
        embedded_c = self.embedding(phi_c)

        return embedded_a, embedded_b, embedded_c

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

    def training_step(self, batch):
        img_a, img_b, img_c = batch
        embedded_a, embedded_b, embedded_c = self(img_a, img_b, img_c)
        loss = self.loss(embedded_a, embedded_b, embedded_c)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        img_a, img_b, img_c = batch
        embedded_a, embedded_b, embedded_c = self(img_a, img_b, img_c)
        loss = self.val_loss(embedded_a, embedded_b, embedded_c)
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

    model = SimilarityNet(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=learning_rate,
        batch_size=batch_size,
    )

    bar = TQDMProgressBar(refresh_rate=1)
    early_stop = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=5, verbose=True
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        min_epochs=1,
        max_epochs=250,
        callbacks=[bar, early_stop],
        auto_lr_find=False,
        auto_scale_batch_size=False,
    )

    trainer.fit(model)

    # predictions
    test_dataset = TripletsDataset(test_path, img_dir, tfms)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    output = trainer.predict(model, dataloaders=test_loader)

    y_hat = []

    df_pred = pd.DataFrame(y_hat)
    df_pred.to_csv("submission.txt", index=False, header=None)
