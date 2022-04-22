import os
import pickle
from functools import partial
from PIL import Image
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.linalg import norm
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from sklearn.model_selection import train_test_split


base = "../"  # where the data is located
img_dir = base + "food"  # where you unzipped food.zip
train_path = base + "train_triplets.txt"
test_path = base + "test_triplets.txt"


def get_split():
    triplets = np.loadtxt(train_path, delimiter=" ").astype(int)

    train_triplets, val_triplets = train_test_split(
        triplets, test_size=0.2, random_state=489, shuffle=True
    )

    return train_triplets, val_triplets


class TripletsDataset(Dataset):
    def __init__(self, triplets, features):
        self.triplets = triplets
        self.features = features

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        a, b, c = self.features[triplet]
        return a, b, c


# this defines our network.
# we use a pytorch lightning LightningModule instead of a nn.Module
# for its convenient features. Hence why we implement many of the
# methods below, they are reserved by lightning and used by the
# trainer
class SimilarityNet(pl.LightningModule):
    def __init__(self, features_dim, lr=1e-3, batch_size=8):
        super().__init__()

        self.batch_size = batch_size
        self.learning_rate = lr

        self.embedding = nn.Linear(features_dim, 1024)

        self.loss = nn.TripletMarginLoss(margin=1.0)
        self.val_loss = partial(F.triplet_margin_loss, margin=0)

    def forward(self, a, b, c):
        embedded_a = self.embedding(a)
        embedded_b = self.embedding(b)
        embedded_c = self.embedding(c)

        return embedded_a, embedded_b, embedded_c

    def training_step(self, batch):
        a, b, c = batch
        embedded_a, embedded_b, embedded_c = self(a, b, c)
        loss = self.loss(embedded_a, embedded_b, embedded_c)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, idx):
        a, b, c = batch
        embedded_a, embedded_b, embedded_c = self(a, b, c)
        loss = self.val_loss(embedded_a, embedded_b, embedded_c)
        self.log("val_loss", loss)

        d_ab = norm(embedded_a - embedded_b, axis=-1).squeeze()
        d_ac = norm(embedded_a - embedded_c, axis=-1).squeeze()
        acc = (d_ab < d_ac).float().mean()
        self.log("val_acc", acc)

        return {"val_loss": loss, "val_acc": acc}

    def predict_step(self, batch, idx):
        a, b, c = batch
        embedded_a, embedded_b, embedded_c = self(a, b, c)
        d_ab = norm(embedded_a - embedded_b, axis=-1).squeeze()
        d_ac = norm(embedded_a - embedded_c, axis=-1).squeeze()
        pred = (d_ab <= d_ac).int()
        return pred

    def configure_optimizers(self):
        return SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=1e-3,
        )

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_acc)


# first we compute the embedded images with a pretrained resnet
backbone = models.resnet152(pretrained=True)
backbone.requires_grad_(False)
features_dim = list(backbone.children())[-1].in_features

if os.path.exists("r152_features.pkl"):
    # fetch precopmuted features from earlier run
    with open("r152_features.pkl", "rb") as f:
        features = pickle.load(f)
else:
    # compute features
    feature_map = nn.Sequential(
        *list(backbone.children())[:-1]
    )  # create embedding map by omitting final classification layer
    feature_map.cuda()
    feature_map.eval()

    tfms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((242, 354)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )  # preprocessing function

    n_imgs = 10_000
    features = torch.empty((n_imgs, features_dim))
    with torch.no_grad():
        for i in tqdm(range(n_imgs)):
            path = os.path.join(img_dir, str(i).rjust(5, "0") + ".jpg")
            img = tfms(Image.open(path)).unsqueeze(0).cuda()
            phi = feature_map(img)  # forward pass
            features[i] = phi.squeeze().cpu().float()

    # save features
    with open("r152_features.pkl", "wb") as f:
        pickle.dump(features, f)


# now we train our classifier
learning_rate = 5e-4
batch_size = 2048
num_workers = 32

train_triplets, val_triplets = get_split()
train_dataset = TripletsDataset(train_triplets, features)
val_dataset = TripletsDataset(val_triplets, features)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
)


model = SimilarityNet(
    features_dim=features_dim,
    lr=learning_rate,
    batch_size=batch_size,
)

bar = TQDMProgressBar(refresh_rate=1)
early_stop = EarlyStopping(
    monitor="val_acc", mode="max", min_delta=0.002, patience=10, verbose=True
)

trainer = Trainer(
    # fast_dev_run=True,
    accelerator="gpu",
    devices=[1],
    # auto_select_gpus=True,
    min_epochs=1,
    max_epochs=250,
    callbacks=[bar, early_stop],
    auto_lr_find=True,
    auto_scale_batch_size=False,
)

# trainer.tune(model)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# predict
test_triplets = np.loadtxt(test_path, delimiter=" ").astype(int)
test_dataset = TripletsDataset(test_triplets, features)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    shuffle=False,
)

predictions = trainer.predict(model, test_loader)
predictions = torch.cat(predictions).tolist()

df_pred = pd.DataFrame(predictions)
df_pred.to_csv("../predictions/r152.txt", index=False, header=None)
