import os
import sys
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
from sklearn.model_selection import train_test_split


IMG_DIR = "food"
TRAIN_PATH = "train_triplets.txt"
TEST_PATH = "test_triplets.txt"


def get_split(val_ratio):
    triplets = np.loadtxt(TRAIN_PATH, delimiter=" ").astype(int)
    train_triplets, val_triplets = train_test_split(
        triplets, test_size=val_ratio, shuffle=True
    )
    return train_triplets, val_triplets


def get_features(name):
    pkl_path = f"{name}_features.pkl"

    if name == "ResNet18":
        backbone = models.resnet18(pretrained=True)
    elif name == "ResNet34":
        backbone = models.resnet34(pretrained=True)
    elif name == "ResNet50":
        backbone = models.resnet50(pretrained=True)
    elif name == "ResNet101":
        backbone = models.resnet101(pretrained=True)
    elif name == "ResNet152":
        backbone = models.resnet152(pretrained=True)
    elif name == "EfficientNetb0":
        backbone = models.efficientnet_b0(pretrained=True)
    elif name == "EfficientNetb1":
        backbone = models.efficientnet_b1(pretrained=True)
    elif name == "EfficientNetb2":
        backbone = models.efficientnet_b2(pretrained=True)
    elif name == "EfficientNetb3":
        backbone = models.efficientnet_b3(pretrained=True)
    elif name == "EfficientNetb4":
        backbone = models.efficientnet_b4(pretrained=True)
    elif name == "EfficientNetb5":
        backbone = models.efficientnet_b5(pretrained=True)
    elif name == "EfficientNetb6":
        backbone = models.efficientnet_b6(pretrained=True)
    elif name == "EfficientNetb7":
        backbone = models.efficientnet_b7(pretrained=True)
    elif name == "ViT_b_16":
        backbone = models.vit_b_16(pretrained=True)
    elif name == "ViT_b_32":
        backbone = models.vit_b_32(pretrained=True)
    elif name == "ViT_l_16":
        backbone = models.vit_l_16(pretrained=True)
    elif name == "ViT_l_32":
        backbone = models.vit_l_32(pretrained=True)
    else:
        sys.exit("Error: This model is not implemented.")

    if name.startswith("ResNet"):
        features_dim = backbone.fc.in_features
    if name.startswith("EfficientNet"):
        features_dim = backbone.classifier[1].in_features
    elif name.startswith("ViT"):
        features_dim = backbone.heads[0].in_features

    if os.path.exists(pkl_path):
        # fetch precopmuted features from earlier run
        with open(pkl_path, "rb") as f:
            features = pickle.load(f)
    else:
        # compute features
        if name.startswith("ResNet") or name.startswith("EfficientNet"):
            feature_map = nn.Sequential(*list(backbone.children())[:-1])
            tfms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((242, 354)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )  # preprocessing function
        elif name.startswith("ViT"):
            feature_map = nn.Sequential(*list(backbone.children())[:-1])[1]
            tfms = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )  # preprocessing function

        feature_map.cuda().eval()

        n_imgs = 10_000
        features = torch.empty((n_imgs, features_dim))

        with torch.no_grad():
            for i in tqdm(range(n_imgs)):
                path = os.path.join(IMG_DIR, f"{str(i).rjust(5, '0')}.jpg")
                img = tfms(Image.open(path)).unsqueeze(0).cuda()
                if name.startswith("ResNet") or name.startswith("EfficientNet"):
                    phi = feature_map(img)  # forward pass
                elif name.startswith("ViT"):
                    x = backbone._process_input(img)
                    batch_class_token = backbone.class_token.expand(x.shape[0], -1, -1)
                    x = torch.cat([batch_class_token, x], dim=1)
                    phi = feature_map(x)[:, 0]
                features[i] = phi.squeeze()

        features = features.cpu().float()

        # save features
        with open(pkl_path, "wb") as f:
            pickle.dump(features, f)

    return features, features_dim


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
    def __init__(
        self,
        features_dim,
        margin=5.0,
        embedding_dim=1024,
        lr=1e-3,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-3,
        batch_size=8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.learning_rate = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.embedding = nn.Linear(features_dim, embedding_dim)

        self.loss = nn.TripletMarginLoss(margin=margin)
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
        optimizer = SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="max"),
                "monitor": "val_acc",
                "frequency": 2,
            },
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_acc", avg_acc)


# first we compute the embedded images with a pretrained network
backbone = "ResNet152"  # choose one of ResNet{18,34,50,101,152}, EfficientNetb{0,1,2,3,4,5,6,7}, ViT_{b_16,b_32,l_16,l_32}
features, features_dim = get_features(backbone)

# hyperparameters
EMBEDDING_DIM = 4096
LEARNING_RATE = 5e-3
MOMENTUM = 0.9
NESTEROV = True
WEIGHT_DECAY = 1e-3
MARGIN = 5.0
VAL_RATIO = 0.1
BATCH_SIZE = 4096
NUM_WORKERS = 32

train_triplets, val_triplets = get_split(VAL_RATIO)
train_dataset = TripletsDataset(train_triplets, features)
val_dataset = TripletsDataset(val_triplets, features)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
)

bar = TQDMProgressBar(refresh_rate=1)
early_stop = EarlyStopping(
    monitor="val_acc", mode="max", min_delta=0.0, patience=20, verbose=True
)

trainer = Trainer(
    # fast_dev_run=True, # uncomment to debug
    accelerator="gpu",
    devices=torch.cuda.device_count(),
    auto_select_gpus=True,
    min_epochs=1,
    max_epochs=500,
    callbacks=[bar, early_stop],
    auto_lr_find=True,
    auto_scale_batch_size=False,
)

model = SimilarityNet(
    features_dim=features_dim,
    margin=MARGIN,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    nesterov=NESTEROV,
    weight_decay=WEIGHT_DECAY,
    batch_size=BATCH_SIZE,
)

# trainer.tune(model)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# predict
test_triplets = np.loadtxt(TEST_PATH, delimiter=" ").astype(int)
test_dataset = TripletsDataset(test_triplets, features)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    shuffle=False,
)

predictions = trainer.predict(model, test_loader)
predictions = torch.cat(predictions).tolist()

df_pred = pd.DataFrame(predictions)
sub_path = f"{backbone}.txt"
df_pred.to_csv(f"../predictions/{sub_path}", index=False, header=None)
