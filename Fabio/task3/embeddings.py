from PIL import Image
from functools import lru_cache
import pickle

import torch
from torch import nn
from torchvision import models, transforms
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 1280
n_samples = 100_000

tfms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)  # preprocessing

efficientnet = models.efficientnet_b0(pretrained=True)  # load pretrained model
embedding = lru_cache(
    nn.Sequential(*list(efficientnet.children())[:-1]), maxsize=512
)  # create embedding map by omitting final classification layer

# compute embeddings
embeddings = np.empty((n_samples, embedding_dim))
with torch.no_grad():
    for i in range(n_samples + 1):
        path = "../food/" + str(i).rjust(5, "0") + ".jpg"
        img = tfms(Image.open(path)).unsqueeze(0).to(device)
        phi = embedding(img)  # forward pass
        embeddings[i] = phi.squeeze().detach().numpy()

# serialize
filename = "../embeddings.pkl"
outfile = open(filename, "wb")
pickle.dump(embeddings, outfile)
outfile.close()
