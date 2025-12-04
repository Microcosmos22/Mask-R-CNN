import cv2
import json
import torch
import torch.nn
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN
from sklearn.model_selection import train_test_split
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F_transforms
from torch.utils.data import Subset

from dataloader import *
from metrics import *

from sklearn.model_selection import KFold
import warnings
from itertools import product
from wakepy import keep

import json
import os


warnings.filterwarnings('ignore')

batch = 4
# Checking GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

base_path = "../recodai-luc-scientific-image-forgery-detection/"
test_dataset = ForgeryDataset(
    None,
    os.path.join(base_path, "supplemental_images"),
    os.path.join(base_path, "supplemental_masks"),
    transform=train_transform
)

full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    transform=train_transform
)

#full_dataset = Subset(full_dataset, list(range(500)))
indices = list(range(len(full_dataset)))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.1,
    random_state=42,
    shuffle=True
)


train_subset = Subset(full_dataset, train_idx)
val_subset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=batch, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_subset, batch_size=batch, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
eval_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


feature_extractors = []

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, n_classes, 1)
        """ outputs logits """

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))

        x = self.conv2(torch.cat([self.up2(x3), x2], dim=1))
        x = self.conv1(torch.cat([self.up1(x), x1], dim=1))

        return self.out(x)

def train_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0

    for images, masks, id in tqdm(dataloader, desc="Training"):

        images = torch.stack(images).to(device)      # (B,3,256,256)
        masks  = torch.stack(masks).float().to(device)  # (B,256,256)

        optimizer.zero_grad()
        outputs = model(images)                      # (B,1,256,256)

        # ensure shapes align
        outputs = outputs.squeeze(1)                 # (B,256,256)
        bce_loss = criterion(outputs, masks)   # call the loss function on tensors
        dice_loss = soft_dice(outputs, masks)  # your Dice loss

        loss = bce_loss + dice_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



def validate_epoch(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks, id in tqdm(dataloader, desc="Validation"):
            images = torch.stack(images).to(device)
            masks  = torch.stack(masks).float().to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)
            dice = soft_dice(outputs, masks)

            loss = criterion(outputs, masks) + soft_dice(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_parameters(train_loader, val_loader, machinepath, num_epochs, device, criterion, lr=0.001):
    model = UNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, val_losses = [], []

    best_val = float("inf")
    patience = 2
    wait = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        train_losses.append(train_loss)

        val_loss = validate_epoch(model, val_loader, device, criterion) if val_loader else 0
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            torch.save(model.state_dict(), machinepath)  # save best model
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    return model, train_losses, val_losses

if __name__ == "__main__":
    losses = {"params": [], "errors": []}
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))


    machinepath = "unet_overfit_model.pth"
    num_epochs = 30
    full_dataset = Subset(full_dataset, list(range(5)))

    feat_ex = [0]
    lr = [0.001]
    weight_decay = [0.001]
    step_size = [5]
    gamma = [0.1]
    samplR=2
    rpn_pre_train = 1000
    rpn_pre_test = 1000
    rpn_post_train = 200
    rpn_post_test = 200

    out_ch = [1]

    all_combinations = list(product(
        feat_ex, out_ch, lr, weight_decay,
        step_size, gamma
    ))

    print(f"Total combinations: {len(all_combinations)}")
    print(all_combinations)

    np.save(f"losses.npy", np.asarray([1,2,3,4]))

    with keep.running():

        for combo in all_combinations:

            eval_subset = torch.utils.data.Subset(train_subset, [0])

            print(len(train_subset), len(eval_subset))

            # optionally set transforms
            val_subset.dataset.transform = val_transform



            #print(f"Feat_ex: {feat_ex}, out_ch: {out_ch}, lr: {lr}, weight_d: {weight_decay}, step_size: {step_size}, gamma: {gamma}, samplR: {samplR}, rpn_pre_train: {rpn_pre_train} ")
            model, train_loss, val_loss = train_parameters(train_loader, val_loader, machinepath, num_epochs, device, criterion)

            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.savefig("last_training_overfit_UNETTTT.png")
            plt.show()


            print(" Saving losses.json")
            losses["params"].append(combo)
            with open("losses.json", "w") as f:
                json.dump(losses, f, indent=4)
