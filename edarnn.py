import os
import cv2
import json
import torch
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
from scoring import *

import warnings


warnings.filterwarnings('ignore')

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

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    transform=train_transform
)
#full_dataset = Subset(full_dataset, list(range(400)))

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

feature_extractors = []

def create_light_mask_rcnn(feat_ex = 0, out_ch=256, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=1,
rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200, num_classes = 2):
    if feat_ex ==0:
        backbone = torchvision.models.mobilenet_v3_small(pretrained=False).features
    elif feat_ex == 1:
        backbone = torchvision.models.mobilenet_v3_large(pretrained=False).features
    elif feat_ex == 2:
        backbone = torchvision.models.resnet34(pretrained=False).features

    backbone.out_channels = 576

    # extracts characteristics from an image
    backbone = nn.Sequential(
        backbone,
        nn.Conv2d(576, out_ch, kernel_size=1),
        nn.ReLU(inplace=True)
    )
    backbone.out_channels = out_ch

    # Anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ROI pools
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=5,
        sampling_ratio=samplR
    )

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=10,
        sampling_ratio=samplR
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=224,
        max_size=224,
        rpn_pre_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=200,
        rpn_post_nms_top_n_test=200,
        box_detections_per_img=100
    )

    return model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, device):
    model.train()  # For validation, we use train mode because of the features of Mask R-CNN
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validation")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)

def train_parameters(feat_ex = 0, out_ch=256, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=1,
rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200):
    model = create_light_mask_rcnn(feat_ex = 0, out_ch=256, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=1,
    rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200)
    model.to(device)
    print(f"Device {device}".format())
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    num_epochs = 6
    train_losses = []
    val_losses = []
    # Early stopping parameters
    patience = 3        # epochs to wait for improvement
    best_iou = float('inf')
    epochs_no_improve = 0


    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        """ Train, validate, evaluate """
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        iou, dice, props = evaluate_segmentation(model, val_loader, device)

        scheduler.step()


        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"IoU: {np.mean(iou):.4f}, DICE: {np.mean(dice):.4f}")

        """ Early stopping check """
        if iou < best_iou:
            best_iou = iou
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), 'mask_rcnn_best.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                early_stop = True
                break
    return np.mean(iou), np.mean(dice), train_loss, val_loss

if __name__ == "__main__":
    for feat_ex in [0,1,2]:
        for out_ch in [256]:
            #print(f"Feat_ex: {feat_ex}, out_ch: {out_ch}, lr: {lr}, weight_d: {weight_decay}, step_size: {step_size}, gamma: {gamma}, samplR: {samplR}, rpn_pre_train: {rpn_pre_train} ")
            iou, dice, train_loss, val_loss = train_parameters()
