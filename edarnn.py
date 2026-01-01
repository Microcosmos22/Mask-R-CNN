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

import os
import time

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

import warnings
from itertools import product
from wakepy import keep

import json
import os
from sklearn.model_selection import GridSearchCV

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



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

full_dataset_subset = Subset(full_dataset, list(range(2)))  # keeps mapping: [0,1]
subset_indices = full_dataset_subset.indices  # ← real original dataset indices

train_idx, val_idx = train_test_split(
    range(len(full_dataset_subset)),
    test_size=0.1,
    random_state=42,
    shuffle=False
)
"""
print("\nTRAIN FILES:")
for idx in train_idx:
    orig_idx = subset_indices[idx]     # map subset index → original dataset index
    filename = full_dataset.get_filename(orig_idx)
    print(str(orig_idx)+": ", filename['image_id']+".png")"""



train_subset = Subset(full_dataset_subset, train_idx)
val_subset = Subset(full_dataset_subset, val_idx)


feature_extractors = []


eval_subset = torch.utils.data.Subset(train_subset, [0])

print(len(train_subset), len(eval_subset))

# optionally set transforms
val_subset.dataset.transform = val_transform

train_loader = DataLoader(train_subset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def get_coco_initialized_model(num_classes=2):
    # Load full COCO-trained Mask R-CNN
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")

    # ---- Replace box predictor (class + bbox) ----
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    # ---- Replace mask predictor ----
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for idx, (images, targets, _) in enumerate(tqdm(dataloader, desc="Training")):


        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        for idx, t in enumerate(targets):
            H, W = raw_img.shape[:2]

            boxes = t['boxes']
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        model.train()
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader), loss_dict['loss_mask'].item(), loss_dict['loss_box_reg'], loss_dict['loss_classifier']


def validate_epoch(model, dataloader, device):
    model.train()  # For validation, we use train mode because of the features of Mask R-CNN
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(tqdm(dataloader, desc="Validation")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)

def save_model_safe(model, path, max_retries=5, delay=0.5):
    """Save model safely on Windows, retrying if file is locked."""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                os.remove(path)  # remove previous file
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)
            print(f"Saved model to {path}")

            print("\n")
            return
        except PermissionError:
            print(f"File {path} is locked. Waiting {delay}s and retrying...")
            time.sleep(delay)
    raise PermissionError(f"Could not write to {path} after {max_retries} retries")


def train_parameters(train_loader, val_loader, eval_loader, model, num_epochs, feat_ex = 0, out_ch=256, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=2,
rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200, early = False):

    if model is not None:
        print(" LOADING MODEL: ")
        for p in model.backbone.parameters():
            p.requires_grad = True
    else:
        model = get_coco_initialized_model()
        for p in model.backbone.parameters():
            p.requires_grad = False
    model.to(device)

    # 3️⃣ Create the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.001)
    #optimizer = torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    val_losses = []
    rcnn_losses = []
    # Early stopping parameters
    patience = 4        # epochs to wait for improvement
    best_val = 10000000.1
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        """ Train, validate, evaluate """
        train_loss, loss_mask, loss_box_reg, loss_classifier = train_epoch(model, train_loader, optimizer, device, epoch)

        train_losses.append(train_loss)

        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device)
            val_losses.append(val_loss)
            rcnn_losses.append([loss_mask, loss_box_reg.detach().numpy(), loss_classifier.detach().numpy()])
            print(f"\nLOSSES: Train: {train_loss:.4f}, Val: {val_losses[-1]:.4f}, Mask: {loss_mask:.4f}, Box regr. {loss_box_reg:.4f}, Classifier: {loss_classifier:.4f}")
        else:
            val_losses.append(0)
            print(f"\nTrain Loss: {train_loss:.4f}")


        if eval_loader is not None:
            iou, dice, props = evaluate_segmentation(model, eval_loader, device)

            scheduler.step()

            print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_losses[-1]:.4f}")
            print(f"IoU: {np.mean(iou):.4f}, DICE: {np.mean(dice):.4f}")

            best_iou = np.mean(iou)
            best_dice = np.mean(dice)

        if not early:
            epochs_no_improve = 0
            save_model_safe(model, machinepath)

        if early:
            if (best_val > val_loss):
                best_val = val_loss
                epochs_no_improve = 0
                if os.path.exists(machinepath):
                    os.remove(machinepath)   # safely remove old file
                save_model_safe(model, machinepath)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    early_stop = True
                    break
    if eval_loader is not None:
        return model, best_iou, best_dice, train_loss, val_loss
    else:
        return model, train_losses, val_losses, np.asarray(rcnn_losses)

if __name__ == "__main__":
    losses = {"params": [], "errors": []}
    count = 0

    machinepath = "../pretrained.pth"
    num_epochs = 300
    batch = 1

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

    with keep.running():

        for combo in all_combinations:

            for batch_idx, (images, targets, _) in enumerate(tqdm(train_loader, desc="Validation")):
                print(len(targets[0]["boxes"]), targets[0]["masks"].sum())
            #print(f"Feat_ex: {feat_ex}, out_ch: {out_ch}, lr: {lr}, weight_d: {weight_decay}, step_size: {step_size}, gamma: {gamma}, samplR: {samplR}, rpn_pre_train: {rpn_pre_train} ")
            model, train_losses, val_losses, rcnn_losses = train_parameters(train_loader, val_loader, None, None, 50, combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], samplR, rpn_pre_train, rpn_pre_test, rpn_post_train, rpn_post_test, False)
            model, train_losses, val_losses, rcnn_losses = train_parameters(train_loader, val_loader, None, model, 50, combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], samplR, rpn_pre_train, rpn_pre_test, rpn_post_train, rpn_post_test, False)


            plt.clf()
            plt.plot(np.log(train_losses), label="Log Train Err")
            plt.plot(np.log(val_losses), label="Log Val Err")
            plt.legend()
            plt.savefig("./data/last_training_errs.png")
            plt.clf()
            plt.plot(np.log(rcnn_losses[:,0]), label="Mask")
            plt.plot(np.log(rcnn_losses[:,1]), label=" Box regr.")
            plt.plot(np.log(rcnn_losses[:,2]), label="Classifier")
            plt.legend()
            plt.savefig("./data/last_training_boxerrs.png")


            print(" Saving losses.json")
            losses["params"].append(combo)
            with open("./data/losses.json", "w") as f:
                json.dump(losses, f, indent=4)
