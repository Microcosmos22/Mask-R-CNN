import numpy as np
import torch
import matplotlib.pyplot as plt
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
from edarnn import *

def to_numpy(mask):
    """Convert torch tensor to 2D NumPy bool array."""
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    return mask.astype(bool)

def plot_masks(true_mask, pred_mask, title_prefix="", save_path=None):
    true_mask = np.squeeze(true_mask)
    pred_mask = np.squeeze(pred_mask)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(true_mask, cmap="gray")
    ax[0].set_title(f"{title_prefix} True Mask")
    ax[1].imshow(pred_mask, cmap="gray")
    ax[1].set_title(f"{title_prefix} Predicted Mask")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show(block=True)  # keep window open in scripts

import torch

def soft_dice(pred_mask, target, verbose=False):
    """
    Computes soft Dice score between predicted mask and ground truth.

    pred_mask: torch.Tensor, shape [H, W] or [1, H, W], float values in [0, 1]
    target: dict from Mask R-CNN containing 'masks': [N, H, W]
    verbose: bool, print intermediate values

    Returns:
        dice: float
    """
    if target.ndim == 2:   # single mask
        true_mask = target.unsqueeze(0).cpu()  # [1, H, W]
    else:
        true_mask = target.float().cpu()       # [N, H, W]


    # Flatten prediction
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0).cpu()
    pred_flat = pred_mask.contiguous().view(-1).cpu()

    full_true_mask = (true_mask.sum(dim=0) > 0).float()
    true_flat = full_true_mask.contiguous().view(-1)

    # Compute soft dice
    intersection = (pred_flat * true_flat).sum()
    denominator = pred_flat.sum() + true_flat.sum()
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)


    return dice.item()



def binary_iou(pred_mask, true_mask, debug=False):
    pred_mask = to_numpy(pred_mask)
    true_mask = to_numpy(true_mask)

    print(f" Masks shape {pred_mask.shape}, {true_mask.shape}")

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else (1.0 if pred_mask.sum() == 0 else 0.0)

    return iou

def binary_dice(pred_mask, true_mask, debug=True):
    pred_mask = to_numpy(pred_mask)
    true_mask = to_numpy(true_mask)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    dice = (2 * intersection / total) if total != 0 else (1.0 if pred_mask.sum() == 0 else 0.0)


    return dice

# Make sure device is consistent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = get_coco_initialized_model(num_classes=2)
#state = torch.load("./images/frozen_natural_300/natural_frozen_300epochs.pth", map_location=device)  # load directly to device
state = torch.load("../pretrained_final.pth", map_location=device)
model.load_state_dict(state)

model.to(device)  # ensure model is on same device as inputs
model.eval()

base_path = "../recodai-luc-scientific-image-forgery-detection/"

test_dataset = ForgeryDataset(paths['train_authentic'],paths['train_forged'],paths['train_masks'],)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=lambda x: tuple(zip(*x)))

def evaluate_allauth():
    dices = []
    img_size = []
    avg_boxsize = []
    N_boxes = []

    FP_forged_imgs = 0
    FN_forged_imgs = 0
    TP_forged_imgs = 0
    TN_forged_imgs = 0

    true_forged_pixels = 0
    pred_forged_pixels = 0


    for idx, (image, target, filename) in enumerate(train_loader):
        image = image[0]    # take first item from batch
        target = target[0]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        with torch.no_grad():

            if(len(target["boxes"]) == 0 ):
                dice = 1
                dices.append(dice)
                TN_forged_imgs += 1

            elif(len(target["boxes"]) != 0):
                target_mask = combine_resize_submasks(target, raw_img, threshold = None)
                outputs_orig_size = np.zeros(target_mask.shape)
                true_forged_pixels += np.sum(target_mask.cpu().numpy())
                pred_forged_pixels += 0

                dice = 0
                boxes_size = []

                for box in target["boxes"]:
                    size = (box[2]-box[0])*(box[3]-box[1])
                    boxes_size.append(size)

                FN_forged_imgs += 1

                N_boxes.append(len(target["boxes"]))
                img_size.append(raw_img.shape[0]*raw_img.shape[1])
                avg_boxsize.append(np.mean(boxes_size))
                dices.append(dice)

    return np.asarray(dices), np.asarray(avg_boxsize), np.asarray(img_size), np.asarray(N_boxes), np.asarray(pred_forged_pixels/true_forged_pixels), TP_forged_imgs, TN_forged_imgs, FP_forged_imgs, FN_forged_imgs


def evaluate(threshold):
    dices = []
    img_size = []
    avg_boxsize = []
    N_boxes = []

    FP_forged_imgs = 0
    FN_forged_imgs = 0
    TP_forged_imgs = 0
    TN_forged_imgs = 0

    true_forged_pixels = 0
    pred_forged_pixels = 0


    for idx, (image, target, filename) in enumerate(train_loader):
        image = image[0]    # take first item from batch
        target = target[0]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))  # forward pass

            if ((len(outputs[0]["boxes"]) == 0) and (len(target["boxes"]) == 0)):
                TN_forged_imgs += 1
                dice = 1
                dices.append(dice)
            else:
                target_mask = combine_resize_submasks(target, raw_img, threshold = None)
                outputs_orig_size = combine_resize_submasks(outputs[0], raw_img, threshold = threshold)
                true_forged_pixels += np.sum(target_mask.cpu().numpy())
                pred_forged_pixels += np.sum(outputs_orig_size.cpu().numpy())

                dice = soft_dice(outputs_orig_size, target_mask, True)

                boxes_size = []

                for box in target["boxes"]:
                    size = (box[2]-box[0])*(box[3]-box[1])
                    boxes_size.append(size)

                if (len(target["boxes"] == 0) and (len(outputs[0]["boxes"]>0))):
                    FP_forged_imgs += 1
                if (len(target["boxes"] > 0) and (len(outputs[0]["boxes"]==0))):
                    FN_forged_imgs += 1
                if (len(target["boxes"] > 0) and (len(outputs[0]["boxes"]>0))):
                    TP_forged_imgs += 1




                N_boxes.append(len(target["boxes"]))
                img_size.append(raw_img.shape[0]*raw_img.shape[1])
                avg_boxsize.append(np.mean(boxes_size))
                dices.append(dice)

    return np.asarray(dices), np.asarray(avg_boxsize), np.asarray(img_size), np.asarray(N_boxes), np.asarray(pred_forged_pixels/true_forged_pixels), TP_forged_imgs, TN_forged_imgs, FP_forged_imgs, FN_forged_imgs


if __name__ == "__main__":

    """ ONLY PLOTS THE FIRST ELEM IN BATCH """
    """ mean DICE w.r.t threshold
    histogram DICES w.r.t image size
    histogram DICES w.r.t avg submask size
    histogram DICES w.r.t Number submasks

    """
    plot = False

    for thresh in [0.8, 0.85, 0.9, 0.95]:
        dices, avg_boxsize, img_size, N_boxes, pixel_predtrue_proportion, TP_forged_imgs, TN_forged_imgs, FP_forged_imgs, FN_forged_imgs = evaluate(thresh)

        print(" MEAN DICE: ")
        print(np.mean(dices), np.std(dices))

        """plt.scatter(avg_boxsize, dices)
        plt.show()

        plt.scatter(img_size, dices)
        plt.show()

        plt.scatter(N_boxes, dices)
        plt.show()"""


        # align lengths FIRST
        n = min(len(avg_boxsize), len(dices))
        avg_boxsize = avg_boxsize[:n]
        dices = dices[:n]

        # now remove NaNs
        mask = ~np.isnan(avg_boxsize) & ~np.isnan(dices)
        corr = np.corrcoef(avg_boxsize[mask], dices[mask])[0, 1]

        print(corr, np.corrcoef(img_size, dices)[0,1], np.corrcoef(N_boxes, dices)[0,1])
        print(f" Pixel pred/true proportion:{pixel_predtrue_proportion}")
        print(f"FF: {FP_forged_imgs}, FA: {FN_forged_imgs}, TF: {TP_forged_imgs}, TA: {TN_forged_imgs}")


        # Prepare submission
        """submission = {
            "case_id": filename[0],
            "submission": rle_encode([outputs_orig_size])
        }
        """
