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

def soft_dice(pred_mask, target):
    """
    Computes soft Dice score between predicted mask and ground truth.

    pred_mask: torch.Tensor, shape [H, W] or [1, H, W], float values in [0, 1]
    target: dict from Mask R-CNN containing 'masks': [N, H, W]

    Returns:
        dice: float
    """

    # Flatten prediction
    if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
        pred_mask = pred_mask.squeeze(0)
    pred_flat = pred_mask.contiguous().view(-1)

    # Combine all instance masks into a single mask
    true_mask = target['masks'].float()  # [N, H, W]
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

def evaluate_segmentation(model, dataloader, device, firstN = None, threshold=0.5, debug=False):
    """  """

    model.eval()
    iou_scores = []
    dice_scores = []
    properties = defaultdict(list)

    with torch.no_grad():
        for idx, (images, targets, _) in enumerate(tqdm(dataloader, desc="Evaluating", disable = debug)):
            if firstN is not None and idx == firstN:
                break

            raw_image, raw_mask = full_dataset.get_raw_img_mask(idx)

            images = [img.to(device) for img in images]
            output = model(images)

            true_mask = full_mask_from_instance_masks(targets[0], raw_image.shape)
            pred_mask = full_mask_from_instance_masks(output[0], raw_image.shape)
            #plot_masks(true_mask, pred_mask)

            """
            # Fetch original untransformed image + mask using the dataset index
            dataset_index = int(targets[0]['image_id'].item())
            raw_img, raw_mask = dataloader.get_raw_img_mask(idx)

            # Save properties for later correlation analysis
            prop = dataloader.get_image_props(raw_img, raw_mask)
            for key, value in prop.items():
                properties[key].append(value)
                #print(properties)
            """
            iou_scores.append(binary_iou(pred_mask.cpu().numpy(), true_mask.cpu().numpy()))
            dice_scores.append(binary_dice(pred_mask.cpu().numpy(), true_mask.cpu().numpy()))

            if debug:
                print(f"Image {idx} with size: {properties['Npixels'][-1]} and whiteness {properties['WhiteNess'][-1]}")


    return iou_scores, dice_scores, properties

if __name__ == "__main__":
    """ EVALUATE THE IOU AND DICE """
    model = create_light_mask_rcnn()                 # create model
    model.load_state_dict(torch.load("mask_rcnn_best.pth", map_location="cpu"))
    model.eval()
    model.to(device)

    base_path = "../recodai-luc-scientific-image-forgery-detection/"
    test_dataset = ForgeryDataset(
        None,
        os.path.join(base_path, "supplemental_images"),
        os.path.join(base_path, "supplemental_masks"),
        transform=train_transform
    )
    test_dataset = Subset(train_subset, list(range(2)))


    # Creating dataloaders
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    print(f"Train samples: {len(test_dataset)}")


    iou, dice, props = evaluate_segmentation(model, test_loader, device)

    mean_iou = np.mean(iou) if iou else 0.0
    mean_dice = np.mean(dice) if dice else 0.0

    print(f"\nMean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")

    sizes = props["Npixels"]
    wn = props["WhiteNess"]

    """
    print(sizes)
    plt.scatter(sizes, iou)
    plt.scatter(sizes, dice)
    plt.show()

    plt.scatter(wn, iou)
    plt.scatter(wn, dice)
    plt.show()"""
