import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
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


def binary_iou(pred_mask, true_mask, debug=False):
    pred_mask = to_numpy(pred_mask)
    true_mask = to_numpy(true_mask)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else (1.0 if pred_mask.sum() == 0 else 0.0)

    if debug and true_mask.sum()!=0:
        print("\n==== IOU DEBUG ====")
        print("Pred positives:", pred_mask.sum())
        print("True positives:", true_mask.sum())
        print("Intersection:", intersection)
        print("Union:", union)
        print("IoU:", iou)
        plot_masks(true_mask, pred_mask, title_prefix="Debug", save_path="mask_debug.png")


    return iou

def binary_dice(pred_mask, true_mask, debug=True):
    pred_mask = to_numpy(pred_mask)
    true_mask = to_numpy(true_mask)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()
    dice = (2 * intersection / total) if total != 0 else (1.0 if pred_mask.sum() == 0 else 0.0)

    print(true_mask.sum())
    if debug and true_mask.sum()!=0:
        print("\n==== DICE DEBUG ====")
        print("Pred positives:", pred_mask.sum())
        print("Whiteness:", true_mask.sum()/65536*100)
        print("DICE: ", dice)

        print("Height:", true_mask.shape[1])
        print("Width:", true_mask.shape[0])
        plot_masks(true_mask, pred_mask, title_prefix="Debug", save_path="mask_debug.png")


    return dice

def evaluate_segmentation(model, dataloader, device, threshold=0.5, debug=False):
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                # Ground truth mask (combine all objects)
                true_mask = torch.zeros_like(images[0][0], dtype=torch.uint8, device=device)
                for gt_mask in target['masks']:
                    true_mask = torch.max(true_mask, gt_mask.to(device))


                # Predicted mask (combine all instances)
                if len(pred['masks']) == 0:
                    pred_mask_np = torch.zeros_like(true_mask).cpu().numpy()
                else:
                    pred_mask = torch.max(pred['masks'].squeeze(1), dim=0)[0]
                    pred_mask_np = (pred_mask.cpu().numpy() > threshold).astype(np.uint8)

                true_mask_np = true_mask.cpu().numpy()

                iou_scores.append(binary_iou(pred_mask_np, true_mask_np, False))
                dice_scores.append(binary_dice(pred_mask_np, true_mask_np, True))

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0

    print(f"\nMean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
    return mean_iou, mean_dice

if __name__ == "__main__":
    model = create_light_mask_rcnn()                 # create model
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    model.to(device)

    base_path = "../recodai-luc-scientific-image-forgery-detection/"
    test_dataset = ForgeryDataset(
        None,
        os.path.join(base_path, "supplemental_images"),
        os.path.join(base_path, "supplemental_masks"),
        transform=train_transform
    )
    #test_dataset = Subset(test_dataset, list(range(400)))

    # Creating dataloaders
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    print(f"Train samples: {len(test_dataset)}")


    val_iou, val_dice = evaluate_segmentation(model, test_loader, device)
    print(f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")
