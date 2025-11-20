import numpy as np
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

from dataloader import *

def binary_iou(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return 1.0 if pred_mask.sum() == 0 else 0.0
    return intersection / union


def binary_dice(pred_mask, true_mask):
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()

    if total == 0:
        return 1.0 if pred_mask.sum() == 0 else 0.0
    return 2 * intersection / total

def evaluate_segmentation(model, dataloader, device, threshold=0.5):
    model.eval()
    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            predictions = model(images)

            for pred, target in zip(predictions, targets):

                # GT mask (combine multiple objects)
                true_mask = torch.zeros_like(images[0][0], dtype=torch.uint8)
                for gt_mask in target['masks']:
                    true_mask = torch.max(true_mask, gt_mask.cpu())

                # Predicted mask
                if len(pred['masks']) == 0:
                    pred_mask = torch.zeros_like(true_mask)
                else:
                    # Combine predicted instance masks via max
                    pred_mask = torch.max(pred['masks'].squeeze(1), dim=0)[0]
                    pred_mask = (pred_mask.cpu().numpy() > threshold).astype(np.uint8)

                true_mask_np = true_mask.numpy()

                iou_scores.append(binary_iou(pred_mask, true_mask_np))
                dice_scores.append(binary_dice(pred_mask, true_mask_np))

    return np.mean(iou_scores), np.mean(dice_scores)
