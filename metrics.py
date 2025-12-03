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
import torch

def binary_iou(pred_mask, true_mask, threshold=0.5, debug=False):
    """
    Computes IoU between predicted mask and ground truth mask.

    Args:
        pred_mask: torch.Tensor or np.array, logits or probabilities
        true_mask: torch.Tensor or np.array, binary mask (0/1)
        threshold: float, value above which a pixel is considered foreground
        debug: bool, print shapes

    Returns:
        float: IoU score
    """

    # If pred_mask contains logits, convert to probabilities then binarize
    if pred_mask.dtype != np.bool:
        pred_mask = (pred_mask > threshold)

    true_mask = true_mask.astype(bool)

    if debug:
        print(f"Masks shape {pred_mask.shape}, {true_mask.shape}")

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else (1.0 if pred_mask.sum() == 0 else 0.0)

    return iou


def binary_dice(pred_logits, true_mask, threshold=0.5):
    pred_logits = torch.from_numpy(pred_logits)
    true_mask = torch.from_numpy(true_mask)
    # convert logits to probabilities
    pred_prob = torch.sigmoid(pred_logits)
    # binarize predictions
    pred_bin = (pred_prob > threshold).float()
    true_mask = true_mask.float()

    intersection = (pred_bin * true_mask).sum()
    total = pred_bin.sum() + true_mask.sum()
    dice = (2 * intersection / total) if total > 0 else 1.0

    return dice

def soft_dice(pred_logits, true_mask, eps=1e-6):
    pred_prob = torch.sigmoid(pred_logits)
    true_mask = true_mask.float()
    intersection = (pred_prob * true_mask).sum()
    total = pred_prob.sum() + true_mask.sum()
    dice = 2 * intersection / (total + eps)
    return 1 - dice  # return 1-dice as a loss term
