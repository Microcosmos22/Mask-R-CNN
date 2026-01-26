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
from torchvision.ops import box_iou

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

def of1_score(pred_mask: np.ndarray, gt_mask: np.ndarray, n_pred: int, n_gt: int):
    pred_flat = pred_mask.cpu().numpy().flatten()
    gt_flat = gt_mask.cpu().numpy().flatten()

    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    penalty = n_gt / max(n_pred, n_gt) # Penalty for predicting more submasks than there are gt masks

    if (precision + recall) > 0:
        return 2 * penalty * (precision * recall) / (precision + recall), recall, precision
    else:
        return 0, recall

from torchvision.ops import box_iou

def debug_box_scores(pred, gt, iou_thr=0.8, score_thr=0.6):
    pred_boxes = pred["boxes"].cpu()
    gt_boxes = gt["boxes"].cpu()

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return None

    ious = box_iou(gt_boxes, pred_boxes)  # [G, P]
    scores = pred["scores"].cpu()          # <--- fix here

    best_iou, best_idx = ious.max(dim=1)
    best_scores = scores[best_idx]

    gt_covered = (best_iou >= iou_thr) & (best_scores >= score_thr)
    high_score_gt_recall = gt_covered.float().mean().item()


    # False positives = preds not matched to any GT
    matched_preds = best_idx.unique()
    fp_mask = torch.ones(len(scores), dtype=torch.bool)
    fp_mask[matched_preds] = False

    mean_fp_score = scores[fp_mask].mean().item() if fp_mask.any() else 0.0
    tp_mask = best_iou >= iou_thr
    mean_tp_score = best_scores[tp_mask].mean().item() if tp_mask.any() else 0.0


    return {
        "gt_high_score_recall": high_score_gt_recall,
        "mean_tp_score": mean_tp_score,
        "mean_fp_score": mean_fp_score,
        "num_preds": len(scores),
        "num_gt": len(gt["boxes"])
    }



def evaluate_allauthentic(loader):
    oF1s = []
    img_size = []
    avg_boxsize = []
    N_boxes = []

    true_forged_pixels = 0
    pred_forged_pixels = 0

    for idx, (image, target, filename) in enumerate(tqdm(loader)):
        image = image[0]    # take first item from batch
        target = target[0]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        with torch.no_grad():

            if ((len(target["boxes"]) == 0)):
                oF1s.append(1)
                print("Correct: Authentic")
            elif((len(target["boxes"]) != 0)):
                oF1s.append(0)
                print("Wrong: Forged")

    return np.asarray(oF1s)

def evaluate(loader, threshold):
    oF1s = []
    img_size = []
    avg_boxsize = []
    N_boxes = []
    recalls = []
    precisions = []

    true_forged_pixels = 0
    pred_forged_pixels = 0

    box_overlap_scores = {
        "gt_high_score_recall": [],
        "mean_tp_score": [],
        "mean_fp_score": [],
        "num_preds": [],
        "num_gt": [],
    }

    for image, target, filename, idxs in tqdm(loader):
        image = image[0]    # take first item from batch
        target = target[0]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idxs[0])

        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))  # forward pass

            if ((len(outputs[0]["boxes"]) == 0) and (len(target["boxes"]) == 0)):
                oF1s.append(1)
            elif((len(target["boxes"]) == 0) and (len(outputs[0]["boxes"]) != 0)):
                oF1s.append(0)
            elif((len(target["boxes"]) != 0) and (len(outputs[0]["boxes"]) == 0)):
                oF1s.append(0)
            else:
                # Force submasks into place, gets rid of the box errors. If oF1 improves, box regression is largest ERROR
                #outputs[0]["boxes"]  = target["boxes"]
                #outputs[0]["labels"] = target["labels"]
                scores = outputs[0]['scores']

                gt_mask = combine_resize_submasks(target, raw_img, threshold = None)
                pred_mask = combine_resize_submasks(outputs[0], raw_img, threshold = threshold)

                out = debug_box_scores(outputs[0], target, iou_thr = 0.8, score_thr = threshold)

                if out is None:
                    continue
                else:
                    for (k,v) in out.items():
                        box_overlap_scores[k].append(v)


                oF1, recall, precision = of1_score(pred_mask, gt_mask, len(outputs[0]["boxes"]), len(target["boxes"]))

                boxes_size = []

                for box in target["boxes"]:
                    size = (box[2]-box[0])*(box[3]-box[1])
                    boxes_size.append(size)

                N_boxes.append(len(target["boxes"]))
                img_size.append(raw_img.shape[0]*raw_img.shape[1])
                avg_boxsize.append(np.mean(boxes_size))
                oF1s.append(oF1)
                recalls.append(recall)
                precisions.append(precision)

    return box_overlap_scores, np.asarray(oF1s), np.asarray(avg_boxsize), np.asarray(img_size), np.asarray(N_boxes), np.asarray(recalls), np.asarray(precision)

def corr_oF1_avgboxsize(oF1s, avg_boxsize):

    # align lengths FIRST
    n = min(len(avg_boxsize), len(oF1s))
    avg_boxsize = avg_boxsize[:n]
    oF1s = oF1s[:n]

    # now remove NaNs
    mask = ~np.isnan(avg_boxsize) & ~np.isnan(oF1s)
    corr = np.corrcoef(avg_boxsize[mask], oF1s[mask])[0, 1]


if __name__ == "__main__":

    # Make sure device is consistent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = get_coco_initialized_model(num_classes=2)
    #state = torch.load("./images/frozen_natural_300/natural_frozen_300epochs.pth", map_location=device)  # load directly to device
    state = torch.load("../pretrained_final.pth", map_location=device)
    model.load_state_dict(state)

    model.to(device)  # ensure model is on same device as inputs
    model.eval()

    """ ONLY PLOTS THE FIRST ELEM IN BATCH """
    """ mean DICE w.r.t threshold
    histogram oF1s w.r.t image size
    histogram oF1s w.r.t avg submask size
    histogram oF1s w.r.t Number submasks
    """
    plot = False
    # For all authentic prediction, compute the oF1 scores in the train and test dataset
    #oF1s = evaluate_allauthentic(train_loader)
    #print(" Baseline oF1s TRAIN: ")
    #print(np.mean(oF1s), np.std(oF1s))
    """
     Baseline oF1s TRAIN:
     0.404 +- 0.490
    """
    #oF1s = evaluate_allauthentic(test_loader)
    #print(" Baseline oF1s TEST: ")
    #print(np.mean(oF1s), np.std(oF1s))



    for thresh in [0.6, 0.8]:
        """# MEAN oF1s:
        0.21066425060809993 0.24159184788399646
        RECALL:"""
        box_overlap_scores, oF1s, avg_boxsize, img_size, N_boxes, recalls = evaluate(train_loader, thresh)

        print(" MEAN oF1s: ")
        print(np.mean(oF1s), np.std(oF1s))
        print(" RECALL:")
        print(np.mean(recalls), np.std(recalls))
        for (k, v) in box_overlap_scores.items():
            print(f"{k}: {np.mean(v)} +- {np.std(v)}")


        # corr = corr_oF1_avgboxsize(oF1s, avg_boxsize)
        #print(f"Corr oF1 w avg_boxsize: {corr:.4f} \n oF1 w img_size: {np.corrcoef(img_size, oF1s)[0,1]:.4f} \n oF1 w N_boxes: {np.corrcoef(N_boxes, oF1s)[0,1]:.4f}")

        #print(f"FP: {FP_forged_imgs}, FN: {FN_forged_imgs}, TP: {TP_forged_imgs}, TN: {TN_forged_imgs}")
