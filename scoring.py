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
from metrics import *


def inv_transform(output_mask, image_shape):
    """
    output: dict from MaskRCNN
    image_shape: network input shape, either (H, W, C) or (C, H, W)
    H_orig, W_orig: original image size
    """


    # Handle both orderings
    if len(image_shape.shape) == 3:
        if image_shape.shape[0] in [1, 3]:  # likely (C, H, W)
            C, H, W = image_shape.shape
        else:  # likely (H, W, C)
            H, W, C = image_shape.shape
    else:
        raise ValueError(f"Unexpected image_shape: {image_shape}")
        print(len(image_shape.shape), len(image_shape), image_shape.shape)


    # Resize full mask to original image size
    full_mask_resized = F.interpolate(
        output_mask.float(),
        size=(H, W),
        mode='nearest'
    )[0, 0].byte()

    return full_mask_resized

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
    plot= False

    for idx, (image, target, filename) in enumerate(train_loader):
        """ skip authentic images """
        """if (len(target[0]['boxes']) == 0):
            continue"""
        # a loader with collate_fn returns batches of lists
        image = image[0]           # take first item from batch
        target = target[0]

        with torch.no_grad():
            image = image.unsqueeze(0).to(device)
            outputs = model(image)   # must be list
            image = image.squeeze(0).permute(1,2,0)
            """ Plot image, mask_pred and mask_true"""
            #full_pred_mask = full_mask_from_instance_masks(outputs[0], raw_image.shape)  # shape = network input (H_net, W_net)
            # pred_mask is (H_net, W_net)


        outputs_orig_size = inv_transform(outputs, image)

        pred = torch.from_numpy(outputs_orig_size.cpu().numpy()).float()
        #true_mask = torch.from_numpy(target).float()

        dice = soft_dice(pred, target)

        print(f"\nIdx: {idx} Dice: {dice:.4f}")

        if plot:
            fig, ax = plt.subplots(2)
            ax[0].imshow(image)
            ax[0].imshow(target, alpha=0.5)

            ax[1].imshow(outputs_orig_size.squeeze(0).squeeze(0))
            plt.show()



        """ Convert to numpy and encode """
        #print(filename[0]['image_id'])
        submission = {
            "case_id": filename[0]['image_id'],
            "submission": rle_encode([outputs_orig_size.numpy()])
        }

        #print(rle_encode([outputs_orig_size.numpy()]))


        #rle = rle_encode(full_pred_mask_resized.numpy())
        #print(f"rle encoded mask: {rle}")
