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


def full_mask_from_instance_masks(output, image_info):
    """
    output: dict from MaskRCNN
    image_info: torch tensor (C,H,W) or numpy array (H,W,C) or tuple shape
    """

    # --- extract numeric dims safely ---
    if hasattr(image_info, "shape"):                 # tensor or numpy array
        shape = list(image_info.shape)
    elif isinstance(image_info, (tuple, list)):      # raw tuple passed
        shape = list(image_info)
    else:
        raise ValueError(f"Unsupported image_info type: {type(image_info)}")

    assert len(shape) == 3, f"Unexpected image shape: {shape}"

    # detect channel-first vs channel-last
    if shape[0] in (1, 3):  # (C,H,W)
        C, H, W = shape
    else:                   # (H,W,C)
        H, W, C = shape

    full_mask = torch.zeros((H, W), dtype=torch.uint8)

    boxes = output["boxes"]
    masks = output["masks"]

    for box, mask in zip(boxes, masks):

        # convert box to ints correctly
        x1, y1, x2, y2 = [int(v.item()) for v in box]

        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue

        # ensure mask -> (1,1,h,w)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)

        mask_resized = F.interpolate(
            mask.float(),
            size=(y2 - y1, x2 - x1),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        full_mask[y1:y2, x1:x2] = (mask_resized > 0.5).byte()

    return full_mask



class ForgeryDataset(Dataset):
    def __init__(self, authentic_path, forged_path, masks_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        # Collect all data samples
        self.samples = []

        # Forged images
        for file in sorted(os.listdir(forged_path)):
            img_path = os.path.join(forged_path, file)
            base_name = file.split('.')[0]
            mask_path = os.path.join(masks_path, f"{base_name}.npy")

            self.samples.append({
                'image_path': img_path,
                'mask_path': mask_path,
                'is_forged': True,
                'image_id': base_name
            })

        # Authentic images
        if (authentic_path is not None):
            for file in sorted(os.listdir(authentic_path)):
                img_path = os.path.join(authentic_path, file)
                base_name = file.split('.')[0]
                mask_path = os.path.join(masks_path, f"{base_name}.npy")

                self.samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'is_forged': False,
                    'image_id': base_name
                })

    def __len__(self):
        return len(self.samples)

    def get_raw_img_mask(self, idx):
        sample = self.samples[idx]
        image_raw = Image.open(sample['image_path']).convert('RGB')
        image_raw = np.array(image_raw)  # (H, W, 3)
        mask = np.load(sample['mask_path'])

        print(self.samples[idx]['image_path'])

        return image_raw, mask

    def get_image_props(self, image, mask):
        boxes, labels, masks = self.mask_to_boxes(mask)
        mask_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        mask_whiteness = mask_np.sum() / (image.shape[1] * image.shape[2])

        return {
            "Npixels" : len(image[0])*len(image[1]),
            "WhiteNess" : mask_whiteness
            }

    def get_filename(self, idx):
        return self.samples[idx]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)  # (H, W, 3)

        # Load and process mask
        if os.path.exists(sample['mask_path']):
            mask = np.load(sample['mask_path'])

            # Handle multi-channel masks
            if mask.ndim == 3:
                if mask.shape[0] <= 10:  # channels first (C, H, W)
                    mask = np.any(mask, axis=0)
                elif mask.shape[-1] <= 10:  # channels last (H, W, C)
                    mask = np.any(mask, axis=-1)
                else:
                    raise ValueError(f"Ambiguous 3D mask shape: {mask.shape}")

            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize mask to match image if needed
        H_img, W_img = image.shape[:2]
        if mask.shape != (H_img, W_img):
            print(f"[WARN] pre-resizing mask {mask.shape} -> {(H_img, W_img)}")
            mask = cv2.resize(mask.astype(np.uint8), (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        # Apply transformations
        if self.transform:
            #print("Transforming: ", image.shape, image.dtype, image.min(), image.max())
            transformed = self.transform(image=image, mask=mask)


            image = transformed['image']
            mask = transformed['mask']
            #print("Transformed: ", image.shape, image.dtype, image.min(), image.max())

        else:
            image = F_transforms.to_tensor(image)
            mask = torch.tensor(mask, dtype=torch.uint8)

        # Prepare targets for Mask R-CNN
        if sample['is_forged'] and mask.sum() > 0:
            boxes, labels, masks = self.mask_to_boxes(mask)

            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
        else:
            # For authentic images or images without masks
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'masks': torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }

        return image, target, self.samples[idx]['image_path']  # return filename too

    def mask_to_boxes(self, mask):
        """Convert segmentation mask to bounding boxes for Mask R-CNN"""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)

        # --- FIX: squeeze extra dimensions ---
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()

        # --- FIX: ensure binary + correct type ---
        mask_np = (mask_np > 0).astype(np.uint8)

        # Safety: fall back to empty if shape still wrong
        if mask_np.ndim != 2:
            return (torch.zeros((0,4),dtype=torch.float32),
                    torch.zeros((0,),dtype=torch.int64),
                    torch.zeros((0, *mask_np.shape[-2:]),dtype=torch.uint8))

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        masks = []

        for contour in contours:
            if len(contour) > 0:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small regions
                if w > 5 and h > 5:
                    boxes.append([x, y, x + w, y + h])
                    # Create binary mask for this contour
                    contour_mask = np.zeros_like(mask_np)
                    cv2.fillPoly(contour_mask, [contour], 1)
                    masks.append(contour_mask)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)

        return boxes, labels, masks

base_path = "../recodai-luc-scientific-image-forgery-detection/"
paths = {
        'train_authentic': os.path.join(base_path, "train_images/authentic"),
        'train_forged': os.path.join(base_path, "train_images/forged"),
        'train_masks': os.path.join(base_path, "train_masks"),
        'test_images': os.path.join(base_path, "test_images")
    }

# Transformations for learning
train_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),

    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
