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

class ForgeryDataset(Dataset):
    def __init__(self, authentic_path, forged_path, masks_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        # Collect all data samples
        self.samples = []

        # Forged images
        for file in os.listdir(forged_path):
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
            for file in os.listdir(authentic_path):
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

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)  # (H, W, 3)

        # Load mask
        mask = np.load(sample['mask_path'])
        if mask.ndim == 3:
            if mask.shape[0] <= 10:
                mask = np.any(mask, axis=0)
            elif mask.shape[-1] <= 10:
                mask = np.any(mask, axis=-1)
            else:
                raise ValueError(f"Ambiguous 3D mask shape: {mask.shape}")
        mask = (mask > 0).astype(np.uint8)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = F_transforms.to_tensor(image)
            mask = torch.tensor(mask, dtype=torch.uint8)

        return image, mask, self.samples[idx]  # only image and mask

base_path = "../recodai-luc-scientific-image-forgery-detection/"
paths = {
        'train_authentic': os.path.join(base_path, "train_images/authentic"),
        'train_forged': os.path.join(base_path, "train_images/forged"),
        'train_masks': os.path.join(base_path, "train_masks"),
        'test_images': os.path.join(base_path, "test_images")
    }

# Transformations for learning
train_transform = A.Compose([
    A.Resize(256, 256),
    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
