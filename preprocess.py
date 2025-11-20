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

def get_unique_sizes(directory):
    size_counts = defaultdict(int)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', 'JPG')):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        size = img.size
                        size_counts[size] += 1
                except Exception as e:
                    print(f"Error {file}: {e}")

    return size_counts

folders = [
    "../recodai-luc-scientific-image-forgery-detection/train_images/authentic",
    "../recodai-luc-scientific-image-forgery-detection/train_images/forged",
    "../recodai-luc-scientific-image-forgery-detection/test_images"
]

def print_image_sizes():

    for folder in folders:
        print(f"\n📂 Folder: {folder}")
        sizes = get_unique_sizes(folder)

        if not sizes:
            print("No images or mistake in code")
            continue

        sorted_sizes = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

        print("┌───────────────┬───────────────┬─────────┐")
        print("│  Width (px)  │ Height (px) │ Quantity │")
        print("├───────────────┼───────────────┼─────────┤")
        for (w, h), count in sorted_sizes:
            print(f"│ {w:<13} │ {h:<13} │ {count:<7} │")
        print("└───────────────┴───────────────┴─────────┘")

def analyze_data_structure():
    base_path = '../recodai-luc-scientific-image-forgery-detection'

    # Checking train images
    train_authentic_path = os.path.join(base_path, 'train_images/authentic')
    train_forged_path = os.path.join(base_path, 'train_images/forged')
    train_masks_path = os.path.join(base_path, 'train_masks')
    test_images_path = os.path.join(base_path, 'test_images')

    print(f"Authentic images: {len(os.listdir(train_authentic_path))}")
    print(f"Forged images: {len(os.listdir(train_forged_path))}")
    print(f"Masks: {len(os.listdir(train_masks_path))}")
    print(f"Test images: {len(os.listdir(test_images_path))}")

    # Let's analyze some examples of masks
    mask_files = os.listdir(train_masks_path)[:5]
    print(f"Examples of mask files: {mask_files}")

    # Checking the mask format
    sample_mask = np.load(os.path.join(train_masks_path, mask_files[0]))
    print(f"Mask format: {sample_mask.shape}, dtype: {sample_mask.dtype}")

    test_files = os.listdir(test_images_path)
    print(f"Test images: {test_files}")

    return {
        'train_authentic': train_authentic_path,
        'train_forged': train_forged_path,
        'train_masks': train_masks_path,
        'test_images': test_images_path
    }

def visualize():

    # Visualize authentic images
    authentic_files = sorted(os.listdir(paths['train_authentic']))[:num_samples]
    forged_files = sorted(os.listdir(paths['train_forged']))[:num_samples]
    mask_files = sorted(os.listdir(paths['train_masks']))[:num_samples]

    fig, axes = plt.subplots(3, num_samples, figsize=(15, 10))

    # Authentic images
    for i, file in enumerate(authentic_files):
        img_path = os.path.join(paths['train_authentic'], file)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Authentic: {file}')
        axes[0, i].axis('off')

    # Forged images
    for i, file in enumerate(forged_files):
        img_path = os.path.join(paths['train_forged'], file)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'Forged: {file}')
        axes[1, i].axis('off')

    # Masks
    for i, file in enumerate(mask_files):
        mask_path = os.path.join(paths['train_masks'], file)
        mask = np.load(mask_path)
        mask = np.squeeze(mask)
        axes[2, i].imshow(mask, cmap='gray')
        axes[2, i].set_title(f'Mask: {file}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_batch_samples(dataloader, model=None, device=device):
    images, targets = next(iter(dataloader))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(min(4, len(images))):
        # Original image
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # denormalize
        img = np.clip(img, 0, 1)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i}')
        axes[0, i].axis('off')

        # Mask
        mask = torch.zeros_like(images[i][0])
        for target_mask in targets[i]['masks']:
            mask = torch.max(mask, target_mask.cpu())

        axes[1, i].imshow(mask, cmap='hot')
        axes[1, i].set_title(f'Mask {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
