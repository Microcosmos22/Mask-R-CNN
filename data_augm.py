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

from dataloader import *

paths = {
        'train_authentic': os.path.join(base_path, "train_images/authentic"),
        'train_forged': os.path.join(base_path, "train_images/forged"),
        'train_masks': os.path.join(base_path, "train_masks"),
        'test_images': os.path.join(base_path, "test_images")
    }

print()
print(base_path)
full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    transform=train_transform
)

""" ####################### """
import albumentations as A
import cv2
import os

# Augmentation pipeline — choose meaningful transformations
augment = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.Affine(
        scale=(0.95, 1.05),
        rotate=(-15, 15),          # mild rotation
        translate_percent=(-0.02, 0.02),
        fit_output=False,            # avoid cutting off image
        p=0.7
    ),
    # Photometric
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    A.GaussNoise(var_limit=(1.0, 4.0), p=0.3),

    # Post-processing / compression
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),

    ToTensorV2()
])

base_path = r"C:\Users\PC\Documents\Image-forgery\recodai-luc-scientific-image-forgery-detection"

image_a = os.path.join(base_path, "train_images", "authentic")
image_f = os.path.join(base_path, "train_images", "forged")
mask_dir = os.path.join(base_path, "train_masks")

image_a_out = os.path.join(base_path, "train_images", "authentic")
image_f_out = os.path.join(base_path, "train_images", "forged")
mask_dir_out = os.path.join(base_path, "train_masks")

print(image_a)

print(image_a)
print(image_f)
print(mask_dir)


for fname in os.listdir(image_f):
    """ forged images """

    # Load image
    image = Image.open(os.path.join(image_f, fname)).convert('RGB')
    img = np.array(image)  # (H, W, 3)

    # Load mask
    mask = np.load(os.path.join(mask_dir, fname.replace(".png", ".npy")))
    if mask.ndim == 3:
        if mask.shape[0] <= 10:
            mask = np.any(mask, axis=0)
        elif mask.shape[-1] <= 10:
            mask = np.any(mask, axis=-1)
        else:
            raise ValueError(f"Ambiguous 3D mask shape: {mask.shape}")
    mask = (mask > 0).astype(np.uint8)


    #img  = cv2.imread(os.path.join(image_f, fname))
    #mask = np.load(os.path.join(mask_dir, fname.replace(".png", ".npy")))
    #mask = np.transpose(mask, (1, 2, 0))

    print(os.path.join(image_f, fname))
    print("IMG:", img.shape, "MASK:", mask.shape)



    for i in range(1):  # number of new copies
        augmented = augment(image=img, mask=mask)
        aug_img  = augmented["image"]
        aug_mask = augmented["mask"]

        img_new_name = fname.replace(".png", f"_aug{i}.png")
        mask_new_name = fname.replace(".png", f"_aug{i}.npy")

        # aug_img is a tensor from ToTensorV2()
        if isinstance(aug_img, torch.Tensor):
            # (C,H,W) -> (H,W,C)
            aug_img = aug_img.permute(1,2,0).cpu().numpy()
            # scale to 0-255 if needed
            aug_img = (aug_img * 255).astype(np.uint8)
            # ensure it's BGR for cv2
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(image_f_out, img_new_name), aug_img)
        #cv2.imwrite(, aug_mask)
        # if mask is 2D, add channel dimension
        if aug_mask.ndim == 2:
            aug_mask = np.expand_dims(aug_mask, axis=0)  # (1, H, W)

        # now you can save
        np.save(os.path.join(mask_dir_out, mask_new_name), aug_mask)

for fname in os.listdir(image_a):
    """ forged images """
    img  = cv2.imread(os.path.join(image_a, fname))

    for i in range(1):  # number of new copies
        augmented = augment(image=img)
        aug_img  = augmented["image"]

        # convert tensor to HWC uint8
        if isinstance(aug_img, torch.Tensor):
            aug_img = aug_img.permute(1,2,0).cpu().numpy()      # C,H,W -> H,W,C
            aug_img = (aug_img * 255).astype(np.uint8)          # if normalized 0-1
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR) # ensure BGR for cv2

        new_name = fname.replace(".png", f"_aug{i}.png")
        cv2.imwrite(os.path.join(image_a_out, new_name), aug_img)
