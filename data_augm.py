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
train_transform = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.Affine(
        scale=(0.95, 1.05),
        rotate=(-15, 15),          # mild rotation
        translate_percent=(-0.02, 0.02),
        fit_output=True,            # avoid cutting off image
        p=0.7
    ),
    A.RandomCrop(width=224, height=224, p=0.3),  # forces network to learn partial views

    # Photometric
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),

    # Post-processing / compression
    A.ImageCompression(quality_lower=40, quality_upper=100, p=0.3),

    ToTensorV2()
])

image_a = os.path.join(base_path, "train_images/authentic")
image_f = os.path.join(base_path, "train_images/forged")
mask_dir = os.path.join(base_path, "train_masks")

image_a_out = os.path.join(base_path, "train_images/authentic")
image_f_out = os.path.join(base_path, "train_images/forged")
mask_dir_out = os.path.join(base_path, "train_masks")

print(image_a)
print(image_f)
print(mask_dir)


for fname in os.listdir(image_f):
    """ forged images """

    img  = cv2.imread(os.path.join(image_f, fname))
    mask = np.load(os.path.join(mask_dir, fname.replace(".png", ".npy")))
    mask = np.transpose(mask, (1, 2, 0))

    print("IMG:", img.shape, "MASK:", mask.shape)



    for i in range(1):  # number of new copies
        augmented = augment(image=img, mask=mask)
        aug_img  = augmented["image"]
        aug_mask = augmented["mask"]

        img_new_name = fname.replace(".png", f"_aug{i}.png")
        mask_new_name = fname.replace(".png", f"_aug{i}.npy")

        print(os.path.join(image_f_out, img_new_name))
        print(os.path.join(mask_dir_out, mask_new_name))

        cv2.imwrite(os.path.join(image_f_out, img_new_name), aug_img)
        #cv2.imwrite(, aug_mask)
        np.save(os.path.join(mask_dir_out, mask_new_name), np.transpose(aug_mask, (2,0,1)))

for fname in os.listdir(image_a):
    """ forged images """
    img  = cv2.imread(os.path.join(image_a, fname))

    for i in range(1):  # number of new copies
        augmented = augment(image=img)
        aug_img  = augmented["image"]

        new_name = fname.replace(".png", f"_aug{i}.png")
        cv2.imwrite(os.path.join(image_a_out, new_name), aug_img)
