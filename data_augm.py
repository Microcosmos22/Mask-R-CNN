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


full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    transform=train_transform
)


forged = 0
non = 0
idx_f = []
images_f = []
mask_f = []

for idx, (img, target) in enumerate(tqdm(full_dataset)):
    if target['masks'].sum() > 10:
        forged+=1
        idx_f.append(idx)
        images_f.append(img)
        mask_f.append(target['masks'])
    else:
        non+=1

print(f"forged {forged}, non-forged: {non}")
#print(f"forged indices: {idx_f}")
