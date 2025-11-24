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

from skimage.measure import find_contours
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

def normalize_mask(mask, n_points=200):
    # mask: tensor [H,W] with 0/1
    """ Finds contour (faster), centers it, aligns major axis with x,
    then normalizes the distances of the list of points to the unit circle """
    mask_np = mask.cpu().numpy()
    contours = find_contours(mask_np, 0.5)

    if len(contours) == 0:
        return None

    pts = max(contours, key=len)          # largest contour

    pts = pts - pts.mean(axis=0)          # center

    pca = PCA(n_components=2)             # align rotation
    pts = pca.fit_transform(pts)

    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts = pts / scale                     # normalize scale

    # resample contour to fixed size
    idx = np.linspace(0, len(pts) - 1, n_points).astype(int)
    pts = pts[idx]

    return pts


def contour_distance(c1, c2):
    if c1 is None or c2 is None:
        return 9999.0
    d1 = cdist(c1, c2).min(axis=1).mean()
    d2 = cdist(c2, c1).min(axis=1).mean()
    return (d1 + d2) / 2

def plot_similar_masks(masks):
    print(f"Found rotated/scaled duplicates: {i} and {j}")
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(masks[i])
    axes[1].imshow(masks[j])
    fig.savefig(f"mask_pair_{i}_{j}.png", bbox_inches='tight', dpi=150)


    time.sleep(2)



forged = 0
non = 0
idx_f = []
images_f = []
mask_f = []

copypaste_count = 0

for idx, (img, target) in enumerate(tqdm(full_dataset)):

    if target['masks'].sum() > 10:
        forged+=1
        idx_f.append(idx)

        """ Check if any two regions are similar """
        boxes = target['boxes']
        masks = target['masks']
        length = len(target['boxes'])

        similar = np.zeros((length,length), np.int8)


        if length > 1:
            normalized = [normalize_mask(masks[k].float()) for k in range(length)]

            for i in range(length):
                for j in range(0,i):
                    if i != j:

                        c1 = normalized[i]
                        c2 = normalized[j]

                        dist = contour_distance(c1, c2)

                        if dist < 0.05:   # threshold: <0.05 → almost identical shapes
                            similar[i, j] = 1
                            #plot_similar_masks(masks)
        if np.any(similar):
            copypaste_count +=1
    else:
        non+=1

print(f"forged {forged}, non-forged: {non}")
print(f"copy_paste: {copypaste_count}")
#print(f"forged indices: {idx_f}")
