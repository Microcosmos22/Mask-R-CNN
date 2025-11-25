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

def extract_region_from_box(img, full_mask, box):
    # img: C,H,W
    # full_mask: H,W
    # box: [xmin, ymin, xmax, ymax]

    xmin, ymin, xmax, ymax = box.int().tolist()

    crop_img  = img[:, ymin:ymax, xmin:xmax]
    crop_mask = full_mask[ymin:ymax, xmin:xmax]

    return crop_img, crop_mask, (xmin, ymin)

def tighten_mask(crop_img, crop_mask):
    y, x = np.where(crop_mask.cpu().numpy() > 0)
    ymin, ymax = y.min(), y.max()
    xmin, xmax = x.min(), x.max()

    crop_img  = crop_img[:, ymin:ymax+1, xmin:xmax+1]
    crop_mask = crop_mask[ymin:ymax+1, xmin:xmax+1]

    return crop_img, crop_mask

import torchvision.transforms.functional as TF
import random

def transform_region(crop_img, crop_mask):
    angle = random.uniform(-180, 180)
    scale = random.uniform(0.7, 1.3)

    crop_img_t = TF.affine(crop_img, angle=angle, translate=[0,0],
                           scale=scale, shear=[0,0])
    crop_mask_t = TF.affine(crop_mask.unsqueeze(0), angle=angle,
                            translate=[0,0], scale=scale, shear=[0,0])
    crop_mask_t = (crop_mask_t.squeeze(0) > 0.5).float()

    return crop_img_t, crop_mask_t

def paste_region(img, mask, crop_img, crop_mask, top, left):
    C,H,W = img.shape
    h, w = crop_mask.shape

    img[:, top:top+h, left:left+w] = (
        img[:, top:top+h, left:left+w] * (1 - crop_mask) + crop_img * crop_mask
    )

    mask_new = mask.clone()
    mask_new[top:top+h, left:left+w] = torch.logical_or(
        mask_new[top:top+h, left:left+w], crop_mask
    ).float()

    return img, mask_new



def find_copypaste_forgeries():
    similar_matrices = []

    forged = 0
    non = 0
    idx_f = []
    images_f = []
    mask_f = []
    copymove_idx = []

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

                                """ Two similar sub-masks  """
                                #plot_similar_masks(masks)
            if np.any(similar):
                copypaste_count +=1
                """ Its a copy- move forgery """
                copymove_idx.append(idx)
        else:
            non+=1
        similar_matrices.append(similar)

    return copymove_idx, copypaste_count, similar_matrices

def gen_copypaste_aug(copymove_idx, similar_matrices):

    for idx in copymove_idx:
        image, target = full_dataset[idx]
        boxes = target['boxes']
        masks = target['masks']
        length = len(target['boxes'])
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        """ there is similar regions """
        similar = np.asarray(similar_matrices[idx])
        indices = np.argwhere(similar == 1)  # works for any shape, returns list of coords
        if similar.ndim == 2:
            i, j = indices[0]

            box = target['boxes'][i] # x0, y0, x1, y1
            mask = target['masks'][i]   # mask aligned with box

            print(f"Mask size: {raw_mask.shape}, submask: {mask.shape}")

            # 1. extract from full image using box
            crop_img, crop_mask, (xmin, ymin) = extract_region_from_box(
                img, full_mask, box
            )

            # 2. tighten mask inside the box
            crop_img, crop_mask = tighten_mask(crop_img, crop_mask)

            # 3. transform
            crop_img2, crop_mask2 = transform_region(crop_img, crop_mask)

            # 4. pick new position
            H,W = img.shape[1:]
            h,w = crop_mask2.shape
            top  = random.randint(0, H - h)
            left = random.randint(0, W - w)

            # 5. paste
            aug_img, aug_mask = paste_region(
                img.clone(),
                target['masks'].sum(dim=0).clone(),
                crop_img2,
                crop_mask2,
                top,
                left
            )
"""
copymove_idx, copymove_count, similar_matrices = find_copypaste_forgeries()
copymove = {
    "copymove_idx": copymove_idx,  # plain list
    "similar_matrices": [sm.tolist() for sm in similar_matrices]
}

with open("copymove.json", "w") as f:
    json.dump(copymove, f, indent=4)
"""
with open("copymove.json", "r") as f:
    copymove = json.load(f)
gen_copypaste_aug(copymove['copymove_idx'], copymove['similar_matrices'])

#print(f"forged {forged}, non-forged: {non}")
#print(f"copy_paste: {copypaste_count}")
#print(f"forged indices: {idx_f}")
