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

from edarnn import *
from dataloader import *
import numpy as np

def plot_errors_parameters():


    with open("losses.json","r") as f:
        cverrors = json.load(f)
    errors = np.array(cverrors["errors"])   # shape = (num_paramsets, 5, 4)

    # sum across folds (axis=1)
    summed = errors.sum(axis=1)         # shape = (num_paramsets, 4)

    print(summed.shape)

    cverrors['train'] = summed[:,0]
    cverrors['val'] = summed[:,1]
    cverrors['IoU'] = summed[:,2]
    cverrors['DICE'] = summed[:,3]
    cverrors= pd.DataFrame(cverrors)
    cverrors = cverrors.drop(columns=["errors"])

    cverrors = cverrors.drop(columns=["IoU"])
    cverrors = cverrors.drop(columns=["params"])

    print(cverrors)
    cverrors.head()



    plt.figure(figsize=(5,4))
    plt.title("Train, val, IOU and DICE for 3 diff model backbones")
    plt.imshow(cverrors, cmap="viridis")
    plt.colorbar(label="error magnitude")

    plt.xticks(range(4), [f"E{k}" for k in range(1,5)])
    plt.yticks(range(3), [f"param {i}" for i in range(3)])

    plt.tight_layout()
    plt.show()

def plot_errors_folds():
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    with open("losses.json","r") as f:
        data = json.load(f)

    errors = np.array(data["errors"], dtype=float)   # (3 paramsets, 5 folds, 4 metrics)

    num_params = errors.shape[0]
    num_folds  = errors.shape[1]
    num_metrics = errors.shape[2]

    fig, axes = plt.subplots(1, num_params, figsize=(14, 4), sharey=True)
    fig.suptitle("Folds × Metrics for Each Parameter Set", fontsize=14)

    # global vmin/vmax for consistent color scaling
    vmin = errors.min()
    vmax = errors.max()

    for p in range(num_params):
        ax = axes[p]
        mat = errors[p]      # (5 × 4)

        im = ax.imshow(mat, cmap="magma", vmin=vmin, vmax=vmax)

        ax.set_title(f"Param set {p}")
        ax.set_xticks(range(num_metrics))
        ax.set_yticks(range(num_folds))

        ax.set_xticklabels(["Train", "Val", "IoU", "DICE"], rotation=45)
        ax.set_yticklabels([f"F{i}" for i in range(num_folds)])

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label("Error magnitude", fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_eval_examples():
    """ plots single masks """
    
    model = create_light_mask_rcnn()
    state = torch.load("best_model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    for image, target in test_loader:
        # a loader with collate_fn returns batches of lists
        image = image[0]           # take first item from batch
        target = target[0]

        with torch.no_grad():
            outputs = model([image])   # must be list
            pred = outputs[0]

            print(pred.keys())
            print(pred["masks"].shape)

        # predicted mask: (N, H, W) float tensor
        if len(pred["masks"]) > 0:
            mask = pred["masks"][0, 0].cpu().numpy()

            plt.figure(figsize=(6,3))
            plt.subplot(1,2,1)
            plt.title("input")
            plt.imshow(image.permute(1,2,0).cpu())
            plt.axis("off")

            plt.subplot(1,2,2)
            plt.title("prediction")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.show()





plot_eval_examples()
