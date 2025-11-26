
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
#from scoring import *

from sklearn.model_selection import KFold
import warnings
from itertools import product
from wakepy import keep

import json
from edarnn import *


if __name__ == "__main__":
    losses = {"params": [], "errors": []}

    full_dataset = ForgeryDataset(
        paths['train_authentic'],
        paths['train_forged'],
        paths['train_masks'],
        transform=train_transform
    )
    count = 0
    from torch.utils.data import random_split, DataLoader
    # 80% train, 20% validation
    # DataLoaders
    num_epochs = 10

    feat_ex = [1]
    out_ch = [256]
    lr = [0.001]
    weight_decay = [0.001]
    step_size = [5]
    gamma = [0.1]
    samplR=1
    rpn_pre_train = 1000
    rpn_pre_test = 1000
    rpn_post_train = 200
    rpn_post_test = 200

    all_combinations = list(product(
        feat_ex, out_ch, lr, weight_decay,
        step_size, gamma
    ))

    print(f"Total combinations: {len(all_combinations)}")
    print(all_combinations)

    np.save(f"losses.npy", np.asarray([1,2,3,4]))

    with keep.running():

        for combo in all_combinations:

                #full_dataset = Subset(full_dataset, list(range(50)))
                indices = list(range(len(full_dataset)))

                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=0.1,
                    random_state=42,
                    shuffle=True
                )


                train_subset = Subset(full_dataset, train_idx)
                val_subset = Subset(full_dataset, val_idx)

                # optionally set transforms
                val_subset.dataset.transform = val_transform

                train_loader = DataLoader(train_subset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
                val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

                #print(f"Feat_ex: {feat_ex}, out_ch: {out_ch}, lr: {lr}, weight_d: {weight_decay}, step_size: {step_size}, gamma: {gamma}, samplR: {samplR}, rpn_pre_train: {rpn_pre_train} ")
                model, iou, dice, train_loss, val_loss = train_parameters(train_loader, val_loader, 10, combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], samplR, rpn_pre_train, rpn_pre_test, rpn_post_train, rpn_post_test)
                torch.save(model.state_dict(), "best_model.pth")
                print(" SAVE MODEL")
