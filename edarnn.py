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
from torch.utils.data import Subset

from dataloader import *

from sklearn.model_selection import KFold
import warnings
from itertools import product
from wakepy import keep

import json
import os
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')

# Checking GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

base_path = "../recodai-luc-scientific-image-forgery-detection/"
test_dataset = ForgeryDataset(
    None,
    os.path.join(base_path, "supplemental_images"),
    os.path.join(base_path, "supplemental_masks"),
    transform=train_transform
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

full_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
    transform=train_transform
)

full_dataset_subset = Subset(full_dataset, list(range(2)))  # keeps mapping: [0,1]
subset_indices = full_dataset_subset.indices  # ← real original dataset indices

train_idx, val_idx = train_test_split(
    range(len(full_dataset_subset)),
    test_size=0.1,
    random_state=42,
    shuffle=False
)

print("\nTRAIN FILES:")
for idx in train_idx:
    orig_idx = subset_indices[idx]     # map subset index → original dataset index
    filename = full_dataset.get_filename(orig_idx)
    print(str(orig_idx)+": ", filename['image_id']+".png")



train_subset = Subset(full_dataset_subset, train_idx)
val_subset = Subset(full_dataset_subset, val_idx)


feature_extractors = []


eval_subset = torch.utils.data.Subset(train_subset, [0])

print(len(train_subset), len(eval_subset))

# optionally set transforms
val_subset.dataset.transform = val_transform

train_loader = DataLoader(train_subset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
eval_loader = DataLoader(eval_subset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


def create_light_mask_rcnn(feat_ex = 0, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=1,
rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200, num_classes = 2):
    if feat_ex == 0:
        backbone = torchvision.models.mobilenet_v3_small(pretrained=True).features
        in_ch = 576
        backbone.out_channels = 256
        out_ch = 256
    elif feat_ex == 1:
        backbone = torchvision.models.mobilenet_v3_large(pretrained=True).features
        in_ch = 960
        backbone.out_channels = 256
        out_ch = 256
    elif feat_ex == 2:
        resnet = torchvision.models.resnet34(pretrained=True)
        backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        in_ch = 512
        backbone.out_channels = 512   # resnet3 4's final feature depth
        out_ch = 512

    # extracts characteristics from an image
    backbone = nn.Sequential(
        backbone,
        nn.Conv2d(in_ch, out_ch, kernel_size=1),
        nn.ReLU(inplace=True)
    )
    backbone.out_channels = out_ch


    # Anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ROI pools
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=5,
        sampling_ratio=samplR
    )

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=56, sampling_ratio=2)
    #model = MaskRCNN(backbone, num_classes=2, mask_roi_pool=mask_roi_pool)

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        min_size=512,
        max_size=512,
        rpn_pre_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=200,
        rpn_post_nms_top_n_test=200,
        box_detections_per_img=100
    )


    for p in model.roi_heads.mask_head.parameters():
        p.requires_grad = False

    for p in model.roi_heads.mask_predictor.parameters():
        p.requires_grad = False
    model.roi_heads.mask_on = False


    for p in model.backbone.parameters():
        p.requires_grad = False
    model.roi_heads.score_thresh = 0.000


    return model

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets, _ in tqdm(dataloader, desc="Training"):


        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        for t in targets:
            print(f"yo {len(t['masks'])} masks in target")
            """
            if len(t["boxes"]) > 0:
                for box in t["boxes"]:
                    x1, y1, x2, y2 = box
                    h = y2 - y1
                    w = x2 - x1
                    # do your mask logic here

                    t["masks"] = torch.ones_like(t["masks"])  # force full masks
            """
        #full_mask = full_mask_from_instance_masks(targets[0], images[0])  # shape = network input (H_net, W_net)
        #plt.imshow(full_mask)
        #plt.show()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()  # temporarily switch to eval mode
        with torch.no_grad():
            preds = model([images[0]])
            scores = preds[0]['scores']  # confidence scores for each box
            #print(boxes.shape)
        model.train()  # switch back to training

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader), loss_dict['loss_mask'].item(), loss_dict['loss_box_reg'], loss_dict['loss_classifier']


def validate_epoch(model, dataloader, device):
    model.train()  # For validation, we use train mode because of the features of Mask R-CNN
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (images, targets, _) in enumerate(tqdm(dataloader, desc="Validation")):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(dataloader)
import torch
import os
import time

def save_model_safe(model, path, max_retries=5, delay=0.5):
    """Save model safely on Windows, retrying if file is locked."""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                os.remove(path)  # remove previous file
            if isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)
            print(f"Saved model to {path}")
            return
        except PermissionError:
            print(f"File {path} is locked. Waiting {delay}s and retrying...")
            time.sleep(delay)
    raise PermissionError(f"Could not write to {path} after {max_retries} retries")


def train_parameters(train_loader, val_loader, eval_loader, machinepath, num_epochs, feat_ex = 0, out_ch=256, lr = 0.001, weight_decay = 0.001, step_size = 5, gamma = 0.1, samplR=2,
rpn_pre_train = 1000, rpn_pre_test = 1000, rpn_post_train=200, rpn_post_test=200, early = False):
    model = create_light_mask_rcnn(feat_ex, lr, weight_decay, step_size, gamma, samplR,
    rpn_pre_train, rpn_pre_test, rpn_post_train, rpn_post_test)
    if os.path.isfile(machinepath):
        model.load_state_dict(torch.load(machinepath))
        print(" LOADING: "+machinepath)
    model.to(device)
    print("\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses = []
    val_losses = []
    rcnn_losses = []
    # Early stopping parameters
    patience = 4        # epochs to wait for improvement
    best_val = 10000000.1
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        """ Train, validate, evaluate """
        train_loss, loss_mask, loss_box_reg, loss_classifier = train_epoch(model, train_loader, optimizer, device)

        train_losses.append(train_loss)

        if val_loader is not None:
            val_loss = validate_epoch(model, val_loader, device)
            val_losses.append(val_loss)
            rcnn_losses.append([loss_mask, loss_box_reg.detach().numpy(), loss_classifier.detach().numpy()])
            print(f"\nLOSSES: Train: {train_loss:.4f}, Val: {val_losses[-1]:.4f}, Mask: {loss_mask:.4f}, Box regr. {loss_box_reg:.4f}, Classifier: {loss_classifier:.4f}")
        else:
            val_losses.append(0)
            print(f"\nTrain Loss: {train_loss:.4f}")


        if eval_loader is not None:
            iou, dice, props = evaluate_segmentation(model, eval_loader, device)

            scheduler.step()

            print(f"\nTrain Loss: {train_loss:.4f}, Val Loss: {val_losses[-1]:.4f}")
            print(f"IoU: {np.mean(iou):.4f}, DICE: {np.mean(dice):.4f}")

            best_iou = np.mean(iou)
            best_dice = np.mean(dice)

        if not early:
            epochs_no_improve = 0
            save_model_safe(model, machinepath)

        if early:
            if (best_val > val_loss):
                best_val = val_loss
                epochs_no_improve = 0
                if os.path.exists(machinepath):
                    os.remove(machinepath)   # safely remove old file
                save_model_safe(model, machinepath)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    early_stop = True
                    break
    if eval_loader is not None:
        return model, best_iou, best_dice, train_loss, val_loss
    else:
        return model, train_losses, val_losses, np.asarray(rcnn_losses)

if __name__ == "__main__":
    losses = {"params": [], "errors": []}
    count = 0

    machinepath = "./data/200epoch_10017.pth"
    num_epochs = 200
    batch = 1

    feat_ex = [0]
    lr = [0.001]
    weight_decay = [0.001]
    step_size = [5]
    gamma = [0.1]
    samplR=2
    rpn_pre_train = 1000
    rpn_pre_test = 1000
    rpn_post_train = 200
    rpn_post_test = 200

    out_ch = [1]

    all_combinations = list(product(
        feat_ex, out_ch, lr, weight_decay,
        step_size, gamma
    ))

    print(f"Total combinations: {len(all_combinations)}")
    print(all_combinations)

    with keep.running():

        for combo in all_combinations:

            for batch_idx, (images, targets, _) in enumerate(tqdm(train_loader, desc="Validation")):
                print(len(targets[0]["boxes"]), targets[0]["masks"].sum())
            #print(f"Feat_ex: {feat_ex}, out_ch: {out_ch}, lr: {lr}, weight_d: {weight_decay}, step_size: {step_size}, gamma: {gamma}, samplR: {samplR}, rpn_pre_train: {rpn_pre_train} ")
            model, train_losses, val_losses, rcnn_losses = train_parameters(train_loader, val_loader, None, machinepath, num_epochs, combo[0], combo[1], combo[2], combo[3], combo[4], combo[5], samplR, rpn_pre_train, rpn_pre_test, rpn_post_train, rpn_post_test, False)

            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Val")
            plt.plot(rcnn_losses[:,0], label="Mask")
            plt.plot(rcnn_losses[:,1], label=" Box regr.")
            plt.plot(rcnn_losses[:,2], label="Classifier")
            plt.legend()
            plt.savefig("./data/last_training.png")
            plt.show()


            print(" Saving losses.json")
            losses["params"].append(combo)
            with open("./data/losses.json", "w") as f:
                json.dump(losses, f, indent=4)
