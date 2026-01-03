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



def paint_boxes(output, target, combined_mask, topk=10, thickness=2):
    """
    Paints:
      - top-k predicted boxes (from output)
      - all GT target boxes
    onto combined_mask (in image space).
    """

    _, _, H, W = combined_mask.shape

    scores = output["scores"]
    pred_boxes = output["boxes"]
    gt_boxes = target["boxes"]

    # Sort predictions by score
    idx = scores.argsort(descending=True)
    pred_boxes = pred_boxes[idx]

    def paint_box(cm, x1, y1, x2, y2):
        x1, x2 = int(x1), int(x2)
        y1, y2 = int(y1), int(y2)

        # clamp safely
        x1 = max(0, min(x1, W - 1))
        x2 = max(1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(1, min(y2, H))

        # top / bottom
        cm[:, :, y1:y1+thickness, x1:x2] = 1
        cm[:, :, y2-thickness:y2, x1:x2] = 1

        # left / right
        cm[:, :, y1:y2, x1:x1+thickness] = 1
        cm[:, :, y1:y2, x2-thickness:x2] = 1

    # --- paint top-k predictions ---
    for i in range(min(topk, len(pred_boxes))):
        print(f"Box {i}: score = {scores[i].item():.4f}")
        paint_box(combined_mask, *pred_boxes[i])

    # --- paint GT boxes ---
    for box in gt_boxes:
        paint_box(combined_mask, *box)

    return combined_mask



def resize_mask(combined_mask, target_image):
    """
    Resizes a mask to match target_image size.
    Accepts combined_mask of shape [1,H,W] or [1,1,H,W].
    """
    # Ensure mask has shape [N, C, H, W] for F.interpolate
    if combined_mask.ndim == 3:
        combined_mask = combined_mask.unsqueeze(0)  # -> [1, 1, H, W]
    elif combined_mask.ndim != 4:
        raise ValueError(f"Unexpected mask shape: {combined_mask.shape}")

    # Get target height and width
    if isinstance(target_image, torch.Tensor):
        if target_image.ndim == 3:  # (C,H,W) or (H,W,C)
            if target_image.shape[0] in (1, 3):      # (C,H,W)
                H_img, W_img = target_image.shape[1], target_image.shape[2]
            else:                                    # (H,W,C)
                H_img, W_img = target_image.shape[0], target_image.shape[1]

        elif target_image.ndim == 2:  # (H,W)
            H_img, W_img = target_image.shape

        else:
            raise ValueError(f"Unexpected target_image shape: {target_image.shape}")
    else:  # assume numpy
        H_img, W_img = target_image.shape[:2]

    # Interpolate mask to target size
    mask_resized = F.interpolate(
        combined_mask.float(),
        size=(H_img, W_img),
        mode='nearest'
    )
    return mask_resized

def combine_resize_submasks(output, target_image, threshold):
    """ Combines all submasks into a full image,
     """

    if threshold is not None:
        keep = output["scores"] > threshold
        masks = output["masks"][keep]
    else:
        masks = output["masks"]

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    combined_mask = masks.sum(dim=0)               # (H, W)
    combined_mask = torch.clamp(combined_mask, 0, 1)
    combined_mask = combined_mask.unsqueeze(0).unsqueeze(0)
    print(f" Combining {len(masks)} masks and resizing to original")

    mask_resized = resize_mask(combined_mask, target_image)

    return mask_resized.squeeze(0).squeeze(0)  # (H_img, W_img)


class ForgeryDataset(Dataset):
    def __init__(self, authentic_path, forged_path, masks_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        # Collect all data samples
        self.samples = []

        # Forged images
        for file in sorted(os.listdir(forged_path)):
            if file[0] == ".":
                continue
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
            for file in sorted(os.listdir(authentic_path)):
                if file[0] == ".":
                    continue
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

    def get_image_props(self, image, mask):
        boxes, labels, masks = self.mask_to_boxes(mask, plot = False)
        mask_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        mask_whiteness = mask_np.sum() / (image.shape[1] * image.shape[2])

        return {
            "Npixels" : len(image[0])*len(image[1]),
            "WhiteNess" : mask_whiteness
            }

    def get_filename(self, idx):
        return self.samples[idx]

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)  # (H, W, 3)

        # Load and process mask
        if os.path.exists(sample['mask_path']):
            mask = np.load(sample['mask_path'])

            # Handle multi-channel masks
            if mask.ndim == 3:
                if mask.shape[0] <= 10:  # channels first (C, H, W)
                    mask = np.any(mask, axis=0)
                elif mask.shape[-1] <= 10:  # channels last (H, W, C)
                    mask = np.any(mask, axis=-1)
                else:
                    raise ValueError(f"Ambiguous 3D mask shape: {mask.shape}")

            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # Resize mask to match image if needed
        H_img, W_img = image.shape[:2]
        if mask.shape != (H_img, W_img):
            print(f"[WARN] pre-resizing mask {mask.shape} -> {(H_img, W_img)}")
            mask = cv2.resize(mask.astype(np.uint8), (W_img, H_img), interpolation=cv2.INTER_NEAREST)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = F_transforms.to_tensor(image)
            mask = torch.tensor(mask, dtype=torch.uint8)

        # Prepare targets for Mask R-CNN
        if sample['is_forged'] and mask.sum() > 0:
            boxes, labels, masks = self.mask_to_boxes(mask, plot = False)

            """
            PAINTING FILLED BOXES ON LOAD
            # build instance masks from boxes (cheap, done once)
            N = len(boxes)
            H, W = mask.shape
            instance_masks = torch.zeros((N, H, W), dtype=torch.uint8)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.int()
                instance_masks[i, y1:y2, x1:x2] = 1
            """

            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
        else:
            # For authentic images or images without masks
            H, W = image.shape[1:]
            # Example: one box covering the whole image
            boxes = torch.tensor([[0.0, 0.0, W, H]], dtype=torch.float32)
            labels = torch.zeros((1,), dtype=torch.int64)  # label=0 → background
            masks = torch.zeros((0, H, W), dtype=torch.uint8)  # no mask
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((1,), dtype=torch.float32),
                'iscrowd': torch.zeros((1,), dtype=torch.int64)
            }

        return image, target, self.samples[idx]['image_path']  # return filename too

    def mask_to_boxes(self, mask, plot = False):
        """Convert segmentation mask to bounding boxes for Mask R-CNN"""
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.array(mask)

        # --- FIX: squeeze extra dimensions ---
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim != 2:
            raise ValueError("mask_to_boxes expects a single 2D mask")

        # --- FIX: ensure binary + correct type ---
        mask_np = (mask_np > 0).astype(np.uint8)

        # Safety: fall back to empty if shape still wrong
        if mask_np.ndim != 2:
            return (torch.zeros((0,4),dtype=torch.float32),
                    torch.zeros((0,),dtype=torch.int64),
                    torch.zeros((0, *mask_np.shape[-2:]),dtype=torch.uint8))

        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        masks = []

        for contour in contours:
            if len(contour) > 0:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out very small regions
                if w > 5 and h > 5:
                    boxes.append([x, y, x + w, y + h])
                    # Create binary mask for this contour
                    contour_mask = np.zeros_like(mask_np)
                    cv2.fillPoly(contour_mask, [contour], 1)
                    masks.append(contour_mask)

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=torch.uint8)

        if plot:
            # --- DEBUG VIS ---
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(mask_np, cmap="gray")

            for box in boxes:
                x1, y1, x2, y2 = box.int().tolist()
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="red", linewidth=2
                )
                ax.add_patch(rect)

            ax.set_title("mask_to_boxes debug")
            ax.axis("off")
            plt.show()


        return boxes, labels, masks

base_path = "../recodai-luc-scientific-image-forgery-detection/"
paths = {
        'train_authentic': os.path.join(base_path, "train_images/authentic"),
        'train_forged': os.path.join(base_path, "train_images/forged"),
        'train_masks': os.path.join(base_path, "train_masks"),
        'test_images': os.path.join(base_path, "test_images"),
    }

tenimg_paths = {
        'train_authentic': os.path.join(base_path, "train_images/10img/authentic"),
        'train_forged': os.path.join(base_path, "train_images/10img/forged"),
        'train_masks': os.path.join(base_path, "train_images/10img/masks")
    }

# Transformations for learning
train_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),

    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.RandomRotate90(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

if __name__ == "__main__":

    print("test")
