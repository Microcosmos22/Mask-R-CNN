import json
import torch
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize

from edarnn import *
from dataloader import *
from scoring import *





class ParticipantVisibleError(Exception):
    pass


@numba.jit(nopython=True)
def _rle_encode_jit(x: npt.NDArray, fg_val: int = 1) -> list[int]:
    """Numba-jitted RLE encoder."""
    dots = np.where(x.T.flatten() == fg_val)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def rle_encode(masks: list[npt.NDArray], fg_val: int = 1) -> str:
    """
    Adapted from contrails RLE https://www.kaggle.com/code/inversion/contrails-rle-submission
    Args:
        masks: list of numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encodings as a string, with each RLE JSON-encoded and separated by a semicolon.
    """
    return ';'.join([json.dumps(_rle_encode_jit(x, fg_val)) for x in masks])


@numba.njit
def _rle_decode_jit(mask_rle: npt.NDArray, height: int, width: int) -> npt.NDArray:
    """
    s: numpy array of run-length encoding pairs (start, length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if len(mask_rle) % 2 != 0:
        # Numba requires raising a standard exception.
        raise ValueError('One or more rows has an odd number of values.')

    starts, lengths = mask_rle[0::2], mask_rle[1::2]
    starts -= 1
    ends = starts + lengths
    for i in range(len(starts) - 1):
        if ends[i] > starts[i + 1]:
            raise ValueError('Pixels must not be overlapping.')
    img = np.zeros(height * width, dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> npt.NDArray:
    """
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """

    mask_rle = json.loads(mask_rle)
    mask_rle = np.asarray(mask_rle, dtype=np.int32)
    starts = mask_rle[0::2]
    if sorted(starts) != list(starts):
        raise ParticipantVisibleError('Submitted values must be in ascending order.')
    try:
        return _rle_decode_jit(mask_rle, shape[0], shape[1]).reshape(shape, order='F')
    except ValueError as e:
        raise ParticipantVisibleError(str(e)) from e
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def resize_mask_to_image(output, target_image):
    """
    Resizes the predicted mask from a Mask R-CNN output to the original image size.

    Args:
        output (dict): Mask R-CNN output for a single image.
            Must contain 'masks' key with shape (N, 1, H_pred, W_pred)
        target_image (numpy.ndarray or torch.Tensor): Original image to match.
            Shape: (H_img, W_img, C) or (C, H_img, W_img)

    Returns:
        torch.Tensor: Resized mask (H_img, W_img) with values between 0 and 1
    """

    masks = output['masks']  # (N, 1, H_pred, W_pred)

    # Sum all instance masks into a single mask
    combined_mask = masks.sum(dim=0, keepdim=True)  # (1, 1, H_pred, W_pred)
    combined_mask = torch.clamp(combined_mask, 0, 1)  # ensure values 0/1

    # Get target height and width
    if isinstance(target_image, torch.Tensor):
        if target_image.ndim == 3:
            H_img, W_img = target_image.shape[1], target_image.shape[2] if target_image.shape[0] in [1, 3] else target_image.shape[0], target_image.shape[1]
        else:
            raise ValueError(f"Unexpected target_image shape: {target_image.shape}")
    else:  # assume numpy
        H_img, W_img = target_image.shape[:2]

    # Interpolate mask to target size
    mask_resized = F.interpolate(
        combined_mask.float(),
        size=(H_img, W_img),
        mode='bilinear',
        align_corners=False
    )

    return mask_resized.squeeze(0).squeeze(0)  # (H_img, W_img)

# Make sure device is consistent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = create_light_mask_rcnn()
state = torch.load("best_overfit_machine.pth", map_location=device)  # load directly to device
model.load_state_dict(state)
model.to(device)  # ensure model is on same device as inputs
model.eval()

base_path = "../recodai-luc-scientific-image-forgery-detection/"

test_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

if __name__ == "__main__":
    """ ONLY PLOTS THE FIRST ELEM IN BATCH """
    plot = True

    for idx, (image, target, filename) in enumerate(train_loader):
        image = image[0]    # take first item from batch
        target = target[0]

        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)  # move input to same device
            outputs = model(image_batch)  # forward pass

            # Convert image for plotting
            image_plot = image.permute(1, 2, 0).cpu().numpy()

            # Inverse transform to original size (if needed)
            outputs_orig_size = resize_mask_to_image(outputs[0], image_plot)

            # Compute dice score
            pred = torch.from_numpy(outputs_orig_size.cpu().numpy()).float()
            dice = soft_dice(pred, target)
            print(f"\nIdx: {idx} Dice: {dice:.4f}")

            if plot:
                import matplotlib
                matplotlib.use("TkAgg")  # force interactive backend
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 2, figsize=(10,5))

                # Denormalize image for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                image_denorm = image.cpu() * std + mean
                image_plot = np.clip(image_denorm.permute(1,2,0).numpy(), 0, 1)

                ax[0].imshow(image_plot)
                ax[0].imshow(target['masks'][0].cpu().numpy(), alpha=0.5)

                mask_plot = np.clip(outputs_orig_size.cpu().numpy(), 0, 1)
                ax[1].imshow(mask_plot)

                plt.show(block=True)


            # Prepare submission
            submission = {
                "case_id": filename[0],
                "submission": rle_encode([outputs_orig_size])
            }
