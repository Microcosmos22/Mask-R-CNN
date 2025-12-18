import json
import torch
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize
import torch
import torch.nn.functional as F

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

# Make sure device is consistent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = create_light_mask_rcnn()
#state = torch.load("./images/frozen_natural_300/natural_frozen_300epochs.pth", map_location=device)  # load directly to device
state = torch.load("./data//natural_frozen_100epochs.pth", map_location=device)  # load directly to device

print("Model weights: ")
print(model.load_state_dict(state, strict=False))

model.load_state_dict(state)
model.to(device)  # ensure model is on same device as inputs
model.eval()

base_path = "../recodai-luc-scientific-image-forgery-detection/"

test_dataset = ForgeryDataset(paths['train_authentic'],paths['train_forged'],paths['train_masks'],)

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,collate_fn=lambda x: tuple(zip(*x)))



if __name__ == "__main__":
    train_loader = DataLoader(full_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


    """ ONLY PLOTS THE FIRST ELEM IN BATCH """
    plot = True



    for idx, (image, target, filename) in enumerate(train_loader):
        image = image[0]    # take first item from batch
        target = target[0]
        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)


        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))  # forward pass

            target_mask = combine_resize_submasks(target, raw_img, threshold = None)
            outputs_orig_size = combine_resize_submasks(outputs[0], raw_img, threshold = None)

            ######
            target_mask_norm = resize_mask(target_mask.unsqueeze(0), image) # to normalized
            target_paintedboxes = paint_boxes(outputs[0], target, torch.tensor(target_mask_norm))
            target_paintedboxes = resize_mask(target_paintedboxes, raw_img) # to original



            print(outputs_orig_size.shape, target_mask.shape)

            dice = soft_dice(outputs_orig_size, target_mask, True)
            print(f"\nIdx: {idx} Dice: {dice:.4f}")

            if plot:

                fig, ax = plt.subplots(1, 2, figsize=(10,5))

                # Denormalize image for display
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                image_denorm = image.cpu() * std + mean
                image_denorm = resize_mask(image_denorm, target_mask)
                image_plot = np.clip(image_denorm.squeeze(0).permute(1,2,0).numpy(), 0, 1)


                ax[0].imshow(image_plot)
                ax[0].imshow(target_paintedboxes.squeeze(0).squeeze(0).cpu().numpy(), alpha=0.5)

                outputs_orig_size = np.clip(outputs_orig_size.cpu().numpy(), 0, 1)
                ax[1].imshow(outputs_orig_size)
                ax[0].set_title("Target mask + output boxes")
                ax[1].set_title(" Output mask")

                plt.show(block=True)


            # Prepare submission
            """
            submission = {
                "case_id": filename[0],
                "submission": rle_encode([outputs_orig_size])
            }
            """
