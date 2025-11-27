import json

import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize

from edarnn import *
from scoring import *
from dataset import *


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

model = create_light_mask_rcnn(feat_ex = 1)
state = torch.load("best_model.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

files = [base_path+"supplemental_images/"+number for number in os.listdir(base_path+"supplemental_images/")]

print(files)


for idx, (image, target) in enumerate(test_loader):
    # a loader with collate_fn returns batches of lists
    image = image[0]           # take first item from batch
    target = target[0]
    raw_image, raw_mask = test_dataset.get_raw_img_mask(idx)

    print(raw_image.shape)

    with torch.no_grad():
        outputs = model([image])   # must be list
        pred = outputs[0]


        def full_mask_from_instance_masks(output, image_shape):
            """
            output: dict from MaskRCNN
            image_shape: (H, W, C) of original image
            """
            H, W, _ = image_shape
            full_mask = torch.zeros((H, W), dtype=torch.uint8)

            for box, mask in zip(output['boxes'], output['masks']):
                # clamp box coordinates to image bounds
                x1, y1, x2, y2 = box.int()

                if x2 <= x1 or y2 <= y1 or x1<0 or y1<0 or x2>W or y2>H:
                    continue  # skip degenerate boxes

                # resize mask to box size
                mask_resized = F.interpolate(
                    mask[None], size=(y2 - y1, x2 - x1),
                    mode='bilinear', align_corners=False
                )[0, 0]

                print(f"\n shapes mask, mask[None] and mask_resized:")
                print(mask.shape, mask.unsqueeze(0).shape, mask_resized.shape)

                mask_bin = (mask_resized > 0.5).byte()
                """ actual pasting of the submask"""
                full_mask[y1:y2, x1:x2] = mask_bin

            return full_mask

        pred_mask = full_mask_from_instance_masks(pred, image.shape)  # shape = network input (H_net, W_net)

        print(f"pred_mask shape {pred_mask.shape}")
        # pred_mask is (H_net, W_net)
        H_orig, W_orig, _ = raw_image.shape

        # Ensure float for interpolation
        pred_mask_resized = F.interpolate(
            pred_mask.unsqueeze(0).unsqueeze(0).float(),  # (1,1,H_net,W_net)
            size=(H_orig, W_orig),                        # (height, width) in original image
            mode='nearest'
        )[0, 0].byte()  # back to (H_orig, W_orig)

        if idx >= len(files):
            break
        case_id = files[idx]


        """ Convert to numpy and encode """
        submission = {
            "case_id": files[idx],
            "submission": rle_encode([pred_mask_resized.numpy()])
        }
        plt.imshow(raw_image)
        plt.imshow(pred_mask_resized, alpha=0.5)
        plt.show()

        print(pred.keys())
        print(pred["masks"].shape)
