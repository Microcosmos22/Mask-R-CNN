import json

import numba
import numpy as np
from numba import types
import numpy.typing as npt
import pandas as pd
import scipy.optimize

from edarnn import *
from dataset import *
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


model = create_light_mask_rcnn(feat_ex = 0)
state = torch.load("best_overfit_machine.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

base_path = "../recodai-luc-scientific-image-forgery-detection/"
test_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
)
""" transform = train_transform """


if __name__ == "__main__":
    train_loader = DataLoader(train_subset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))


    for idx, (image, target, _) in enumerate(train_loader):
        """ skip authentic images """
        """if (len(target[0]['boxes']) == 0):
            continue"""
        # a loader with collate_fn returns batches of lists
        image = image[0]           # take first item from batch
        target = target[0]
        raw_image, raw_mask = full_dataset.get_raw_img_mask(idx)

        print("\n raw image shape: ")
        print(raw_image.shape)

        with torch.no_grad():
            outputs = model([image])   # must be list
            """ Plot image, mask_pred and mask_true"""
            full_pred_mask = full_mask_from_instance_masks(outputs[0], raw_image.shape)  # shape = network input (H_net, W_net)
            # pred_mask is (H_net, W_net)
            H_orig, W_orig, _ = raw_image.shape

            fig, ax = plt.subplots(2)
            ax[0].imshow(raw_image)
            ax[0].imshow(raw_mask[0], alpha=0.5)

            ax[1].imshow(full_pred_mask)
            plt.show()

        print(full_pred_mask.shape, raw_mask.shape)
        iou = binary_iou(full_pred_mask.cpu().numpy(), np.sum(raw_mask, axis = 0))
        dice = binary_dice(full_pred_mask.cpu().numpy(), np.sum(raw_mask, axis = 0))
        print(f"\nMean IoU: {iou:.4f}, Mean Dice: {dice:.4f}")



        """ Convert to numpy and encode """
        """submission = {
            "case_id": files[int(idx*4)],
            "submission": rle_encode([full_pred_mask_resized.numpy()])
        }"""


        #rle = rle_encode(full_pred_mask_resized.numpy())
        #print(f"rle encoded mask: {rle}")
