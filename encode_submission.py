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

def predict_test_images(model, test_path, device):
    submission = {
        "case_id": [],
        "submission": []
    }


    model.eval()
    predictions = {}

    test_files = sorted(os.listdir(test_path))

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    for file in tqdm(test_files, desc="Processing test images"):
        case_id = file.split('.')[0]

        # Load and preprocess image
        img_path = os.path.join(test_path, file)
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        original_size = image_np.shape[:2]

        # Apply transformations
        transformed = transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(device)

        # Model prediction
        with torch.no_grad():

            outputs = model(image_tensor)   # must be list
            image = image_tensor.squeeze(0).permute(1,2,0)
            outputs = inv_transform(outputs, image)


            """ If 3% forged -> forged"""
            confidence_threshold = 0.03     # CHANGE THIS TO SEE RESULTS(changes)

            if torch.sum(outputs) / outputs.numel() < 0.03:
                # No detections -> authentic image
                predictions[case_id] = "authentic"
            else:
                # Combine all detected masks

                # RLE encoding
                if torch.sum(outputs) == 0:
                    predictions[case_id] = "authentic"
                else:
                    submission["case_id"].append(case_id)
                    submission["submission"].append(rle_encode([outputs.numpy()]))


    df = pd.DataFrame(submission)

    # save as CSV
    df.to_csv("submission.csv", index=False)

    return predictions




base_path = "../recodai-luc-scientific-image-forgery-detection/"
test_dataset = ForgeryDataset(
    paths['train_authentic'],
    paths['train_forged'],
    paths['train_masks'],
)
""" transform = train_transform """


if __name__ == "__main__":
    model = UNet()
    state = torch.load("unet_overfit_model.pth", map_location="cpu")
    model.load_state_dict(state)

    predictions = predict_test_images(model, '../recodai-luc-scientific-image-forgery-detection/test_images', device)
