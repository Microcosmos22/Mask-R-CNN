import matplotlib.pyplot as plt
import numpy as np
import os

base = "../recodai-luc-scientific-image-forgery-detection/train_masks"

for file in os.listdir(base):
    path = os.path.join(base, file)


    mask = np.load(path, allow_pickle=True)
    mask_single = np.any(mask, axis=0).astype(np.uint8)  # shape (H, W)


    plt.imshow(mask_single, cmap="grey")
    plt.show()
