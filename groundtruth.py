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

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def plot_all_boxes(
    image,                  # [3,H,W] CPU tensor
    gt_boxes,               # [G,4]
    gt_labels,              # [G] (0 or 1)
    pred_boxes=None,        # [N,4]
    pred_scores=None,       # [N]
    title="GT & predictions"
):
    img = F.to_pil_image(image)

    plt.figure(figsize=(7,7))
    plt.imshow(img)
    ax = plt.gca()

    # ---- GT boxes ----
    for box, label in zip(gt_boxes, gt_labels):
        x1,y1,x2,y2 = box.tolist()
        color = "green" if label == 1 else "blue"
        ax.add_patch(
            plt.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                fill=False, edgecolor=color, linewidth=2
            )
        )
        ax.text(
            x1, y1-4,
            f"GT:{int(label)}",
            color=color,
            fontsize=9,
            weight="bold"
        )

    # ---- Predicted boxes ----
    if pred_boxes is not None and pred_scores is not None:
        for box, score in zip(pred_boxes, pred_scores):
            x1,y1,x2,y2 = box.tolist()
            ax.add_patch(
                plt.Rectangle(
                    (x1,y1), x2-x1, y2-y1,
                    fill=False, edgecolor="red", linewidth=1
                )
            )
            ax.text(
                x1, y2+10,
                f"P:{score:.2f}",
                color="red",
                fontsize=8
            )

    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    for idx, (image, target, filename) in enumerate(train_loader):

        images_t = model.transform(images)
        features = model.backbone(images_t.tensors)
        proposals, _ = model.rpn(images_t, features)

        plot_all_boxes(
            image.cpu(),
            target["boxes"].cpu(),
            target["labels"].cpu(),   # 0 or 1
            proposals.cpu(),
            cls_probs[:, 1].cpu(),     # forged prob
            title="Green=forged GT | Blue=authentic GT | Red=pred"
        )
        break
