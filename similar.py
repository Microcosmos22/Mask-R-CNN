import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from dataloader import *
from scoring import *

# Model
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
backbone = model.features
backbone.eval()

device = "cpu"
backbone.to(device)

# Params
thresholds = [0.9, 0.92, 0.94, 0.96, 0.98]
MAX_IMAGES = 8

train_loader = DataLoader(
    full_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)

# -----------------------------
# Precompute features once
# -----------------------------
features_cache = []
meta_cache = []

with torch.no_grad():
    for idx, (image, _, _) in enumerate(train_loader):
        if idx >= MAX_IMAGES:
            break

        x = image[0].to(device)                    # [3,H,W]
        feat = backbone(x.unsqueeze(0)).cpu()      # [1,C,Hf,Wf]

        raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)

        features_cache.append(feat)
        meta_cache.append((x, raw_img, raw_mask))

# -----------------------------
# Threshold loop
# -----------------------------

for thresh in thresholds:
    dices = []
    mask_pairs_list = []

    for feat, (x, raw_img, raw_mask) in zip(features_cache, meta_cache):

        _, C, Hf, Wf = feat.shape
        Hpx, Wpx = x.shape[1:]

        cell_h = Hpx // Hf
        cell_w = Wpx // Wf

        # -------- flatten + normalize features
        feat_flat = feat.view(C, -1).T            # [HW, C]
        feat_flat = torch.nn.functional.normalize(feat_flat, dim=1)

        # -------- similarity matrix (FAST)
        sim = feat_flat @ feat_flat.T              # [HW, HW]

        # -------- threshold
        mask_pairs = sim > thresh
        ks, kps = torch.nonzero(mask_pairs, as_tuple=True)

        # -------- mapping k → pixel coords
        ys = (torch.arange(Hf) * cell_h).repeat_interleave(Wf)
        xs = (torch.arange(Wf).repeat(Hf) * cell_w)

        combined_mask = torch.zeros((Hpx, Wpx))

        # -------- paint mask
        for k, kp in zip(ks.tolist(), kps.tolist()):
            y, x_ = ys[k], xs[k]
            combined_mask[y:y+cell_h, x_:x_+cell_w] = 1

            y2, x2 = ys[kp], xs[kp]
            combined_mask[y2:y2+cell_h, x2:x2+cell_w] = 1

        # -------- resize + score
        combined_mask = resize_mask(combined_mask.unsqueeze(0),raw_img)

        dice = soft_dice(combined_mask.unsqueeze(0),torch.tensor(raw_mask))

        dices.append(dice)
        mask_pairs_list.append(mask_pairs.sum())



    print(f"Threshold {thresh:.2f} → Mean DICE: {np.mean(dices):.4f}, Mean Cells: {np.mean(mask_pairs_list)}")
