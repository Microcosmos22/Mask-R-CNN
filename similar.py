import torch
import torchvision.models as models
import numpy as np
from tqdm import tqdm

from dataloader import *
from scoring import *

# Model
import timm
model = models.resnet50(weights="IMAGENET1K_V1")
backbone = torch.nn.Sequential(*list(model.children())[:-2])
backbone.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone.to(device)

# Params
thresholds = [0.7, 0.8, 0.9]
MAX_IMAGES = 80
mindistlist = [0.2]

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

    for feat, (x, raw_img, raw_mask) in tqdm(zip(features_cache, meta_cache)):
        amount_masks = 0

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
        mask_pairs.fill_diagonal_(False)

        ks, kps = torch.nonzero(mask_pairs, as_tuple=True) # list of matching k, kps

        # -------- mapping k → pixel coords
        ys = (torch.arange(Hf) * cell_h).repeat_interleave(Wf)
        xs = (torch.arange(Wf).repeat(Hf) * cell_w)

        combined_mask = torch.zeros((Hpx, Wpx))

        # -------- paint mask
        for k, kp in zip(ks.tolist(), kps.tolist()):
            y, x_ = ys[k], xs[k]

            y2, x2 = ys[kp], xs[kp]

            if np.sqrt(np.power(x_-x2,2)+np.power(y-y2,2)) > round(0.2*Hpx):
                combined_mask[y:y+cell_h, x_:x_+cell_w] = 1
                combined_mask[y2:y2+cell_h, x2:x2+cell_w] = 1
                amount_masks += 2

        # -------- resize + score
        combined_mask = resize_mask(combined_mask.unsqueeze(0),raw_img)

        dice = soft_dice(combined_mask.unsqueeze(0),torch.tensor(raw_mask))


        fig, ax = plt.subplots(1,2)
        ax[0].imshow(combined_mask.squeeze(0).squeeze(0))
        ax[1].imshow(raw_mask.sum(0))
        plt.savefig("Dice.png")

        dices.append(dice)
        mask_pairs_list.append(amount_masks)



    print(f"Threshold {thresh:.2f} → Mean DICE: {np.mean(dices):.4f}, Mean Cells: {np.mean(mask_pairs_list)}")
