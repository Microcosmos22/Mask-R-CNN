import torchvision.models as models
import torch
import matplotlib.pyplot as plt

from edarnn import *
from dataloader import *
from scoring import *

model = models.mobilenet_v2(weights="IMAGENET1K_V1")
backbone = model.features
backbone.eval()
thresh = 0.75


train_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
for idx, (image, target, filename) in enumerate(train_loader):
    #print()
    raw_img, raw_mask = full_dataset.get_raw_img_mask(idx)
    x = image[0]    # take first item from batch
    target = target[0]
    feat = backbone(x.unsqueeze(0))


    img = x.permute(1, 2, 0)
    feat = feat.detach()

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(raw_mask[0])
    ax[0].set_title("Given Mask")
    img_vis = x.permute(1,2,0)
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
    ax[1].imshow(img_vis)
    ax[1].set_title("Input")
    ax[2].imshow(feat[0].permute(1,2,0)[:,:,0], cmap="gray")
    ax[2].set_title("Features Channel 0")
    ax[3].imshow(raw_img)
    ax[3].set_title(" Raw given Image")
    plt.savefig("out.png")

    feat = feat.cpu()

    _, C, N_H, N_W = feat.shape
    H_px, W_px = img.shape[:2]
    n_h = int(H_px / N_H)
    n_w = int(W_px / N_W)

    # Collapse H and W into one dimension
    feat_flat = feat.view(C, N_H*N_W).T
    sim = torch.zeros((feat.shape[2]**2, feat.shape[2]**2))
    combined_mask = torch.zeros((img.shape[0], img.shape[1]))


    for k in range(feat.shape[2]**2): # 16 x 16 = 256
        for k_prime in range(feat.shape[2]**2):

            f1 = feat_flat[:, k]
            f2 = feat_flat[:, k_prime]
            sim[k,k_prime] = torch.nn.functional.cosine_similarity(f1, f2, dim=0)

            if sim[k, k_prime] > thresh:
                i, j = divmod(k, N_W) # first cell
                i_prime, j_prime = divmod(k_prime, N_W) # second cell

                i_px = int(i / N_H * H_px)
                j_px = int(j / N_W * W_px)
                i_px_prime = int(i_prime / N_H * H_px)
                j_px_prime = int(j_prime / N_W * W_px)
                #print(f"Cell ({i_cell},{j_cell}) maps roughly to pixel ({i_img},{j_img}) in input image")
                combined_mask[i_px:(i_px+n_w), j_px:(j_px+n_h)] = 1
                combined_mask[i_px_prime:(i_px_prime+n_w), j_px_prime:(j_px_prime+n_h)] = 1

    #plt.clf()
    #plt.imshow(combined_mask, cmap = "grey")
    #plt.savefig("mask.png")

    combined_mask = resize_mask(torch.tensor(combined_mask).unsqueeze(0), raw_img)

    dice = soft_dice(combined_mask.unsqueeze(0), torch.tensor(raw_mask))
    print(f"DICE: {dice}")



    #plt.clf()
    #plt.imshow(sim, cmap="viridis")
    #plt.colorbar()
    #plt.title(f"Similarity to position ({i},{j})")
    #plt.savefig("similarity.png")
