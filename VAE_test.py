import torch
import os
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from VAE import VAE, MyDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

data_transforms = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(256),
    v2.ToDtype(torch.float32, scale=True)
])

HEIGHT_DIR = './data/stage1_val/height'
FLOW_DIR = './data/stage1_val/flow'
TPI_DIR  = './data/stage1_val/tpi'
files = [(os.path.join(HEIGHT_DIR, file), os.path.join(FLOW_DIR, file), os.path.join(TPI_DIR, file)) for file in os.listdir(FLOW_DIR) if file.endswith('.tif')]
dataset = MyDataset(files, data_transforms)
loader = DataLoader(dataset, batch_size=32, num_workers=4, prefetch_factor=4, shuffle=True)

checkpoint = torch.load("./checkpoints/Attempt 4/VAE/VAE_75.pth", weights_only=False)

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
l1_weight = 0.8

total_loss = 0

weights = checkpoint["vae_state_dict"]
vae = VAE().eval().to(device)
vae.load_state_dict(weights)
for i, images in enumerate(loader):
    with torch.no_grad():
        images = images.to(device)
        recons, mu, log_var = vae(images)

    norm_orig = images[:, 0:1]
    flow_orig = images[:, 1:2]
    tpi_orig = images[:, 2:3]
    norm_recon, flow_recon, tpi_recon = recons

    norm_loss = l1_weight * l1_loss(norm_recon, norm_orig) + (1 - l1_weight) * l2_loss(norm_recon, norm_orig)
    flow_loss = l1_weight * l1_loss(flow_recon, flow_orig) + (1 - l1_weight) * l2_loss(flow_recon, flow_orig)
    tpi_loss = l1_weight * l1_loss(tpi_recon, tpi_orig) + (1 - l1_weight) * l2_loss(tpi_recon, tpi_orig)

    print(f"Height Loss = {norm_loss.item()}")
    print(f"Height (sigma) = {vae.s1.item()}")
    print(f"Flow Loss = {flow_loss.item()}")
    print(f"Flow (sigma) = {vae.s2.item()}")
    print(f"TPI Loss = {tpi_loss.item()}")
    print(f"TPI (sigma) = {vae.s3.item()}")

    total_loss += (norm_loss).item()
print(f"Total Loss = {total_loss/len(loader)}")
    # for i in range(norm_orig.shape[0]):
    #     norm_orig_16 = norm_orig[i].squeeze(0)
    #     norm_orig_16 = ((norm_orig_16 - norm_orig_16.min()) / (norm_orig_16.max() - norm_orig_16.min())).detach().cpu().numpy()
    #     norm_orig_16 = np.clip(norm_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(norm_orig_16).save(f"./visualisations/val_norm_original_{i+1}.png")

    #     norm_recon_16 = norm_recon[i].squeeze(0)
    #     norm_recon_16 = ((norm_recon_16 - norm_recon_16.min()) / (norm_recon_16.max() - norm_recon_16.min())).detach().cpu().numpy()
    #     norm_recon_16 = np.clip(norm_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(norm_recon_16).save(f"./visualisations/val_norm_reconstruction_{i+1}.png")

    #     norm_orig_16 = flow_orig[i].squeeze(0)
    #     norm_orig_16 = ((norm_orig_16 - norm_orig_16.min()) / (norm_orig_16.max() - norm_orig_16.min())).detach().cpu().numpy()
    #     norm_orig_16 = np.clip(norm_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(norm_orig_16).save(f"./visualisations/val_flow_original_{i+1}.png")

    #     flow_recon_16 = flow_recon[i].squeeze(0)
    #     flow_recon_16 = ((flow_recon_16 - flow_recon_16.min()) / (flow_recon_16.max() - flow_recon_16.min())).detach().cpu().numpy()
    #     flow_recon_16 = np.clip(flow_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(flow_recon_16).save(f"./visualisations/val_flow_reconstruction_{i+1}.png")

    #     norm_orig_16 = tpi_orig[i].squeeze(0)
    #     norm_orig_16 = ((norm_orig_16 - norm_orig_16.min()) / (norm_orig_16.max() - norm_orig_16.min())).detach().cpu().numpy()
    #     norm_orig_16 = np.clip(norm_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(norm_orig_16).save(f"./visualisations/val_tpi_original_{i+1}.png")

    #     tpi_recon_16 = tpi_recon[i].squeeze(0)
    #     tpi_recon_16 = ((tpi_recon_16 - tpi_recon_16.min()) / (tpi_recon_16.max() - tpi_recon_16.min())).detach().cpu().numpy()
    #     tpi_recon_16 = np.clip(tpi_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    #     Image.fromarray(tpi_recon_16).save(f"./visualisations/val_tpi_reconstruction_{i+1}.png")