import torch
import os
import torch.nn as nn

from torch.utils.data import DataLoader
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

# BASE_DIR = "data/stage1_global_z10"
# files = [os.path.join(BASE_DIR, file) for file in os.listdir(BASE_DIR)[:10]]
# dataset = MyDataset(files, data_transforms)
# loader = DataLoader(dataset, batch_size=10, shuffle=True)
files = [('data/stage1_val/artificial_norm.tif', 'data/stage1_val/artificial_flow.tif', 'data/stage1_val/artificial_tpi.tif')]
dataset = MyDataset(files, data_transforms)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

checkpoint = torch.load("./checkpoints/VAE_40.pth", weights_only=False)

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
l1_weight = 0.8
    
weights = checkpoint["vae_state_dict"]
vae = VAE().eval().to(device)
vae.load_state_dict(weights)
for i, images in enumerate(loader):
    images = images.to(device)
    recons, mu, log_var = vae(images)


    norm_orig = images[:, 0:1]
    flow_orig = images[:, 1:2]
    tpi_orig = images[:, 2:3]

    norm_recon = recons[:, 0:1]
    flow_recon = recons[:, 1:2]
    tpi_recon = recons[:, 2:3]


    norm_loss = l1_weight * l1_loss(norm_recon, norm_orig) + (1 - l1_weight) * l2_loss(norm_recon, norm_orig)
    flow_loss = l1_weight * l1_loss(flow_recon, flow_orig) + (1 - l1_weight) * l2_loss(flow_recon, flow_orig)
    tpi_loss = l1_weight * l1_loss(tpi_recon, tpi_orig) + (1 - l1_weight) * l2_loss(tpi_recon, tpi_orig)

    print(f"Height Loss = {norm_loss.item()}")
    print(f"Height (sigma) = {vae.s1.item()}")
    print(f"Flow Loss = {flow_loss.item()}")
    print(f"Flow (sigma) = {vae.s2.item()}")
    print(f"TPI Loss = {tpi_loss.item()}")
    print(f"TPI (sigma) = {vae.s3.item()}")

    norm_orig_16 = norm_orig[0].squeeze(0)
    norm_orig_16 = ((norm_orig_16 - norm_orig_16.min()) / (norm_orig_16.max() - norm_orig_16.min())).detach().cpu().numpy()
    norm_orig_16 = np.clip(norm_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(norm_orig_16).save(f"./visualisations/val_norm_original.png")

    norm_recon_16 = norm_recon[0].squeeze(0)
    norm_recon_16 = ((norm_recon_16 - norm_recon_16.min()) / (norm_recon_16.max() - norm_recon_16.min())).detach().cpu().numpy()
    norm_recon_16 = np.clip(norm_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(norm_recon_16).save(f"./visualisations/val_norm_reconstruction.png")
    
    flow_orig_16 = flow_orig[0].squeeze(0)
    flow_orig_16 = ((flow_orig_16 - flow_orig_16.min()) / (flow_orig_16.max() - flow_orig_16.min())).detach().cpu().numpy()
    flow_orig_16 = np.clip(flow_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(flow_orig_16).save(f"./visualisations/val_flow_original.png")

    flow_recon_16 = flow_recon[0].squeeze(0)
    flow_recon_16 = ((flow_recon_16 - flow_recon_16.min()) / (flow_recon_16.max() - flow_recon_16.min())).detach().cpu().numpy()
    flow_recon_16 = np.clip(flow_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(flow_recon_16).save(f"./visualisations/val_flow_reconstruction.png")

    tpi_orig_16 = tpi_orig[0].squeeze(0)
    tpi_orig_16 = ((tpi_orig_16 - tpi_orig_16.min()) / (tpi_orig_16.max() - tpi_orig_16.min())).detach().cpu().numpy()
    tpi_orig_16 = np.clip(tpi_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(tpi_orig_16).save(f"./visualisations/val_tpi_original.png")

    tpi_recon_16 = tpi_recon[0].squeeze(0)
    tpi_recon_16 = ((tpi_recon_16 - tpi_recon_16.min()) / (tpi_recon_16.max() - tpi_recon_16.min())).detach().cpu().numpy()
    tpi_recon_16 = np.clip(tpi_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
    Image.fromarray(tpi_recon_16).save(f"./visualisations/val_tpi_reconstruction.png")