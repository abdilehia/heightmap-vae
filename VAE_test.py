import torch
import os

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

BASE_DIR = "data/stage1_global_z10"
files = [os.path.join(BASE_DIR, file) for file in os.listdir(BASE_DIR)[:10]]
dataset = MyDataset(files, data_transforms)
loader = DataLoader(dataset, batch_size=10, shuffle=True)

checkpoint = torch.load("./checkpoints/VAE_30.pth", weights_only=False)

weights = checkpoint["vae_state_dict"]
vae = VAE().eval().to(device)
vae.load_state_dict(weights)
for i, images in enumerate(loader):
    recons, mu, log_var = vae(images.to(device))

    orig_tensor = images.detach().cpu().numpy()
    orig_16 = np.clip(orig_tensor * 65535.0, 0, 65535).astype(np.uint16) # 0-1 -> 0-65536

    recon_tensor = recons.detach().cpu().numpy()
    recon_16 = np.clip(recon_tensor * 65535.0, 0, 65535).astype(np.uint16)

    for i in range(orig_16.shape[0]):
        Image.fromarray(orig_16[i].squeeze(0)).save(f"./visualisations/test_{i}_original.png")
        Image.fromarray(recon_16[i].squeeze(0)).save(f"./visualisations/test_{i}_recon.png")
