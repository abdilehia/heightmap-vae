import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from torch.optim.lr_scheduler import ConstantLR, LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.io import read_image, write_png, ImageReadMode

import numpy as np
from PIL import Image

# Convolutional VAE made to understand and eventually generate heightmaps
class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # 3 tasks so yeah
        self.s1 = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        self.s2 = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)
        self.s3 = nn.Parameter(torch.tensor([0], dtype=torch.float), requires_grad=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1), # 256x256 -> 128x128
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.LeakyReLU(),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1), # 512 -> 32
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        img = self.decode(z)
        return img, mu, log_var

    def encode(self, x):
        emb = self.encoder(x) # (8, 32, 32)
        bottleneck = self.bottleneck(emb)
        mu, log_var = torch.chunk(bottleneck, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + (epsilon * std)
        return z

    def decode(self, x):
        return self.decoder(x)

    # def get_last_layer(self):
    #     return self.decoder[-2].weight # Ignore sigmoid layer

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1),
        )
    def forward(self, x):
        return self.model(x)

# class MyDataset(Dataset):
#     def __init__(self, files, transform) -> None:
#         self.files = files
#         self.transform = transform
#     def __getitem__(self, index) -> torch.Tensor:
#         image = read_image(self.files[index], mode=ImageReadMode.UNCHANGED)
#         image_tensor = self.transform(image)
#         return image_tensor
#     def __len__(self):
#         return len(self.files)


import rasterio as rs
class MyDataset(Dataset):
    def __init__(self, files, transform) -> None:
        self.files = files
        self.transform = transform
    def __getitem__(self, index) -> torch.Tensor:
        norm, flow, tpi = self.files[index]

        norm = rs.open(norm).read()
        flow = rs.open(flow).read()
        tpi = rs.open(tpi).read()

        norm = torch.tensor(norm)

        flow = torch.tensor(flow)
        flow = flow.clamp(min=torch.quantile(flow, q=0.02), max=torch.quantile(flow, q=0.98))
        flow = torch.log1p(flow)
        flow = (flow - flow.min()) / (flow.max() - flow.min())
        flow = flow * 2 - 1

        tpi = torch.tensor(tpi)
        tpi = tpi.clamp(min=torch.quantile(tpi, q=0.02), max=torch.quantile(tpi, q=0.98))
        tpi = (tpi - tpi.min()) / (tpi.max() - tpi.min())
        tpi = tpi * 2 - 1

        image_tensor = self.transform(torch.cat([norm, flow, tpi], dim=0))
        return image_tensor
    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # BASE_DIR = "data/stage1_global_z10"

    NORM_DIR = './data/stage1_norm'
    FLOW_DIR = './data/stage1_flow'
    TPI_DIR  = './data/stage1_tpi'
    files = [(os.path.join(NORM_DIR, file), os.path.join(FLOW_DIR, file), os.path.join(TPI_DIR, file)) for file in os.listdir(NORM_DIR) if file.endswith('.tif')]
    dataset = MyDataset(files, data_transforms)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, prefetch_factor=4, shuffle=True, drop_last=True)


    vae = VAE().train().to(device)


    epoch = 0
    total_epochs = 50
    warmup_epochs = 20

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    l1_weight = 0.8


    kl_weight = 0.01
    optim_vae = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.9))


    vae_warmup_scheduler = LinearLR(optim_vae, start_factor=0.01, total_iters=warmup_epochs)
    vae_main_scheduler = CosineAnnealingLR(optim_vae, T_max=total_epochs - warmup_epochs)
    vae_scheduler = SequentialLR(
        optim_vae, 
        schedulers=[vae_warmup_scheduler, vae_main_scheduler],
        milestones=[warmup_epochs]
    )

    checkpoint = torch.load("./checkpoints/VAE_40.pth", weights_only=False)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    optim_vae.load_state_dict(checkpoint["optim_vae_state_dict"])
    vae_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"] + 1


    for epoch in range(epoch, total_epochs):
        print(f"Epoch {epoch}:")
        total_vae_loss_without_gan = 0
        total_vae_loss = 0
        total_disc_loss = 0
        for batch_idx, images in enumerate(loader):

            # VAE training
            optim_vae.zero_grad()

            images = images.to(device)
            recons, mu, log_var = vae(images)
            # norm_recon, flow_recon, tpi_recon = recons

            norm_orig = images[:, 0:1]
            flow_orig = images[:, 1:2]
            tpi_orig = images[:, 2:3]

            norm_recon = recons[:, 0:1]
            flow_recon = recons[:, 1:2]
            tpi_recon = recons[:, 2:3]

            norm_loss = l1_weight * l1_loss(norm_recon, norm_orig) + (1 - l1_weight) * l2_loss(norm_recon, norm_orig)
            flow_loss = l1_weight * l1_loss(flow_recon, flow_orig) + (1 - l1_weight) * l2_loss(flow_recon, flow_orig)
            tpi_loss = l1_weight * l1_loss(tpi_recon, tpi_orig) + (1 - l1_weight) * l2_loss(tpi_recon, tpi_orig)

            norm_loss = 0.5*torch.exp(-vae.s1) * norm_loss + 0.5*vae.s1
            flow_loss = 0.5*torch.exp(-vae.s2) * flow_loss + 0.5*vae.s2
            tpi_loss = 0.5*torch.exp(-vae.s3) * tpi_loss + 0.5*vae.s3

            norm_loss = norm_loss.mean()
            flow_loss = flow_loss.mean()
            tpi_loss = tpi_loss.mean()
            
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            mtl_loss = norm_loss + flow_loss + tpi_loss
            vae_loss = mtl_loss + kl_weight * kl_loss

            vae_loss_val = vae_loss.item()
            total_vae_loss += vae_loss_val

            vae_loss.backward()
            optim_vae.step()

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(loader)}:")
                print(f"VAE Loss = {vae_loss_val}.")
                print(f"Height Loss = {norm_loss.item()}")
                print(f"Height (sigma) = {vae.s1.item()}")
                print(f"Flow Loss = {flow_loss.item()}")
                print(f"Flow (sigma) = {vae.s2.item()}")
                print(f"TPI Loss = {tpi_loss.item()}")
                print(f"TPI (sigma) = {vae.s3.item()}")

        vae_scheduler.step()
        print(f"Avg VAE loss = {total_vae_loss / len(loader)}", end="\n\n")

        if epoch % 5 == 0:
            print(f"Saving checkpoint 'VAE_{epoch}.pth'")
            norm_orig_16 = norm_orig[0].squeeze(0)
            norm_orig_16 = ((norm_orig_16 - norm_orig_16.min()) / (norm_orig_16.max() - norm_orig_16.min())).detach().cpu().numpy()
            norm_orig_16 = np.clip(norm_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(norm_orig_16).save(f"./visualisations/e{epoch}_norm_original.png")

            norm_recon_16 = norm_recon[0].squeeze(0)
            norm_recon_16 = ((norm_recon_16 - norm_recon_16.min()) / (norm_recon_16.max() - norm_recon_16.min())).detach().cpu().numpy()
            norm_recon_16 = np.clip(norm_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(norm_recon_16).save(f"./visualisations/e{epoch}_norm_reconstruction.png")
            
            flow_orig_16 = flow_orig[0].squeeze(0)
            flow_orig_16 = ((flow_orig_16 - flow_orig_16.min()) / (flow_orig_16.max() - flow_orig_16.min())).detach().cpu().numpy()
            flow_orig_16 = np.clip(flow_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(flow_orig_16).save(f"./visualisations/e{epoch}_flow_original.png")

            flow_recon_16 = flow_recon[0].squeeze(0)
            flow_recon_16 = ((flow_recon_16 - flow_recon_16.min()) / (flow_recon_16.max() - flow_recon_16.min())).detach().cpu().numpy()
            flow_recon_16 = np.clip(flow_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(flow_recon_16).save(f"./visualisations/e{epoch}_flow_reconstruction.png")

            tpi_orig_16 = tpi_orig[0].squeeze(0)
            tpi_orig_16 = ((tpi_orig_16 - tpi_orig_16.min()) / (tpi_orig_16.max() - tpi_orig_16.min())).detach().cpu().numpy()
            tpi_orig_16 = np.clip(tpi_orig_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(tpi_orig_16).save(f"./visualisations/e{epoch}_tpi_original.png")

            tpi_recon_16 = tpi_recon[0].squeeze(0)
            tpi_recon_16 = ((tpi_recon_16 - tpi_recon_16.min()) / (tpi_recon_16.max() - tpi_recon_16.min())).detach().cpu().numpy()
            tpi_recon_16 = np.clip(tpi_recon_16 * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(tpi_recon_16).save(f"./visualisations/e{epoch}_tpi_reconstruction.png")

        # if epoch % 10 == 0:
            checkpoint = {
                "vae_state_dict": vae.state_dict(),
                "optim_vae_state_dict": optim_vae.state_dict(),
                "scheduler_state_dict": vae_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"./checkpoints/VAE_{epoch}.pth")

    print("Finished training.")

    checkpoint = {
        "vae_state_dict": vae.state_dict(),
        "optim_vae_state_dict": optim_vae.state_dict(),
        "scheduler_state_dict": vae_scheduler.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, f"./checkpoints/VAE_{epoch}.pth")
    torch.save(vae.state_dict(), "./checkpoints/VAE.pth")
