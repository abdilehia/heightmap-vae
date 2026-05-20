import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

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

        self.shared_upsample = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        self.norm_decoder_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.flow_decoder_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.tpi_decoder_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        img = self.decode(z)
        return img, mu, log_var

    def encode(self, x):
        emb = self.encoder(x)
        bottleneck = self.bottleneck(emb)
        mu, log_var = torch.chunk(bottleneck, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + (epsilon * std)
        return z

    def decode(self, x):
        upsampled =  self.shared_upsample(x)
        return (self.norm_decoder_head(upsampled), self.flow_decoder_head(upsampled), self.tpi_decoder_head(upsampled))

import rasterio as rs
class MyDataset(Dataset):
    def __init__(self, files, transform) -> None:
        self.files = files
        self.transform = transform
    def __getitem__(self, index) -> torch.Tensor:
        height, flow, tpi = self.files[index]

        height = rs.open(height).read()
        flow = rs.open(flow).read()
        tpi = rs.open(tpi).read()

        height = torch.tensor(height)

        flow = torch.tensor(flow)
        flow = flow.clamp(min=torch.quantile(flow, q=0.02), max=torch.quantile(flow, q=0.98))
        flow = torch.log1p(flow)
        flow = (flow - flow.min()) / (flow.max() - flow.min())
        flow = flow * 2 - 1

        tpi = torch.tensor(tpi)
        tpi = tpi.clamp(min=torch.quantile(tpi, q=0.02), max=torch.quantile(tpi, q=0.98))
        tpi = (tpi - tpi.min()) / (tpi.max() - tpi.min())
        tpi = tpi * 2 - 1
        image_tensor = self.transform(torch.cat([height, flow, tpi], dim=0))
        return image_tensor
    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_transforms = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(256),
        v2.ToDtype(torch.float32, scale=False)
    ])

    def visualise(x: torch.Tensor, filename: str):
        x = x.squeeze(0)
        x = ((x - x.min()) / (x.max() - x.min())).detach().cpu().numpy()
        x = np.clip(x * 65535.0, 0, 65535).astype(np.uint16)
        Image.fromarray(x).save(f"./visualisations/{filename}")

    HEIGHT_DIR = './data/stage1_norm'
    FLOW_DIR = './data/stage1_flow'
    TPI_DIR  = './data/stage1_tpi'

    files = [(os.path.join(HEIGHT_DIR, file), os.path.join(FLOW_DIR, file), os.path.join(TPI_DIR, file)) for file in os.listdir(FLOW_DIR) if file.endswith('.tif')]
    dataset = MyDataset(files, data_transforms)
    loader = DataLoader(dataset, batch_size=32, num_workers=4, prefetch_factor=4, shuffle=True, drop_last=True)
    # loader = DataLoader(dataset, batch_size=10)

    vae = VAE().train().to(device)
    
    epoch = 0
    total_epochs = 100
    warmup_epochs = 40

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

    # checkpoint = torch.load("./checkpoints/VAE_5.pth", weights_only=False)
    # vae.load_state_dict(checkpoint["vae_state_dict"])
    # optim_vae.load_state_dict(checkpoint["optim_vae_state_dict"])
    # vae_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    # epoch = checkpoint["epoch"] + 1

    for epoch in range(epoch, total_epochs):
        print(f"Epoch {epoch}:")
        total_vae_loss = 0
        for batch_idx, images in enumerate(loader):

            # VAE training
            optim_vae.zero_grad()

            images = images.to(device)
            height_orig = images[:, 0:1]
            flow_orig = images[:, 1:2]
            tpi_orig = images[:, 2:3]

            recons, mu, log_var = vae(images)
            height_recon, flow_recon, tpi_recon = recons

            height_loss = l1_weight * l1_loss(height_recon, height_orig) + (1 - l1_weight) * l2_loss(height_recon, height_orig)
            flow_loss = l1_weight * l1_loss(flow_recon, flow_orig) + (1 - l1_weight) * l2_loss(flow_recon, flow_orig)
            tpi_loss = l1_weight * l1_loss(tpi_recon, tpi_orig) + (1 - l1_weight) * l2_loss(tpi_recon, tpi_orig)

            height_loss_val = height_loss.item()
            flow_loss_val = flow_loss.item()
            tpi_loss_val = tpi_loss.item()
            final_loss_val = height_loss_val + flow_loss_val + tpi_loss_val

            height_loss = 0.5*torch.exp(-vae.s1) * height_loss + 0.5*vae.s1
            flow_loss = 0.5*torch.exp(-vae.s2) * flow_loss + 0.5*vae.s2
            tpi_loss = 0.5*torch.exp(-vae.s3) * tpi_loss + 0.5*vae.s3

            
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            mtl_loss = height_loss + flow_loss + tpi_loss
            vae_loss = mtl_loss + kl_weight * kl_loss

            vae_loss_val = vae_loss.item()
            total_vae_loss += vae_loss_val

            vae_loss.backward()
            optim_vae.step()

            if (batch_idx + 1) % 20 == 0:
                print(f"Batch {batch_idx + 1}/{len(loader)}:")
                print(f"VAE Loss = {final_loss_val}.")
                print(f"Height Loss = {height_loss_val}")
                print(f"Height (sigma) = {vae.s1.item()}")
                print(f"Flow Loss = {flow_loss_val}")
                print(f"Flow (sigma) = {vae.s2.item()}")
                print(f"TPI Loss = {tpi_loss_val}")
                print(f"TPI (sigma) = {vae.s3.item()}", end="\n\n")

        vae_scheduler.step()
        print(f"Avg VAE loss = {total_vae_loss / len(loader)}", end="\n\n")

        if (epoch + 1) % 5 == 0:
            print(f"Saving checkpoint 'VAE_{epoch + 1}.pth'")
            visualise(height_orig[0], filename=f"e{epoch + 1}_height_original.png")
            visualise(height_recon[0], filename=f"e{epoch + 1}_height_reconstruction.png")
            visualise(flow_orig[0], filename=f"e{epoch + 1}_flow_original.png")
            visualise(flow_recon[0], filename=f"e{epoch + 1}_flow_reconstruction.png")
            visualise(tpi_orig[0], filename=f"e{epoch + 1}_tpi_original.png")
            visualise(tpi_recon[0], filename=f"e{epoch + 1}_tpi_reconstruction.png")

        # if epoch % 10 == 0:
            checkpoint = {
                "vae_state_dict": vae.state_dict(),
                "optim_vae_state_dict": optim_vae.state_dict(),
                "scheduler_state_dict": vae_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"./checkpoints/VAE_{epoch + 1}.pth")

    print("Finished training.")

    checkpoint = {
        "vae_state_dict": vae.state_dict(),
        "optim_vae_state_dict": optim_vae.state_dict(),
        "scheduler_state_dict": vae_scheduler.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, f"./checkpoints/VAE_{epoch + 1}.pth")
    torch.save(vae.state_dict(), "./checkpoints/VAE.pth")
