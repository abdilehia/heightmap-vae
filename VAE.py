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

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1), # 256x256 -> 128x128
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), # 128x128 -> 64x64
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), # 64x64 -> 32x32
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1) # 512 -> 8
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        img = self.decode(z)
        return img, mu, log_var

    def encode(self, x):
        emb = self.encoder(x) # (8, 32, 32)
        mu, log_var = torch.chunk(emb, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + (epsilon * std)
        return z

    def decode(self, x):
        return self.decoder(x)

    def get_last_layer(self):
        return self.decoder[-2].weight # Ignore sigmoid layer

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

class MyDataset(Dataset):
    def __init__(self, files, transform) -> None:
        self.files = files
        self.transform = transform
    def __getitem__(self, index) -> torch.Tensor:
        image = read_image(self.files[index], mode=ImageReadMode.UNCHANGED)
        image_tensor = self.transform(image)
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

    BASE_DIR = "data/stage1_global_z10"
    files = [os.path.join(BASE_DIR, file) for file in os.listdir(BASE_DIR)[:15000]]
    dataset = MyDataset(files, data_transforms)
    loader = DataLoader(dataset, batch_size=32, num_workers=8, prefetch_factor=4, shuffle=True, drop_last=True)


    vae = VAE().train().to(device)
    disc = Discriminator().train().to(device)


    epoch = 0
    total_epochs = 50
    warmup_epochs = 20
    disc_epochs = 20

    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()
    l1_weight = 0.8


    kl_weight = 1e-5
    optim_vae = optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optim_disc = optim.AdamW(disc.parameters(), lr=1e-4, betas=(0.5, 0.9))


    vae_warmup_scheduler = LinearLR(optim_vae, start_factor=0.01, total_iters=warmup_epochs)
    vae_main_scheduler = CosineAnnealingLR(optim_vae, T_max=total_epochs - warmup_epochs)
    vae_scheduler = SequentialLR(
        optim_vae, 
        schedulers=[vae_warmup_scheduler, vae_main_scheduler],
        milestones=[warmup_epochs]
    )


    def calculate_adaptive_gan_weight(recon_loss, gan_loss, vae_last_layer):
        recon_grads = torch.autograd.grad(recon_loss, vae_last_layer, retain_graph=True)[0]
        disc_grads = torch.autograd.grad(gan_loss, vae_last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / torch.norm(disc_grads + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight


    checkpoint = torch.load("./checkpoints/VAE_20.pth", weights_only=False)
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

            recon_loss = l1_weight * l1_loss(recons, images) \
                       + (1 - l1_weight) * l2_loss(recons, images)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

            if epoch >= disc_epochs:
                disc_logits = disc(recons)
                gan_loss = F.binary_cross_entropy_with_logits(disc_logits, torch.ones_like(disc_logits))
                disc_weight = calculate_adaptive_gan_weight(recon_loss, gan_loss, vae.get_last_layer())
                vae_loss_without_gan = recon_loss \
                    + kl_weight * kl_loss \

                vae_loss = vae_loss_without_gan \
                    + disc_weight * gan_loss
            else:
                vae_loss = recon_loss + kl_weight * kl_loss
            # vae_loss = recon_loss + kl_weight * kl_loss

            vae_loss_without_gan_val = vae_loss_without_gan.item()
            vae_loss_val = vae_loss.item()
            total_vae_loss += vae_loss_val
            total_vae_loss_without_gan += vae_loss_without_gan_val

            vae_loss.backward()
            optim_vae.step()


            # Discriminator training

            if epoch >= disc_epochs:
                optim_disc.zero_grad()

                disc_logits_fake = disc(recons.detach())
                disc_loss_fake = F.binary_cross_entropy_with_logits(disc_logits_fake, torch.zeros_like(disc_logits_fake))

                disc_logits_real = disc(images.detach())
                disc_loss_real = F.binary_cross_entropy_with_logits(disc_logits_real, torch.ones_like(disc_logits_real))

                disc_loss = (disc_loss_fake + disc_loss_real) / 2

                disc_loss_val = disc_loss.item()
                total_disc_loss += disc_loss_val

                disc_loss.backward()
                optim_disc.step()

            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(loader)}:")
                print(f"VAE Loss (w/o GAN) = {vae_loss_without_gan_val}. VAE Loss (w/ GAN) = {vae_loss_val}.")
                print(f"Discriminator Loss = {disc_loss_val}", end="\n\n")

        vae_scheduler.step()
        print(f"Avg VAE loss (w/o GAN) = {total_vae_loss_without_gan / len(loader)}", end="\n\n")
        print(f"Avg VAE loss (w/ GAN) = {total_vae_loss / len(loader)}", end="\n\n")
        print(f"Avg Discriminator loss = {total_disc_loss / len(loader)}", end="\n\n")

        if epoch % 5 == 0:
            print(f"Saving checkpoint 'VAE_{epoch}.pth'")
            
            orig_tensor = images[0].detach().cpu().squeeze(0).numpy()
            orig_16 = np.clip(orig_tensor * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(orig_16).save(f"./visualisations/e{epoch}_original.png")

            recon_tensor = recons[0].detach().cpu().squeeze(0).numpy()
            recon_16 = np.clip(recon_tensor * 65535.0, 0, 65535).astype(np.uint16)
            Image.fromarray(recon_16).save(f"./visualisations/e{epoch}_recon.png")

        # if epoch % 10 == 0:
            checkpoint = {
                "vae_state_dict": vae.state_dict(),
                "optim_vae_state_dict": optim_vae.state_dict(),
                "scheduler_state_dict": vae_scheduler.state_dict(),
                "disc_state_dict": disc.state_dict(),
                "optim_disc_state_dict": optim_disc.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"./checkpoints/VAE_{epoch}.pth")

    print("Finished training.")

    checkpoint = {
        "vae_state_dict": vae.state_dict(),
        "optim_vae_state_dict": optim_vae.state_dict(),
        "scheduler_state_dict": vae_scheduler.state_dict(),
        "disc_state_dict": disc.state_dict(),
        "optim_disc_state_dict": optim_disc.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, f"./checkpoints/VAE_{epoch}.pth")
    torch.save(vae.state_dict(), "./checkpoints/VAE.pth")
