import torch
import os
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from VAE import VAE, MyDataset

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
    loader = DataLoader(dataset, num_workers=4, prefetch_factor=4, batch_size=128)


    vae = VAE().to(device)
    vae.load_state_dict(torch.load("./checkpoints/VAE/VAE_30.pth")["vae_state_dict"])
    vae.eval()

    latents = []
    for batch_idx, images in enumerate(loader):
        with torch.no_grad():
            images = images.to(device)

            mu, log_var = vae.encode(images)
            z = vae.reparameterize(mu, log_var)

            latents.append(z.detach().cpu())


        if batch_idx % 5 == 0:
            print(f"\rProcessing: {batch_idx}/{len(loader)}")

    latents = torch.cat(latents, dim=0)

    std = latents.std()
    mean = latents.mean()

    print(f"Std. = {std}. Mean = {mean}")
    print("Finished processing heightmaps.")
    torch.save(
        {
        "latents": latents,
        "std": std,
        "mean": mean
        }, "./map_tensors.pt")