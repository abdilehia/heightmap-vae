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

    HEIGHT_DIR = './data/stage1_norm'
    FLOW_DIR = './data/stage1_flow'
    TPI_DIR  = './data/stage1_tpi'
    files = [(os.path.join(HEIGHT_DIR, file), os.path.join(FLOW_DIR, file), os.path.join(TPI_DIR, file)) for file in os.listdir(HEIGHT_DIR) if file.endswith('.tif')]
    dataset = MyDataset(files, data_transforms)
    loader = DataLoader(dataset, num_workers=4, prefetch_factor=4, batch_size=128)


    vae = VAE().to(device)
    vae.load_state_dict(torch.load("./checkpoints/Attempt 4/VAE_75.pth")["vae_state_dict"])
    vae.eval()

    latents = []
    for batch_idx, images in enumerate(loader):
        with torch.no_grad():
            images = images.to(device)
            # norm_orig = images[:, 0:1]
            # flow_orig = images[:, 1:2]
            # tpi_orig = images[:, 2:3]

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