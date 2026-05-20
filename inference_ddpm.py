import torch
import numpy as np
from PIL import Image
from VAE import VAE
from UNet import UNet, DDPMScheduler


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load("./map_tensors.pt", weights_only=False)

    clean_latents = data["latents"]
    latent_std = data["std"].item()
    latent_mean = data["mean"].item()
    
    print("Mean:", latent_mean)
    print("Std:", latent_std)
    
    vae = VAE().to(device)
    vae.load_state_dict(torch.load("./checkpoints/Attempt 4/VAE/VAE_75.pth", map_location=device)["vae_state_dict"])
    vae.eval()

    unet = UNet(in_channels=16).to(device)
    unet.load_state_dict(torch.load("./checkpoints/Attempt 4/UNet/UNet_400.pth", map_location=device)["model_state_dict"])
    unet.eval()

    scheduler = DDPMScheduler()
    print(clean_latents.shape)
    print("Generating heightmap")
    
    with torch.no_grad():
        latent = torch.randn((1, 16, 32, 32), device=device)
        # latent = torch.nn.functional.interpolate(latent, (32, 32), mode="bicubic", align_corners=False)
        for t in reversed(range(scheduler.num_train_timesteps)):
            if t % 100 == 0:
                print(f"Denoising step: {t}")
                
            t_tensor = torch.tensor([t], device=device).long()
            predicted_noise = unet(latent, t_tensor)
            latent = scheduler.step(predicted_noise, t, latent)

        print("Denoising finished.")
        print(latent.shape)
        latent = (latent * latent_std) + latent_mean
        height, flow, tpi = vae.decode(latent)
        
    # height = height.clamp(min=height.quantile(0.02), max=height.quantile(0.98))
    # flow = flow.clamp(min=flow.quantile(0.02), max=flow.quantile(0.98))
    # tpi = tpi.clamp(min=tpi.quantile(0.02), max=tpi.quantile(0.98))

    # img_tensor = ((height - height.min()) / (height.max() - height.min())).detach().squeeze().cpu().numpy()
    # img_16bit = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    # Image.fromarray(img_16bit).save("./output/UNet/height.png")
    # img_tensor = ((flow - flow.min()) / (flow.max() - flow.min())).detach().squeeze().cpu().numpy()
    # img_16bit = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    # Image.fromarray(img_16bit).save("./output/UNet/flow.png")
    # img_tensor = ((tpi - tpi.min()) / (tpi.max() - tpi.min())).detach().squeeze().cpu().numpy()
    # img_16bit = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    # Image.fromarray(img_16bit).save("./output/UNet/tpi.png")
    
    # print("Saved to /output/UNet/heightmap.png")