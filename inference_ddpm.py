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
    vae.load_state_dict(torch.load("./checkpoints/VAE/VAE_30.pth", map_location=device)["vae_state_dict"])
    vae.eval()

    unet = UNet().to(device)
    unet.load_state_dict(torch.load("./checkpoints/UNet/UNet_400.pth", map_location=device)["model_state_dict"])
    unet.eval()

    scheduler = DDPMScheduler()
    print(clean_latents.shape)
    print("Generating heightmap")
    
    with torch.no_grad():
        latent = torch.randn((1, 4, 32, 32), device=device)

        for t in reversed(range(scheduler.num_train_timesteps)):
            if t % 100 == 0:
                print(f"Denoising step: {t}")
                
            t_tensor = torch.tensor([t], device=device).long()
            predicted_noise = unet(latent, t_tensor)
            latent = scheduler.step(predicted_noise, t, latent)

        print("Denoising finished.")

        latent = (latent * latent_std) + latent_mean
        generated_heightmap = vae.decode(latent)

    img_tensor = generated_heightmap.squeeze().cpu().numpy()
    img_16bit = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    
    Image.fromarray(img_16bit).save("/output/vae/heightmap.png")
    print("Saved to /output/vae/heightmap.png")