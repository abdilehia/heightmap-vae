import os
import gradio as gr
import numpy as np
import trimesh
from scipy.spatial import Delaunay
import torch
from VAE import VAE
from UNet import DDPMScheduler
from ControlAdapter import ConditionedUNet
import torchvision.transforms.v2.functional as F

import uuid

device = "cuda" if torch.cuda.is_available() else "cpu"
data = torch.load("./latents_mean_std.pt", weights_only=False)
latent_std = data["std"].item()
latent_mean = data["mean"].item()

vae = VAE().to(device)
vae.load_state_dict(torch.load("./checkpoints/VAE.pth", map_location=device)["vae_state_dict"])
vae.eval()

unet = ConditionedUNet(in_channels=16).to(device)
unet.load_state_dict(torch.load("./checkpoints/UNet+Adapter_32x32.pth", map_location=device)["model_state_dict"])
unet.eval()

scheduler = DDPMScheduler()

def greet(elevation_hint, roughness_hint):
    elevation_hint = elevation_hint['composite'].copy()
    roughness_hint = roughness_hint['composite'].copy()

    elevation_hint = F.to_image(elevation_hint)[:3, :, :]
    roughness_hint = F.to_image(roughness_hint)[:3, :, :]

    elevation_hint = F.rgb_to_grayscale(elevation_hint)
    roughness_hint = F.rgb_to_grayscale(roughness_hint)

    elevation_hint = F.to_dtype(elevation_hint, torch.float32, scale=True)
    roughness_hint = F.to_dtype(roughness_hint, torch.float32, scale=True)

    hints = torch.cat([elevation_hint, roughness_hint], dim=0).unsqueeze(0).to(device)
    hints = torch.nn.functional.interpolate(hints, (32, 32), mode="nearest")
    
    with torch.no_grad():
        latent = torch.randn((1, 16, 32, 32), device=device)
        for t in reversed(range(scheduler.num_train_timesteps)):
            if t % 100 == 0:
                print(f"Denoising step: {t}")
                
            t_tensor = torch.tensor([t], device=device).long()
            predicted_noise = unet.forward_with_hints(latent, t_tensor, hints)
            latent = scheduler.step(predicted_noise, t, latent)

        print("Denoising finished.")
        print(latent.shape)
        latent = (latent * latent_std) + latent_mean
        height, flow, tpi = vae.decode(latent)

    x = np.linspace(0, 256, 256)
    y = np.linspace(0, 256, 256)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack((X.ravel(), Y.ravel()))

    Z = height.detach().cpu().numpy()
    Z = np.flip(Z, 2)
    Z = Z * 50

    tri = Delaunay(XY)
    faces = tri.simplices
    vertices = np.column_stack((Z.ravel(), XY))

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.material = trimesh.visual.material.SimpleMaterial(
        diffuse=[200, 200, 200, 255],  # RGBA (0-255)
        ambient=[50, 50, 50, 255],
        specular=[255, 255, 255, 255]
    )

    file_path = f"tmp/{uuid.uuid4().hex}.glb"
    mesh.export(file_path)

    unet.hints = None

    img_tensor = ((height - height.min()) / (height.max() - height.min())).detach().squeeze().cpu().numpy()
    
    height = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    
    img_tensor = ((flow - flow.min()) / (flow.max() - flow.min())).detach().squeeze().cpu().numpy()
    flow = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    img_tensor = ((tpi - tpi.min()) / (tpi.max() - tpi.min())).detach().squeeze().cpu().numpy()
    tpi = np.clip(img_tensor * 65535.0, 0, 65535).astype(np.uint16)
    return file_path, height, flow, tpi

# I should add slider inputs for:
# - Roughness weights
# - Elevation weights
# - Terrain height

print(os.environ['GRADIO_TEMP_DIR'])
demo = gr.Interface(
    fn=greet,
    inputs=[gr.ImageEditor(image_mode='L'), gr.ImageEditor(image_mode='L')],
    outputs=[gr.Model3D(), gr.Image(), gr.Image(), gr.Image()],
    api_name="predict",
    cache_examples=False,
)

demo.launch()