import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from UNet import UNet, DDPMScheduler

# class Adapter(nn.Module):
#     def __init__ (self, in_channels:int=2, expands: list[int]=[64, 256, 512], outputs: list[int]=[256, 512]):
#         super().__init__()
#         # Outputs represents the dims at the stages you want to inject this
#         # expands is just about what steps you want to take to reach those dims
#         # For example, the default in channels is 2 but the first output is 256
#         # Instead of going straight from 2 to 256, we go 2->64->256
#         # This adds expand layers but not downsample or zero conv layers
#         # Zero conv layers are conv layers initialized to zero to avoid overwhelming the UNet
#         # Basically, all values in outputs should be in expands but values in expands that don't appear in outputs 
#         # are treated as a progressive expand until we reach one that is
#         expand_layers = []
#         downsample_layers = []
#         zero_conv_layers = []

#         for i, expand in enumerate(expands):
#             if i == 0:
#                 expand_layers.append([
#                     nn.Conv2d(in_channels, expand, kernel_size=3, padding=1),
#                     nn.SiLU()
#                 ])
#             else:
#                 expand_layers[-1].append(nn.Conv2d(expands[i-1], expand, kernel_size=3, padding=1))
#                 expand_layers[-1].append(nn.SiLU())

#             if expand in outputs:
#                 zero_conv_layers.append(nn.Conv2d(expand, expand, kernel_size=1))
#                 if i < len(expands) - 1:
#                     downsample_layers.append(nn.Sequential(
#                         nn.Conv2d(expand, expand, kernel_size=3, stride=2, padding=1),
#                         nn.SiLU()
#                     ))
#                     expand_layers.append([])


#         for i, expand_layer in enumerate(expand_layers):
#             if len(expand_layer) == 0: continue 
#             expand_layer = nn.Sequential(*expand_layer)
#             self.add_module(f"expand{i+1}", expand_layer)
        
#         for i, downsample_layer in enumerate(downsample_layers):
#             self.add_module(f"downsample{i+1}", downsample_layer)

#         for i, zero_conv_layer in enumerate(zero_conv_layers):
#             nn.init.zeros_(zero_conv_layer.weight)
#             if zero_conv_layer.bias is not None:
#                 nn.init.zeros_(zero_conv_layer.bias)
#             self.add_module(f"zero_conv{i+1}", zero_conv_layer)
        
#         print(expand_layers)
#         assert len(expand_layers) == len(zero_conv_layers), "Make sure at least the final value in expands is also in outputs"
#         self.layer_count = len(expand_layers)
#         self.layers = dict(self.named_modules())
#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
#         x_ = x.detach().clone()
#         outputs: list[torch.Tensor] = []
#         for i in range(self.layer_count):
#             expand = self.layers.get(f"expand{i+1}")
#             downsample = self.layers.get(f"downsample{i+1}")
#             zero_conv = self.layers.get(f"zero_conv{i+1}")
#             assert expand is not None
#             assert downsample is not None
#             assert zero_conv is not None
#             x = expand(x)
#             x = zero_conv(x)
#             outputs.append(x)
#             x = downsample(x)
#         assert len(outputs) == self.layer_count
#         return tuple(outputs)

class Adapter(nn.Module):
    def __init__ (self, in_channels:int=2):
        super().__init__()

        self.expand1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )
        
        self.expand2 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.zero_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        nn.init.zeros_(self.zero_conv1.weight)
        if self.zero_conv1.bias is not None:
            nn.init.zeros_(self.zero_conv1.bias)

        self.downsample2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.SiLU()
        )

        self.expand3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.zero_conv2 = nn.Conv2d(512, 512, kernel_size=1)
        nn.init.zeros_(self.zero_conv2.weight)
        if self.zero_conv2.bias is not None:
            nn.init.zeros_(self.zero_conv2.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs: list[torch.Tensor] = []
        x = self.expand1(x)
        x = self.downsample1(x)

        x = self.expand2(x)
        x = self.zero_conv1(x)

        outputs.append(x)
        x = self.downsample2(x)

        x = self.expand3(x)
        x = self.zero_conv2(x)

        outputs.append(x)
        return tuple(outputs)

class ConditionedUNet(UNet):
    def __init__(self, in_channels=4, base_dim=128) -> None:
        super().__init__(in_channels, base_dim)
        self.adapter = Adapter()
        self.hints = None
    def forward_with_hints(self, x, time, hints):
        t = self.time_mlp(time)
        if self.hints is None:
            self.hints = self.adapter(hints)

        adapter_16x16_hint, adapter_8x8_hint = self.hints
        # Encoder section 
        # Notice how, for both the encoder and decoder, attn is only used at resolutions below 32x32

        # 32x32
        x1 = self.in_conv(x)
        x2 = self.enc_resblk1(x1, t)
        x3 = self.enc_resblk2(x2, t)

        x4 = self.downsample1(x3) # 32x32 -> 16x16
        
        # 16x16
        x5 = self.enc_resblk3(x4, t)
        x5 = x5 + adapter_16x16_hint
        x5 = self.enc_attn1(x5)
        x6 = self.enc_resblk4(x5, t)
        x6 = self.enc_attn2(x6)

        x7 = self.downsample2(x6) # 16x16 -> 8x8


        # Bottleneck section

        # 8x8
        x8 = self.mid_resblk1(x7, t)
        x8 = x8 + adapter_8x8_hint
        x8 = self.mid_attn(x8)
        x8 = self.mid_resblk2(x8, t)

        # Decoder section
        # For the decoder section, we concatenate skip connections from encoder at same resolution

        up1 = self.upsample1(x8) # 8x8 -> 16x16

        # 16x16
        up1 = torch.cat([up1, x6], dim=1) # Concatenate along channels dim
        up1 = self.dec_resblk1(up1, t)
        up1 = self.dec_attn1(up1)

        up1 = torch.cat([up1, x5], dim=1)
        up1 = self.dec_resblk2(up1, t)
        up1 = self.dec_attn2(up1)

        up2 = self.upsample2(up1) # 16x16 -> 32x32

        # 32x32
        up2 = torch.cat([up2, x3], dim=1)
        up2 = self.dec_resblk3(up2, t)
        
        up2 = torch.cat([up2, x2], dim=1)
        up2 = self.dec_resblk4(up2, t)

        # Output
        out = self.out_norm(up2)
        out = self.out_act(out)
        return self.out_conv(out)



if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import OneCycleLR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load("./adapter_map_tensors.pt", weights_only=False)
    latents_key, hints_key, std, mean = data

    clean_latents = data[latents_key].to(device)
    hints = data[hints_key]
    std = clean_latents.std()
    mean = clean_latents.mean()

    clean_latents = (clean_latents - mean) / std

    class AdapterDataset(Dataset):
        def __init__(self, latents: torch.Tensor, hints: torch.Tensor) -> None:
            self.latents= latents
            self.hints = hints
        def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
            latents = self.latents[index]
            hints = self.hints[index]
            return (latents, hints)
        def __len__(self):
            return len(self.hints)

    dataset = AdapterDataset(clean_latents, hints)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    unet = ConditionedUNet(in_channels=16).train().to(device)

    scheduler = DDPMScheduler()

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    l1_weight = 0.8
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-2)

    epoch = 0
    epochs = 35
    total_steps = len(loader) * epochs
    lr_scheduler = OneCycleLR(
                    optimizer,
                    max_lr=3e-4,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='cos'
                   )
    
    checkpoint = torch.load("./checkpoints/Attempt 4/UNet/UNet_400.pth", map_location=device)
    missing_keys, unexpected_keys = unet.load_state_dict(checkpoint["model_state_dict"], strict=False)
    expected_missing_keys = ["adapter.expand1.0.weight", "adapter.expand1.0.bias", "adapter.expand1.2.weight", "adapter.expand1.2.bias", "adapter.expand2.0.weight", "adapter.expand2.0.bias", "adapter.downsample1.0.weight", "adapter.downsample1.0.bias", "adapter.downsample2.0.weight", "adapter.downsample2.0.bias", "adapter.zero_conv1.weight", "adapter.zero_conv1.bias", "adapter.zero_conv2.weight", "adapter.zero_conv2.bias", "adapter.expand3.0.weight", "adapter.expand3.0.bias"]

    for key in missing_keys:
        if key not in expected_missing_keys:
            raise RuntimeError(f"Missing key in state_dict: {key}")

    if unexpected_keys:
        raise RuntimeError(f"Found unexpected key(s) in state_dict: {unexpected_keys}")

    unet.requires_grad_(False)
    unet.adapter.requires_grad_(True)

    for epoch in range(epoch, epochs):

        print(f"Epoch {epoch}/{epochs}", end="\n\n")
        total_loss = 0
        for batch_idx, (latents,hints) in enumerate(loader):
            optimizer.zero_grad()
            hints = hints.to(device)
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
            
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            predicted_noise = unet.forward_with_hints(noisy_latents, timesteps, hints)
            loss = l1_weight * l1_loss(predicted_noise, noise) + (1 - l1_weight) * mse_loss(predicted_noise, noise)

            loss_val = loss.item()
            total_loss += loss_val
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            unet.hints = None

            if batch_idx == 0:
                print(f"Batch: {batch_idx}/{len(loader)}. Loss: {loss_val}.")
            elif ((batch_idx + 1) % 20) == 0:
                print(f"Batch: {batch_idx + 1}/{len(loader)}. Loss: {loss_val}.")
        
        print(f"Average loss: {total_loss/len(loader)}", end="\n\n")

        if (epoch + 1) % 5 == 0:
            checkpoint = {
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"./checkpoints/Attempt 4/UNet/UNet_{epoch + 1}_Conditioned.pth")
            print(f"Saving checkpoint to './checkpoints/Attempt 4/UNet/UNet_{epoch + 1}_Conditioned.pth'")


    print("Finished training.")
    torch.save(unet.state_dict(), f"./checkpoints/Attempt 4/UNet/UNet_Conditioned.pth")