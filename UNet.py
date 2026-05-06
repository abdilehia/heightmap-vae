import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# I wrote the Upsample, Downsample and MultiHeadAttention and the training loop on my own
# The rest was heavily aided by AI
# I still did my best to understand what the code does and write it myself
# Unfortunately, at least from my view, that does not change that AI was responsible for it

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class SinusoidalPositionEmbeddings(nn.Module): 
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time): # Math derived from "Attention is All You Need"
        device = time.device
        half_dim = self.dim // 2

        # Calculate frequencies
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # Multiply time steps by embeddings
        embeddings = time[:, None] * embeddings[None, :]

        # Interleave sine and cosine
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bsz = q.size(0)

        # (bsz, size, d_model) -> (bsz, size, num_heads, d_head) -> (bsz, num_heads, size, d_head)
        Q = self.w_q(q).view(bsz, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(bsz, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(bsz, -1, self.num_heads, self.d_head).transpose(1, 2)

        # scaled dot product attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)        
        weights = F.softmax(scores, dim=-1)
        attention_scores = weights @ V
        
        # (bsz, num_heads, size, d_head) -> (bsz, size, num_heads, d_head) -> (bsz, size, d_model)
        combined_attention_scores = attention_scores.transpose(2, 1).contiguous().view(bsz, -1, self.d_model)

        output = self.fc(combined_attention_scores)

        return output

class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4) -> None:
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.mha = MultiHeadAttention(channels, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)


        x = x.view(B, C, H*W).transpose(1, 2) # (B, C, H, W) -> (B, C, H*W) -> (BB, H*W, C)
        x = self.mha(x, x, x)
        x = x.transpose(2, 1).view(B, C, H, W) # (BB, H*W, C) -> (B, C, H*W) -> (B, C, H, W)

        return residual + x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        # Residual connection (original input)
        # Conv layer needed if original input dim does not match output dim
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()


    def forward(self, x, t):

        h = self.layer1(x)

        # Get time embedding and expand so we can add to every pixel
        t_emb = self.time_proj(t) # (bsz, out_channels)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1) # (bsz, out_channels, 1, 1)
        h = h + t_emb

        h = self.layer2(h)

        return h + self.residual(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, base_dim = 128) -> None:
        super().__init__()

        self.time_dim = base_dim * 4 # 4x base channel count

        # Time embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

        # Encoder
        self.in_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        self.enc_resblk1 = ResBlock(base_dim, base_dim, self.time_dim)
        self.enc_resblk2 = ResBlock(base_dim, base_dim, self.time_dim)
            
        self.downsample1 = Downsample(base_dim, base_dim * 2) # 32x32 -> 16x16
        self.enc_resblk3 = ResBlock(base_dim * 2, base_dim * 2, self.time_dim)
        self.enc_attn1 = SpatialSelfAttention(base_dim * 2) # Spatial Attention
        self.enc_resblk4 = ResBlock(base_dim * 2, base_dim * 2, self.time_dim)
        self.enc_attn2 = SpatialSelfAttention(base_dim * 2) # Spatial Attention

        self.downsample2 = Downsample(base_dim * 2, base_dim * 4) # 16x16 -> 8x8

        # Bottleneck
        self.mid_resblk1 = ResBlock(base_dim * 4, base_dim * 4, self.time_dim)
        self.mid_attn = SpatialSelfAttention(base_dim * 4)
        self.mid_resblk2 = ResBlock(base_dim * 4, base_dim * 4, self.time_dim)

        # Decoder
        self.upsample1 = Upsample(base_dim * 4) # 8x8 -> 16x16
        self.dec_resblk1 = ResBlock(base_dim * 4 + base_dim * 2, base_dim * 2, self.time_dim)
        self.dec_attn1 = SpatialSelfAttention(base_dim * 2)
        self.dec_resblk2 = ResBlock(base_dim * 2 + base_dim * 2, base_dim * 2, self.time_dim)
        self.dec_attn2 = SpatialSelfAttention(base_dim * 2)

        self.upsample2 = Upsample(base_dim * 2) # 16x16 -> 32x32
        self.dec_resblk3 = ResBlock(base_dim * 2 + base_dim, base_dim, self.time_dim)
        self.dec_resblk4 = ResBlock(base_dim + base_dim, base_dim, self.time_dim)

        # Output
        self.out_norm = nn.GroupNorm(32, base_dim)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        t = self.time_mlp(time)

        # Encoder section 
        # Notice how, for both the encoder and decoder, attn is only used at resolutions below 32x32

        # 32x32
        x1 = self.in_conv(x)
        x2 = self.enc_resblk1(x1, t)
        x3 = self.enc_resblk2(x2, t)

        x4 = self.downsample1(x3) # 32x32 -> 16x16
        
        # 16x16
        x5 = self.enc_resblk3(x4, t)
        x5 = self.enc_attn1(x5)
        x6 = self.enc_resblk4(x5, t)
        x6 = self.enc_attn2(x6)

        x7 = self.downsample2(x6) # 16x16 -> 8x8


        # Bottleneck section

        # 8x8
        x8 = self.mid_resblk1(x7, t)
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

class DDPMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.2) -> None:
        super().__init__()

        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, steps=num_train_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, original, noise, timestep):
        device = original.device
        bsz = original.size(0)

        self.alphas_cumprod = self.alphas_cumprod.to(device)

        ratio = torch.sqrt(self.alphas_cumprod[timestep])
        ratio = ratio.view(bsz, 1, 1, 1) # Need to match the input shape

        noisy_samples = ratio * original + (1-ratio) * noise
        return noisy_samples

    def step(self, model_output, timestep, sample):
        """
        model_output: The noise predicted by the U-Net
        timestep: The current integer step (e.g., 999 down to 0)
        sample: The current noisy latent
        """
        t = timestep
        
        # Grab the math constants for this specific timestep
        alpha_t = self.alphas[t]
        alpha_prod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # 1. The DDPM Formula to remove the predicted noise
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_prod_t)
        
        prev_sample = coef1 * (sample - (coef2 * model_output))

        # 2. Add fresh noise (unless we are at the very last step t=0)
        if t > 0:
            noise = torch.randn_like(sample)
            # The standard DDPM variance is simply sqrt(beta_t)
            prev_sample = prev_sample + (torch.sqrt(beta_t) * noise)

        return prev_sample
        
    def step_ddim(self, model_output, timestep, prev_timestep, sample):
        """
        A deterministic DDIM step that actually handles skipped steps properly.
        """
        t = timestep
        prev_t = prev_timestep
        
        alpha_prod_t = self.alphas_cumprod[t]
        
        # If prev_t is below 0, it means we are at the final clean image, so alpha is 1.0
        if prev_t >= 0:
            alpha_prod_t_prev = self.alphas_cumprod[prev_t]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=sample.device)
            
        # 1. Predict the original sample (x0)
        pred_original_sample = (sample - torch.sqrt(1.0 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        
        pred_original_sample = torch.clamp(pred_original_sample, -3.0, 3.0)
        # 2. Point towards the noisy image at the previous timestep
        dir_xt = torch.sqrt(1.0 - alpha_prod_t_prev) * model_output
        
        # 3. Calculate the previous sample
        prev_sample = (torch.sqrt(alpha_prod_t_prev) * pred_original_sample) + dir_xt
        
        return prev_sample


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import OneCycleLR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load("./map_tensors.pt", weights_only=False)
    latents_key, std, mean = data

    clean_latents = data[latents_key].to(device)
    std = clean_latents.std()
    mean = clean_latents.mean()

    clean_latents = (clean_latents - mean) / std

    dataset = TensorDataset(clean_latents)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    unet = UNet().train().to(device)
    scheduler = DDPMScheduler()

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-4)

    accumulation_steps = 2
    epochs = 400
    total_steps = (len(loader) // accumulation_steps) * epochs

    lr_scheduler = OneCycleLR(
                    optimizer,
                    max_lr=1e-4,
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='cos'
                   )
    
    for epoch in range(epochs):

        print(f"Epoch {epoch}/{epochs}", end="\n\n")
        total_loss = 0
        optimizer.zero_grad()

        for batch_idx, (latents,) in enumerate(loader):
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device).long()
            
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            predicted_noise = unet(noisy_latents, timesteps)
            loss = mse_loss(predicted_noise, noise)

            loss_val = loss.item()
            total_loss += loss_val

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            
            if batch_idx == 0:
                print(f"Batch: {batch_idx}/{len(loader)}. Loss: {loss_val}.")
            elif ((batch_idx + 1) % 20) == 0:
                print(f"Batch: {batch_idx + 1}/{len(loader)}. Loss: {loss_val}.")
        
        print(f"Average loss: {total_loss/len(loader)}", end="\n\n")

        if (epoch + 1) % 50 == 0:
            checkpoint = {
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"./checkpoints/UNet/UNet_{epoch + 1}.pth")
            print(f"Saving checkpoint to './checkpoints/UNet/UNet_{epoch + 1}.pth'")


    print("Finished training.")
    torch.save(unet.state_dict(), f"./checkpoints/UNet/UNet.pth")