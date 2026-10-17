"""
Standard U-Net Architecture for Denoising Diffusion Probabilistic Models (DDPM).
Includes:
- Sinusoidal Timestep Embeddings
- Residual Blocks with GroupNorm and SiLU
- Multi-Head Self-Attention Blocks at specified resolutions
- Downsampling and Upsampling with Skip Connections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal position embedding for diffusion timesteps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for spatial feature maps.
    """
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        norm_x = self.norm(x)
        
        qkv = self.qkv_conv(norm_x) # (B, 3*C, H, W)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # each: (B, heads, head_dim, N)

        # Transpose for dot product: (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.transpose(-2, -1)) # (B, heads, N, head_dim)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        out = self.proj_conv(out)
        return residual + out


class ResnetBlock(nn.Module):
    """
    Residual block with time embedding injection and optional dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        num_groups: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=min(num_groups, in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        
        # Inject time embedding
        time_proj = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_proj

        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.residual_conv(x)


class Downsample(nn.Module):
    """Downsample feature map by factor of 2."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample feature map by factor of 2 using nearest interpolation + conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for diffusion models.
    Supports flexible channel configurations, depth, and attention resolutions.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        attention_resolutions: List[int] = [1],
        num_res_blocks: int = 2,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if time_emb_dim is None:
            time_emb_dim = base_channels * 4
        self.time_emb_dim = time_emb_dim

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList()
        current_channels = base_channels
        channels_list = [current_channels]

        for level, mult in enumerate(channel_multipliers):
            level_channels = base_channels * mult
            for _ in range(num_res_blocks):
                block = ResnetBlock(current_channels, level_channels, time_emb_dim, dropout=dropout)
                current_channels = level_channels
                attn = AttentionBlock(current_channels) if level in attention_resolutions else nn.Identity()
                self.downs.append(nn.ModuleList([block, attn]))
                channels_list.append(current_channels)
            
            # Downsample between levels except last
            if level != len(channel_multipliers) - 1:
                self.downs.append(Downsample(current_channels))
                channels_list.append(current_channels)

        # Middle bottleneck
        self.mid_block1 = ResnetBlock(current_channels, current_channels, time_emb_dim, dropout=dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid_block2 = ResnetBlock(current_channels, current_channels, time_emb_dim, dropout=dropout)

        # Upsampling path
        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            level_channels = base_channels * mult
            for _ in range(num_res_blocks + 1):
                skip_channels = channels_list.pop()
                block = ResnetBlock(current_channels + skip_channels, level_channels, time_emb_dim, dropout=dropout)
                current_channels = level_channels
                attn = AttentionBlock(current_channels) if level in attention_resolutions else nn.Identity()
                self.ups.append(nn.ModuleList([block, attn]))

            if level != 0:
                self.ups.append(Upsample(current_channels))

        # Final projection to output noise
        self.norm_out = nn.GroupNorm(num_groups=min(8, current_channels), num_channels=current_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
            t: Discrete timesteps of shape (B,)
        Returns:
            Noise prediction epsilon_theta(x, t) of shape (B, C, H, W)
        """
        t_emb = self.time_mlp(t)
        h = self.conv_in(x)
        skips = [h]

        for down in self.downs:
            if isinstance(down, Downsample):
                h = down(h)
                skips.append(h)
            else:
                block, attn = down
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for up in self.ups:
            if isinstance(up, Upsample):
                h = up(h)
            else:
                block, attn = up
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)

        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)

