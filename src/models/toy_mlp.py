"""
Time-conditioned MLP model for 2D synthetic multimodal diffusion.
Ideal for rapid prototyping, phase-portrait trajectory visualization,
and benchmarking birth-death dynamics across energy barriers and modes.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal timestep embeddings from 'Attention Is All You Need' / DDPM.
    Maps discrete timestep integer t to an embedding vector.
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


class ResBlockMLP(nn.Module):
    """
    Residual block with time embedding injection for 2D data.
    """
    def __init__(self, dim: int, time_emb_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(self.fc1(h))
        h = h + self.time_mlp(t_emb)
        h = self.norm2(h)
        h = self.act(self.fc2(h))
        return x + h


class ToyScoreMLP(nn.Module):
    """
    Score-matching / Noise prediction network for 2D coordinates (x in R^2).
    Predicts the noise epsilon ~ N(0, I) added at timestep t.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        time_emb_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResBlockMLP(hidden_dim, time_emb_dim)
            for _ in range(num_layers)
        ])
        
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 2)
            t: Timesteps tensor of shape (B,)
        Returns:
            Noise prediction epsilon_theta(x, t) of shape (B, 2)
        """
        t_emb = self.time_embed(t)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.out_norm(h)
        out = self.out_proj(h)
        return out

