"""Cosine cumulative alpha schedule."""

import torch

from diffusion.schedulers.base import NoiseScheduler


class CosineScheduler(NoiseScheduler):
    def make_betas(self, num_train_timesteps: int) -> torch.Tensor:
        steps = torch.arange(num_train_timesteps + 1, dtype=torch.float64) / num_train_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999).float()
