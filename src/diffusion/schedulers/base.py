"""Base class for discrete variance schedules used by samplers and training."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class NoiseScheduler(nn.Module, ABC):
    def __init__(self, num_train_timesteps: int):
        super().__init__()
        if num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least 2")
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = "epsilon"
        self.variance_type = "fixed_small"
        self.clip_sample = False
        self.clip_sample_range = 1.0
        betas = self.make_betas(num_train_timesteps)
        if betas.shape != (num_train_timesteps,) or not torch.all((betas > 0) & (betas < 1)):
            raise ValueError("make_betas must return one beta in (0, 1) for each training timestep")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", torch.cat([torch.ones(1), alpha_bars[:-1]]))

    @abstractmethod
    def make_betas(self, num_train_timesteps: int) -> torch.Tensor:
        """Create the beta sequence for this scheduler."""

    def extract(self, values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (reference.ndim - 1)))

    def add_noise(self, clean_images: torch.Tensor, timesteps: torch.Tensor,
                  noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_images) if noise is None else noise
        alpha_bar = self.extract(self.alpha_bars, timesteps, clean_images)
        noisy = alpha_bar.sqrt() * clean_images + (1 - alpha_bar).sqrt() * noise
        return noisy, noise

    def training_loss(self, model: nn.Module, clean_images: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.num_train_timesteps, (clean_images.shape[0],), device=clean_images.device)
        noisy, noise = self.add_noise(clean_images, timesteps)
        return torch.nn.functional.mse_loss(model(noisy, timesteps), noise)
