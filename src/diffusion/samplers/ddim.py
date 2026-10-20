"""DDIM reverse update, including reduced inference step counts."""

import torch
from torch import nn

from diffusion.samplers.base import Sampler
from diffusion.schedulers.base import NoiseScheduler


class DDIMSampler(Sampler):
    def __init__(self, eta: float = 0.0):
        if eta < 0:
            raise ValueError("eta must be non-negative")
        self.eta = eta

    def inference_timesteps(self, scheduler: NoiseScheduler, num_inference_steps: int | None,
                            device: torch.device) -> torch.Tensor:
        count = scheduler.num_train_timesteps if num_inference_steps is None else num_inference_steps
        if not 1 <= count <= scheduler.num_train_timesteps:
            raise ValueError("num_inference_steps must be between 1 and training timestep count")
        return torch.linspace(scheduler.num_train_timesteps - 1, 0, count, device=device).round().long()

    @torch.no_grad()
    def step(self, model: nn.Module, noisy_images: torch.Tensor, timestep: int,
             previous_timestep: int, scheduler: NoiseScheduler,
             generator: torch.Generator | None = None) -> torch.Tensor:
        ts = torch.full((noisy_images.shape[0],), timestep, device=noisy_images.device, dtype=torch.long)
        alpha_bar = scheduler.alpha_bars[timestep]
        previous_alpha_bar = (scheduler.alpha_bars[previous_timestep] if previous_timestep >= 0
                              else scheduler.alpha_bars.new_tensor(1.0))
        predicted_noise = model(noisy_images, ts)
        if scheduler.prediction_type != "epsilon":
            raise ValueError(f"DDIM sampler does not support {scheduler.prediction_type} predictions")
        predicted_clean = (noisy_images - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        if scheduler.clip_sample:
            limit = scheduler.clip_sample_range
            predicted_clean = predicted_clean.clamp(-limit, limit)
        sigma = self.eta * torch.sqrt(((1 - previous_alpha_bar) / (1 - alpha_bar) *
                                      (1 - alpha_bar / previous_alpha_bar)).clamp(min=0))
        direction = (1 - previous_alpha_bar - sigma.square()).clamp(min=0).sqrt() * predicted_noise
        result = previous_alpha_bar.sqrt() * predicted_clean + direction
        if self.eta and previous_timestep >= 0:
            noise = torch.randn(noisy_images.shape, device=noisy_images.device,
                                dtype=noisy_images.dtype, generator=generator)
            result = result + sigma * noise
        return result
