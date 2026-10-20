"""Ancestral DDPM reverse update."""

import torch
from torch import nn

from diffusion.samplers.base import Sampler
from diffusion.schedulers.base import NoiseScheduler


class DDPMSampler(Sampler):
    def inference_timesteps(self, scheduler: NoiseScheduler, num_inference_steps: int | None,
                            device: torch.device) -> torch.Tensor:
        if num_inference_steps not in (None, scheduler.num_train_timesteps):
            raise ValueError("DDPM uses all training timesteps; choose DDIM to skip steps")
        return torch.arange(scheduler.num_train_timesteps - 1, -1, -1, device=device)

    @torch.no_grad()
    def step(self, model: nn.Module, noisy_images: torch.Tensor, timestep: int,
             previous_timestep: int, scheduler: NoiseScheduler,
             generator: torch.Generator | None = None) -> torch.Tensor:
        ts = torch.full((noisy_images.shape[0],), timestep, device=noisy_images.device, dtype=torch.long)
        beta = scheduler.betas[timestep]
        alpha = scheduler.alphas[timestep]
        alpha_bar = scheduler.alpha_bars[timestep]
        predicted_noise = model(noisy_images, ts)
        if scheduler.prediction_type != "epsilon":
            raise ValueError(f"DDPM sampler does not support {scheduler.prediction_type} predictions")
        predicted_clean = (noisy_images - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        if scheduler.clip_sample:
            limit = scheduler.clip_sample_range
            predicted_clean = predicted_clean.clamp(-limit, limit)
        alpha_bar_previous = scheduler.alpha_bars_prev[timestep]
        mean = (beta * alpha_bar_previous.sqrt() / (1 - alpha_bar) * predicted_clean +
                (1 - alpha_bar_previous) * alpha.sqrt() / (1 - alpha_bar) * noisy_images)
        if previous_timestep < 0:
            return mean
        if scheduler.variance_type in {"fixed_large", "fixed_large_log"}:
            variance = beta
        elif scheduler.variance_type in {"fixed_small", "fixed_small_log"}:
            variance = beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
        else:
            raise ValueError(f"Unsupported DDPM variance type: {scheduler.variance_type}")
        noise = torch.randn(noisy_images.shape, device=noisy_images.device,
                            dtype=noisy_images.dtype, generator=generator)
        return mean + variance.clamp(min=1e-20).sqrt() * noise
