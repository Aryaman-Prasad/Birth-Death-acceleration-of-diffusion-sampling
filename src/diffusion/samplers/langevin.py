"""Langevin samplers driven by epsilon-prediction diffusion models."""

import torch
from torch import nn

from diffusion.samplers.base import Sampler
from diffusion.schedulers.base import NoiseScheduler


def _noise_std(scheduler: NoiseScheduler, timestep: int) -> torch.Tensor:
    alpha_bar = scheduler.alpha_bars[timestep]
    return ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()


def _score_in_normalized_coordinates(model: nn.Module, samples: torch.Tensor,
                                     timestep: int, scheduler: NoiseScheduler) -> torch.Tensor:
    """Use an epsilon model to estimate score of z=x_t/sqrt(alpha_bar_t)."""
    if scheduler.prediction_type != "epsilon":
        raise ValueError("Langevin samplers require an epsilon-prediction model")
    alpha_bar = scheduler.alpha_bars[timestep]
    sigma = _noise_std(scheduler, timestep)
    # Scheduler validation guarantees beta_t > 0, hence alpha_bar_t < 1 and
    # sigma_t > 0. Avoid a scalar device sync in this hot sampling path.
    x_t = alpha_bar.sqrt() * samples
    times = torch.full((samples.shape[0],), timestep, device=samples.device, dtype=torch.long)
    predicted_epsilon = model(x_t, times)
    return -predicted_epsilon / sigma


class LangevinSampler(Sampler):
    """Unadjusted Langevin algorithm for a fixed diffusion noise level.

    The chain evolves in normalized coordinates ``z=x_t/sqrt(alpha_bar_t)``.
    A known target can therefore provide an exact score and density at the
    chosen training timestep, which is useful for controlled particle-method
    experiments.
    """

    def __init__(self, timestep: int = 0, step_size: float = 1e-3,
                 initial_scale: float = 1.0):
        if timestep < 0:
            raise ValueError("timestep must be non-negative")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if initial_scale <= 0:
            raise ValueError("initial_scale must be positive")
        self.timestep = timestep
        self.step_size = step_size
        self.initial_scale = initial_scale

    def inference_timesteps(self, scheduler: NoiseScheduler, num_inference_steps: int | None,
                            device: torch.device) -> torch.Tensor:
        if self.timestep >= scheduler.num_train_timesteps:
            raise ValueError("Langevin timestep must be less than the scheduler timestep count")
        count = scheduler.num_train_timesteps if num_inference_steps is None else num_inference_steps
        if count < 1:
            raise ValueError("num_inference_steps must be positive")
        return torch.full((count,), self.timestep, device=device, dtype=torch.long)

    def initial_samples(self, shape: tuple[int, ...], device: torch.device,
                        scheduler: NoiseScheduler, timesteps: torch.Tensor,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        return self.initial_scale * torch.randn(shape, device=device, generator=generator)

    def sampling_metadata(self, timestep: int, previous_timestep: int,
                          scheduler: NoiseScheduler) -> dict:
        return {
            "birth_death_delta_t": self.step_size,
            "target_timestep": self.timestep,
            "state_coordinates": "normalized_x_t",
        }

    @torch.no_grad()
    def step(self, model: nn.Module, noisy_images: torch.Tensor, timestep: int,
             previous_timestep: int, scheduler: NoiseScheduler,
             generator: torch.Generator | None = None) -> torch.Tensor:
        score = _score_in_normalized_coordinates(model, noisy_images, timestep, scheduler)
        noise = torch.randn(noisy_images.shape, device=noisy_images.device,
                            dtype=noisy_images.dtype, generator=generator)
        return noisy_images + self.step_size * score + (2 * self.step_size) ** 0.5 * noise


class AnnealedLangevinSampler(Sampler):
    """Annealed Langevin dynamics over a descending DDPM noise schedule.

    Each noise level gets ``steps_per_level`` ULA updates. The state is kept in
    normalized coordinates, and step sizes scale quadratically with the
    current noise relative to the minimum scheduled noise level.
    """

    def __init__(self, step_size: float = 1e-7, steps_per_level: int = 1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if steps_per_level < 1:
            raise ValueError("steps_per_level must be positive")
        self.step_size = step_size
        self.steps_per_level = steps_per_level

    def inference_timesteps(self, scheduler: NoiseScheduler, num_inference_steps: int | None,
                            device: torch.device) -> torch.Tensor:
        levels = scheduler.num_train_timesteps if num_inference_steps is None else num_inference_steps
        if not 1 <= levels <= scheduler.num_train_timesteps:
            raise ValueError("noise-level count must be between 1 and training timestep count")
        level_indices = torch.linspace(
            scheduler.num_train_timesteps - 1, 0, levels, device=device
        ).round().long()
        return level_indices.repeat_interleave(self.steps_per_level)

    def initial_samples(self, shape: tuple[int, ...], device: torch.device,
                        scheduler: NoiseScheduler, timesteps: torch.Tensor,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        sigma_max = _noise_std(scheduler, int(timesteps[0].item())).to(device=device)
        return sigma_max * torch.randn(shape, device=device, generator=generator)

    @torch.no_grad()
    def step(self, model: nn.Module, noisy_images: torch.Tensor, timestep: int,
             previous_timestep: int, scheduler: NoiseScheduler,
             generator: torch.Generator | None = None) -> torch.Tensor:
        score = _score_in_normalized_coordinates(model, noisy_images, timestep, scheduler)
        sigma = _noise_std(scheduler, timestep).to(noisy_images)
        sigma_min = _noise_std(scheduler, 0).to(noisy_images)
        delta_t = self.step_size * (sigma / sigma_min).square()
        noise = torch.randn(noisy_images.shape, device=noisy_images.device,
                            dtype=noisy_images.dtype, generator=generator)
        return noisy_images + delta_t * score + (2 * delta_t).sqrt() * noise
