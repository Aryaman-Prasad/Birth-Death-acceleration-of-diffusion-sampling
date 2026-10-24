"""Exact noisy density and score for the framework's 2D toy mixture."""

import math

import torch
from torch import nn

from diffusion.targets.base import TargetDistribution


class ToyGaussianMixtureTarget(nn.Module, TargetDistribution):
    """Eight equal isotropic Gaussian modes on a ring.

    The diffusion scheduler induces a Gaussian convolution of the clean data
    distribution. In normalized coordinates ``z = x_t / sqrt(alpha_bar_t)``,
    its standard deviation is ``sqrt((1 - alpha_bar_t) / alpha_bar_t)``. The
    resulting mixture has a closed-form log density and score at every time.
    """

    def __init__(self, radius: float = 2.0, standard_deviation: float = 0.15,
                 num_modes: int = 8):
        super().__init__()
        if radius <= 0 or standard_deviation <= 0 or num_modes < 2:
            raise ValueError("radius and standard_deviation must be positive; num_modes must be >= 2")
        angles = torch.arange(num_modes, dtype=torch.float32) * (2 * math.pi / num_modes)
        centers = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
        self.register_buffer("centers", centers)
        self.register_buffer("standard_deviation", torch.tensor(float(standard_deviation)))

    def noise_std(self, timestep: int, scheduler) -> torch.Tensor:
        alpha_bar = scheduler.alpha_bars[timestep]
        return ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()

    def _component_log_prob(self, samples: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        flat = samples.reshape(samples.shape[0], -1)
        if flat.shape[1] != self.centers.shape[1]:
            raise ValueError(
                f"Toy mixture target expects {self.centers.shape[1]}-D samples, got {flat.shape[1]}-D"
            )
        sigma = torch.as_tensor(sigma, device=flat.device, dtype=flat.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(flat.shape[0])
        sigma = sigma.reshape(-1)
        if sigma.numel() != flat.shape[0]:
            raise ValueError("sigma must be scalar or provide one value per sample")

        variance = self.standard_deviation.to(flat).square() + sigma.square()
        difference = flat[:, None, :] - self.centers.to(flat)[None, :, :]
        dimensions = flat.shape[1]
        return (-0.5 * difference.square().sum(dim=-1) / variance[:, None]
                - 0.5 * dimensions * torch.log(2 * math.pi * variance[:, None])
                - math.log(self.centers.shape[0]))

    def log_prob_at_noise(self, samples: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact mixture log density at a supplied noise scale."""
        return torch.logsumexp(self._component_log_prob(samples, sigma), dim=1)

    def score_at_noise(self, samples: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate the exact gradient of the noisy mixture log density."""
        flat = samples.reshape(samples.shape[0], -1)
        component_log_prob = self._component_log_prob(samples, sigma)
        responsibilities = torch.softmax(component_log_prob, dim=1)
        sigma = torch.as_tensor(sigma, device=flat.device, dtype=flat.dtype)
        if sigma.ndim == 0:
            sigma = sigma.expand(flat.shape[0])
        variance = self.standard_deviation.to(flat).square() + sigma.reshape(-1).square()
        component_scores = (self.centers.to(flat)[None, :, :] - flat[:, None, :]) / variance[:, None, None]
        score = (responsibilities[:, :, None] * component_scores).sum(dim=1)
        return score.reshape_as(samples)

    def log_prob(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        return self.log_prob_at_noise(samples, self.noise_std(timestep, scheduler))

    def score(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        return self.score_at_noise(samples, self.noise_std(timestep, scheduler))
