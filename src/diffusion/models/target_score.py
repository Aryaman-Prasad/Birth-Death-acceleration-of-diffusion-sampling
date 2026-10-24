"""Epsilon-prediction adapter for targets with an explicit score."""

import torch

from diffusion.models.base import DiffusionModel
from diffusion.schedulers.base import NoiseScheduler
from diffusion.targets.base import TargetDistribution


class TargetScoreModel(DiffusionModel):
    """Expose a target's normalized-coordinate score as a DDPM epsilon model.

    This adapter lets the existing samplers operate on targets whose score is
    known directly, including analytic toy targets and the empirical Gaussian
    mixture built from a finite dataset subset. ``TargetDistribution.score``
    is defined on ``z = x_t / sqrt(alpha_bar_t)``; epsilon prediction follows
    from ``epsilon = -sqrt((1-alpha_bar_t)/alpha_bar_t) * score(z)``.
    """

    def __init__(self, target: TargetDistribution, scheduler: NoiseScheduler):
        super().__init__()
        self.target = target
        self.scheduler = scheduler

    def forward(self, noisy_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.numel() == 0:
            raise ValueError("timesteps must contain at least one timestep")
        timestep = int(timesteps.reshape(-1)[0].item())
        if not torch.all(timesteps == timestep):
            raise ValueError("TargetScoreModel expects one shared timestep for the whole batch")

        alpha_bar = self.scheduler.alpha_bars[timestep].to(noisy_samples)
        normalized_samples = noisy_samples / alpha_bar.sqrt()
        score = self.target.score(normalized_samples, timestep, self.scheduler)
        sigma = ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()
        return -sigma * score
