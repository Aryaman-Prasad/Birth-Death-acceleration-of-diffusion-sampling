"""Analytic epsilon predictor for the known toy Gaussian mixture."""

import torch

from diffusion.models.base import DiffusionModel
from diffusion.targets.toy_mixture import ToyGaussianMixtureTarget


class ToyMixtureOracleModel(DiffusionModel):
    """Return the exact epsilon implied by the noisy toy-mixture score."""

    def __init__(self, target: ToyGaussianMixtureTarget, alpha_bars: torch.Tensor):
        super().__init__()
        self.target = target
        self.register_buffer("alpha_bars", alpha_bars.detach().clone())

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].to(x)
        broadcast_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        alpha_bar = alpha_bar.reshape(broadcast_shape)
        sigma = ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()
        normalized_state = x / alpha_bar.sqrt()
        score = self.target.score_at_noise(normalized_state, sigma.reshape(-1))
        return -sigma * score
