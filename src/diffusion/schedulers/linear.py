"""Linear beta schedule from the original DDPM baseline."""

import torch

from diffusion.schedulers.base import NoiseScheduler


class LinearScheduler(NoiseScheduler):
    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 2e-2):
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(num_train_timesteps)

    def make_betas(self, num_train_timesteps: int) -> torch.Tensor:
        return torch.linspace(self.beta_start, self.beta_end, num_train_timesteps)
