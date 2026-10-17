"""
Core Denoising Diffusion Probabilistic Model (DDPM) module.
Handles forward noising process, diffusion math, loss calculation, and parameter storage.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union

from src.diffusion.schedules import get_schedule, precompute_diffusion_constants


def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from arr at timesteps and reshape to broadcast_shape.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion framework (Ho et al., 2020).
    Wraps a score/noise-prediction network and manages the forward diffusion process.
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.loss_type = loss_type

        # Generate schedule and precomputed terms
        schedule_kwargs = {}
        if beta_schedule == "linear":
            schedule_kwargs = {"beta_start": beta_start, "beta_end": beta_end}
        betas = get_schedule(beta_schedule, timesteps, **schedule_kwargs)
        constants = precompute_diffusion_constants(betas)

        # Register all constants as persistent buffers
        for key, tensor in constants.items():
            self.register_buffer(key, tensor)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward process: Sample x_t from q(x_t | x_0) analytically.
        x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_x_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x_0 from x_t and predicted noise epsilon:
        x_0 = (x_t - sqrt(1 - alphas_cumprod_t) * noise) / sqrt(alphas_cumprod_t)
        """
        sqrt_recip_alphas_cumprod_t = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance of the true posterior q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute estimated reverse step mean mu_theta(x_t, t) and variance sigma^2_t using model.
        """
        pred_noise = self.model(x_t, t)
        pred_x_start = self.predict_x_start_from_noise(x_t, t, pred_noise)

        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=pred_x_start, x_t=x_t, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, pred_x_start

    def training_losses(self, x_start: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training loss L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ].
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device).long()
        
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.model(x_noisy, t)

        if self.loss_type == "mse":
            loss = F.mse_loss(pred_noise, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred_noise, noise)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return {"loss": loss, "pred_noise": pred_noise, "target_noise": noise, "timesteps": t}

    def forward(self, x_start: torch.Tensor) -> torch.Tensor:
        losses = self.training_losses(x_start)
        return losses["loss"]

