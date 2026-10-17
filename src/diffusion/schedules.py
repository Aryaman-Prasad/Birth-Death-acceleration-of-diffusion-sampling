"""
Noise variance schedules for diffusion models.
Supports Linear, Cosine (Nichol & Dhariwal), and Sigmoid schedules.
"""

import math
import torch
from typing import Dict


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, **kwargs) -> torch.Tensor:
    """
    Standard linear schedule proposed in Ho et al. (DDPM, 2020).
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008, max_beta: float = 0.999, **kwargs) -> torch.Tensor:
    """
    Cosine schedule as proposed in Nichol & Dhariwal (Improved DDPM, 2021).
    Prevents abrupt destruction of information at early timesteps.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, max_beta)


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, **kwargs) -> torch.Tensor:
    """
    Sigmoid beta schedule providing smoother transitions at extremes.
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_schedule(schedule_name: str, timesteps: int, **kwargs) -> torch.Tensor:
    """
    Factory function for noise variance schedules with safe argument filtering.
    """
    schedule_name = schedule_name.lower()
    if schedule_name == "linear":
        beta_start = kwargs.get("beta_start", 1e-4)
        beta_end = kwargs.get("beta_end", 0.02)
        return linear_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    elif schedule_name == "cosine":
        s = kwargs.get("s", 0.008)
        max_beta = kwargs.get("max_beta", 0.999)
        return cosine_beta_schedule(timesteps, s=s, max_beta=max_beta)
    elif schedule_name == "sigmoid":
        beta_start = kwargs.get("beta_start", 1e-4)
        beta_end = kwargs.get("beta_end", 0.02)
        return sigmoid_beta_schedule(timesteps, beta_start=beta_start, beta_end=beta_end)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}. Choose from 'linear', 'cosine', 'sigmoid'.")


def precompute_diffusion_constants(betas: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Precompute all diffusion variables used in forward and reverse processes:
    alphas = 1 - betas
    alphas_cumprod = prod(alphas)
    sqrt_alphas_cumprod = sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_mean_coef1, coef2, etc.
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=betas.dtype, device=betas.device), alphas_cumprod[:-1]])
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)
    
    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # Clipped log variance to avoid log(0) at t=0
    posterior_log_variance_clipped = torch.log(
        torch.cat([posterior_variance[1:2], posterior_variance[1:]])
    )
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
