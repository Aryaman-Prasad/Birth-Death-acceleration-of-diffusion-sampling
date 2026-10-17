"""
Standard DDPM Ancestral Reverse Sampler (Ho et al., 2020).
Serves as the primary ground-truth baseline for sample quality, NFE (typically 1000),
and baseline inference latency.
"""

import time
import torch
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from src.diffusion.ddpm import extract_into_tensor, GaussianDiffusion
from src.samplers.base import BaseSampler


class DDPMSampler(BaseSampler):
    """
    Standard DDPM Ancestral Sampler.
    Iteratively computes:
    x_{t-1} = mu_theta(x_t, t) + sigma_t * z
    where z ~ N(0, I) for t > 0, and z = 0 for t = 0.
    """
    def __init__(self, diffusion: GaussianDiffusion):
        super().__init__(diffusion)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, clip_denoised: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single ancestral reverse step from t to t - 1.
        """
        B = x.shape[0]
        t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)
        
        # Predict noise and x_0
        pred_noise = self.evaluate_model(x, t_tensor)
        pred_x_0 = self.diffusion.predict_x_start_from_noise(x, t_tensor, pred_noise)
        
        if clip_denoised:
            clip_bound = 1.0 if len(x.shape) > 2 else 4.0
            pred_x_0 = torch.clamp(pred_x_0, -clip_bound, clip_bound)
            
        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior_mean_variance(
            x_start=pred_x_0, x_t=x, t=t_tensor
        )
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        nonzero_mask = (t_tensor != 0).float()
        while len(nonzero_mask.shape) < len(x.shape):
            nonzero_mask = nonzero_mask[..., None]

        pred_x_prev = model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
        return pred_x_prev, pred_x_0

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        clip_denoised: bool = True,
        return_intermediates: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Full DDPM Ancestral reverse sampling process starting from pure Gaussian noise.
        """
        self.reset_metrics()
        device = self.device
        x = torch.randn(shape, device=device)
        
        intermediates = [x.cpu()] if return_intermediates else []
        pred_x_0_list = []
        step_times = []
        path_length = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        steps_range = reversed(range(self.diffusion.timesteps))
        iterator = tqdm(steps_range, desc="DDPM Sampling", total=self.diffusion.timesteps) if verbose else steps_range

        for t in iterator:
            step_start = time.perf_counter()
            prev_x = x
            x, pred_x_0 = self.p_sample(x, t, clip_denoised=clip_denoised)

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

            # Accumulate path length ||x_{t-1} - x_t||
            path_length += torch.norm((x - prev_x).reshape(x.shape[0], -1), dim=-1).mean().item()

            if return_intermediates:
                intermediates.append(x.cpu())
                pred_x_0_list.append(pred_x_0.cpu())

        total_time = time.perf_counter() - start_time

        info = {
            "sampler": "DDPM",
            "total_timesteps": self.diffusion.timesteps,
            "inference_time": total_time,
            "latency_per_sample": total_time / shape[0],
            "throughput_samples_per_sec": shape[0] / total_time if total_time > 0 else 0.0,
            "nfe": self.nfe,
            "step_times": step_times,
            "path_length": path_length,
            "intermediates": intermediates,
            "pred_x_0_trajectory": pred_x_0_list,
        }
        return x, info

