"""
Denoising Diffusion Implicit Models (DDIM) Sampler (Song et al., 2020).
Enables accelerated sampling along arbitrary sub-sequences of timesteps
(e.g., S = 10, 20, 50, 100 steps instead of 1000) using a non-Markovian forward process.
"""

import time
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from src.diffusion.ddpm import extract_into_tensor, GaussianDiffusion
from src.samplers.base import BaseSampler


class DDIMSampler(BaseSampler):
    """
    DDIM Inversion & Sampling pipeline.
    Interpolates between deterministic ODE sampling (eta = 0.0)
    and stochastic sampling (eta = 1.0).
    """
    def __init__(self, diffusion: GaussianDiffusion):
        super().__init__(diffusion)

    def get_time_schedule(
        self,
        num_inference_steps: int,
        schedule_type: str = "uniform"
    ) -> List[int]:
        """
        Generate sub-sequence of timesteps tau_1, ..., tau_S in [0, T-1].
        """
        total_T = self.diffusion.timesteps
        if num_inference_steps >= total_T:
            return list(range(total_T))

        if schedule_type == "uniform":
            # e.g., for 50 steps from 1000: 0, 20, 40, ..., 980
            c = total_T // num_inference_steps
            time_steps = [i * c for i in range(num_inference_steps)]
        elif schedule_type == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(total_T * 0.8), num_inference_steps) ** 2).astype(int)
            time_steps = list(np.clip(time_steps, 0, total_T - 1))
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Ensure unique and sorted
        time_steps = sorted(list(set(time_steps)))
        return time_steps

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single DDIM transition from timestep t to t_prev.
        Formula:
        x_{t_prev} = sqrt(alpha_prev) * pred_x_0 +
                     sqrt(1 - alpha_prev - sigma^2) * eps_theta +
                     sigma * noise
        where:
        sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_prev)
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # 1. Predict noise using model
        pred_noise = self.evaluate_model(x_t, t_tensor)

        # 2. Extract alpha_cumprod for t and t_prev
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # 3. Predict x_0
        pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        if clip_denoised:
            clip_bound = 1.0 if len(x_t.shape) > 2 else 4.0
            pred_x_0 = torch.clamp(pred_x_0, -clip_bound, clip_bound)

        # 4. Compute variance parameter sigma
        if eta > 0.0 and t_prev >= 0:
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t))
                * torch.sqrt(1.0 - alpha_t / alpha_prev)
            )
        else:
            sigma = torch.tensor(0.0, device=device)

        # 5. Direction pointing to x_{t_prev}
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma**2, min=0.0)) * pred_noise

        # 6. Random perturbation if eta > 0
        noise = torch.randn_like(x_t) if sigma > 0.0 else torch.zeros_like(x_t)

        # 7. Next sample
        x_prev = torch.sqrt(alpha_prev) * pred_x_0 + dir_xt + sigma * noise
        return x_prev, pred_x_0

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        schedule_type: str = "uniform",
        clip_denoised: bool = True,
        return_intermediates: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        DDIM sampling trajectory over sub-sequence timesteps.
        """
        self.reset_metrics()
        device = self.device
        x = torch.randn(shape, device=device)

        timesteps = self.get_time_schedule(num_inference_steps, schedule_type=schedule_type)
        time_pairs = list(zip(timesteps[1:], timesteps[:-1]))
        time_pairs.reverse() # High t to low t
        # Also include transition from timesteps[0] to -1 (pure x_0)
        time_pairs.append((timesteps[0], -1))

        intermediates = [x.cpu()] if return_intermediates else []
        pred_x_0_list = []
        step_times = []
        path_length = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        iterator = tqdm(time_pairs, desc=f"DDIM Sampling ({len(time_pairs)} steps)") if verbose else time_pairs

        for t, t_prev in iterator:
            step_start = time.perf_counter()
            prev_x = x
            x, pred_x_0 = self.ddim_step(
                x_t=x,
                t=t,
                t_prev=t_prev,
                eta=eta,
                clip_denoised=clip_denoised
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

            path_length += torch.norm((x - prev_x).reshape(x.shape[0], -1), dim=-1).mean().item()

            if return_intermediates:
                intermediates.append(x.cpu())
                pred_x_0_list.append(pred_x_0.cpu())

        total_time = time.perf_counter() - start_time

        info = {
            "sampler": "DDIM",
            "num_inference_steps": len(time_pairs),
            "eta": eta,
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

