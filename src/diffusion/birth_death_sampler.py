"""
Birth-Death Process Based Accelerated Diffusion Sampler.
Implements the methodology derived from:
1. Lu, Lu, & Nolen (2019): 'Accelerating Langevin Sampling with Birth-Death'
2. Song et al. (2020): 'Denoising Diffusion Implicit Models' (DDIM)
3. Song et al. (2020): 'Score-Based Generative Modeling through SDEs' (Predictor-Corrector)

Mathematical Formulation:
--------------------------
Standard diffusion sampling solves the reverse-time SDE:
    dx_t = [ f(x_t, t) - g(t)^2 grad_{x_t} log p_t(x_t) ] dt + g(t) dw_t

When step budget S << T (e.g., S in [10, 20, 50]), discretization errors accumulate,
leading to off-manifold particles and mode collapse.

The Birth-Death Langevin / Diffusion Sampler (BDLS-Diffusion) augments the drift-diffusion
dynamics with an ensemble-level nonlocal Birth-Death jump process:
    d rho_t / dt = L*_{Langevin} rho_t + alpha * rho_t * ( log(pi_t / rho_t) - E_{rho_t}[log(pi_t / rho_t)] )

- DEATH (Pruning): Particles located in low-likelihood valleys, high drift error,
  or over-congested redundant regions are extinguished.
- BIRTH (Branching): Particles discovering promising high-density target modes
  replicate with exploratory perturbation, rapidly shifting probability mass
  across energy barriers without requiring 1000 slow diffusion steps.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from src.diffusion.ddpm import extract_into_tensor, GaussianDiffusion
from src.samplers.base import BaseSampler


def estimate_kernel_density_log(x: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
    """
    Computes log empirical particle density log rho_t(x_i) using a Gaussian Kernel:
        rho(x_i) = (1 / N) * sum_j exp( - ||x_i - x_j||^2 / (2 * h^2) ) / (2*pi*h^2)^(d/2)
    This provides the repulsive / entropy term in the Lu-Lu-Nolen BDLS rate.
    """
    N, D = x.shape[0], x.shape[1]
    
    # Subsample if batch is very large to compute pairwise distances efficiently
    if bandwidth is None:
        # Silverman's rule of thumb for bandwidth selection
        std = torch.std(x, dim=0).mean()
        bandwidth = (std * (4.0 / (3.0 * N)) ** (1.0 / 5.0)).clamp(min=0.1, max=1.5).item()

    # Pairwise squared distances (N, N)
    dists_sq = torch.cdist(x, x, p=2) ** 2
    gamma = 1.0 / (2.0 * (bandwidth ** 2))
    
    # Kernel log-sum-exp for numerical stability
    log_kernels = -gamma * dists_sq
    log_rho = torch.logsumexp(log_kernels, dim=1) - math.log(N) - (D / 2.0) * math.log(2 * math.pi * (bandwidth ** 2))
    return log_rho


class BirthDeathDiffusionSampler(BaseSampler):
    """
    Accelerated Diffusion Sampler powered by a Birth-Death jump process.
    Supports:
    1. Accelerated DDIM/ODE leap (Predictor)
    2. Score-based Langevin drift refinement (Corrector)
    3. Lu-Lu-Nolen Birth-Death branching and pruning (Selection)
    """
    def __init__(
        self,
        diffusion: GaussianDiffusion,
        birth_rate: float = 0.5,
        langevin_steps: int = 1,
        langevin_lr: float = 0.05,
        exploration_noise: float = 0.02,
        diversity_weight: float = 0.3,
        resample_interval: int = 1,
    ):
        super().__init__(diffusion)
        self.birth_rate = birth_rate
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        self.exploration_noise = exploration_noise
        self.diversity_weight = diversity_weight
        self.resample_interval = resample_interval

        # Statistics
        self.total_births = 0
        self.total_deaths = 0

    def get_time_schedule(self, num_inference_steps: int) -> List[int]:
        """Generate sub-sequence of timesteps for accelerated sampling."""
        total_T = self.diffusion.timesteps
        if num_inference_steps >= total_T:
            return list(range(total_T))
        c = total_T // num_inference_steps
        time_steps = [i * c for i in range(num_inference_steps)]
        return sorted(list(set(time_steps)))

    def compute_energy_and_score(
        self,
        x_t: torch.Tensor,
        t: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates:
        1. Predicted noise epsilon_theta(x_t, t)
        2. Score: grad_x log p_t(x_t) = -epsilon_theta / sqrt(1 - alpha_cumprod)
        3. Local target fitness / log-likelihood proxy:
           Based on Tweedie reconstructability and score magnitude consistency.
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        pred_noise = self.evaluate_model(x_t, t_tensor)
        alpha_t = self.diffusion.alphas_cumprod[t]
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)
        sqrt_alpha = torch.sqrt(alpha_t)

        # Score function
        score = -pred_noise / sqrt_one_minus_alpha

        # Predicted x_0 (Tweedie's identity)
        pred_x_0 = (x_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha
        clip_bound = 1.0 if len(x_t.shape) > 2 else 4.0
        pred_x_0 = torch.clamp(pred_x_0, -clip_bound, clip_bound)

        # Fitness proxy: negative consistency energy - ||x_t - sqrt(alpha)*x_0||^2 / (2*(1-alpha))
        # High fitness = particle cleanly aligns with denoising direction
        x_flat = x_t.reshape(B, -1)
        recon_flat = (sqrt_alpha * pred_x_0).reshape(B, -1)
        residual = x_flat - recon_flat
        log_pi = -0.5 * torch.sum(residual ** 2, dim=-1) / (1.0 - alpha_t + 1e-6)

        return pred_noise, score, log_pi

    def langevin_corrector_step(
        self,
        x_t: torch.Tensor,
        score: torch.Tensor,
        step_size: float
    ) -> torch.Tensor:
        """
        Langevin MCMC step along score vector:
        x <- x + (step_size / 2) * grad log p_t(x) + sqrt(step_size) * z
        """
        if step_size <= 0.0:
            return x_t
        noise = torch.randn_like(x_t)
        return x_t + 0.5 * step_size * score + math.sqrt(step_size) * noise

    def birth_death_selection_step(
        self,
        x_t: torch.Tensor,
        log_pi: torch.Tensor,
        bandwidth: Optional[float] = None
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Nonlocal Birth-Death Jump Selection (Lu-Lu-Nolen BDLS):
        1. Compute empirical density rho_t(x_i) via KDE.
        2. Compute fitness rate:
           r_i = log_pi(x_i) - lambda_div * log_rho(x_i) - mean(log_pi - lambda_div * log_rho)
        3. Particles with r_i > 0 replicate (Birth) with exploratory jitter.
        4. Particles with r_i < 0 are pruned (Death) and replaced by offspring.
        """
        N = x_t.shape[0]
        x_flat = x_t.reshape(N, -1)

        # 1. Estimate particle density log rho_t
        log_rho = estimate_kernel_density_log(x_flat, bandwidth=bandwidth)

        # 2. Compute net fitness rate r_i (relative to ensemble average)
        fitness = log_pi - self.diversity_weight * log_rho
        fitness_centered = fitness - torch.mean(fitness)

        # 3. Compute normalized selection weights W_i = exp(gamma * r_i)
        # Clamp to avoid exponential overflow
        clamped_fitness = torch.clamp(self.birth_rate * fitness_centered, min=-5.0, max=5.0)
        weights = torch.exp(clamped_fitness)
        probs = weights / torch.sum(weights)

        # 4. Systematic resampling (preserves total particle count N exactly)
        # Higher-weight particles are duplicated; lower-weight particles are eliminated
        num_copies = torch.multinomial(probs, num_samples=N, replacement=True)
        
        # Count birth and death events
        bincounts = torch.bincount(num_copies, minlength=N)
        deaths = int((bincounts == 0).sum().item())
        births = int((bincounts > 1).sum().item())

        # Select surviving and cloned particles
        x_selected = x_t[num_copies]

        # 5. Exploratory jitter for cloned particles to prevent identical point collapse
        is_clone = (bincounts[num_copies] > 1)
        while len(is_clone.shape) < len(x_t.shape):
            is_clone = is_clone[..., None]
        
        jitter = torch.randn_like(x_t) * self.exploration_noise
        x_selected = torch.where(is_clone, x_selected + jitter, x_selected)

        return x_selected, births, deaths

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        pred_noise: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Deterministic/stochastic DDIM leap step from t to t_prev.
        """
        device = x_t.device
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        pred_x_0 = (x_t - torch.sqrt(1.0 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        if clip_denoised:
            clip_bound = 1.0 if len(x_t.shape) > 2 else 4.0
            pred_x_0 = torch.clamp(pred_x_0, -clip_bound, clip_bound)

        sigma = (
            eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t)) * torch.sqrt(1.0 - alpha_t / alpha_prev)
            if (eta > 0.0 and t_prev >= 0) else torch.tensor(0.0, device=device)
        )
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma ** 2, min=0.0)) * pred_noise
        noise = torch.randn_like(x_t) if sigma > 0.0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_prev) * pred_x_0 + dir_xt + sigma * noise

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        enable_birth_death: bool = True,
        enable_langevin: bool = True,
        clip_denoised: Optional[bool] = None,
        return_intermediates: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Accelerated Birth-Death Diffusion Sampling trajectory.
        
        Args:
            shape: (Batch, Channels, H, W) or (Batch, 2)
            num_inference_steps: Reduced sampling budget S << T (e.g., 20 or 50)
            eta: DDIM stochasticity factor (0 = deterministic ODE)
            enable_birth_death: If True, activates nonlocal birth-death selection
            enable_langevin: If True, executes score-based Langevin corrector
            clip_denoised: Data-aware coordinate clamping
            return_intermediates: Record trajectory evolution
            verbose: Display progress bar
        """
        self.reset_metrics()
        self.total_births = 0
        self.total_deaths = 0

        device = self.device
        x = torch.randn(shape, device=device)

        if clip_denoised is None:
            clip_denoised = True

        timesteps = self.get_time_schedule(num_inference_steps)
        time_pairs = list(zip(timesteps[1:], timesteps[:-1]))
        time_pairs.reverse()
        time_pairs.append((timesteps[0], -1))

        intermediates = [x.cpu()] if return_intermediates else []
        step_times = []
        path_length = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        iterator = tqdm(time_pairs, desc=f"BDLS-Diffusion ({len(time_pairs)} steps)") if verbose else time_pairs

        for step_idx, (t, t_prev) in enumerate(iterator):
            step_start = time.perf_counter()
            prev_x = x

            # 1. Evaluate score, predicted noise, and target log-likelihood proxy
            pred_noise, score, log_pi = self.compute_energy_and_score(x, t)

            # 2. Langevin Corrector (Exploration along score field)
            if enable_langevin and self.langevin_steps > 0:
                for _ in range(self.langevin_steps):
                    # Scale Langevin step size adaptively with noise variance (1 - alpha_t)
                    eff_lr = self.langevin_lr * (1.0 - self.diffusion.alphas_cumprod[t]).item()
                    x = self.langevin_corrector_step(x, score, step_size=eff_lr)

            # 3. Nonlocal Birth-Death Jump Selection (Lu-Lu-Nolen Mechanism)
            # Apply during intermediate/early denoising where mode decisions occur
            if enable_birth_death and (step_idx % self.resample_interval == 0) and (t > timesteps[0]):
                x, b, d = self.birth_death_selection_step(x, log_pi)
                self.total_births += b
                self.total_deaths += d

            # 4. Predictor Step: Accelerated DDIM transition to t_prev
            x = self.ddim_step(
                x_t=x,
                t=t,
                t_prev=t_prev,
                pred_noise=pred_noise,
                eta=eta,
                clip_denoised=clip_denoised
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)

            path_length += torch.norm((x - prev_x).reshape(x.shape[0], -1), dim=-1).mean().item()

            if return_intermediates:
                intermediates.append(x.cpu())

        total_time = time.perf_counter() - start_time

        info = {
            "sampler": "BDLS-Diffusion",
            "num_inference_steps": len(time_pairs),
            "eta": eta,
            "inference_time": total_time,
            "latency_per_sample": total_time / shape[0],
            "throughput_samples_per_sec": shape[0] / total_time if total_time > 0 else 0.0,
            "nfe": self.nfe,
            "step_times": step_times,
            "path_length": path_length,
            "total_births": self.total_births,
            "total_deaths": self.total_deaths,
            "intermediates": intermediates,
        }
        return x, info

