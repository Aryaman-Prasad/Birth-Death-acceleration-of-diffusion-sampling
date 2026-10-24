"""Birth-death acceleration for time-varying reverse diffusion sampling (DDPM & DDIM).

Drop-in location: src/diffusion/acceleration/birth_death_ddpm.py

This module implements Deliverable 3: integrating birth-death particle dynamics
directly into the time-varying reverse diffusion sampling trajectory (DDPM / DDIM).

Theoretical formulation:
  At each reverse diffusion step (t_k -> t_{k-1}), particles transition toward
  clean data. When taking few steps (e.g., 10 or 20 DDIM steps), particles can
  cluster into redundant local modes (mode collapse) or drift into off-manifold
  artifacts.

  Between reverse diffusion steps, BirthDeathDDPM computes a relative fitness /
  density rate for each particle:
      beta_i = log rho_t(x_i) - log pi_t(x_i)
      beta_bar_i = beta_i - (1 / N) * sum_j beta_j

  * Over-represented or degraded particles (beta_bar_i > 0) are killed.
  * Under-represented or promising particles (beta_bar_i < 0) duplicate (give birth).
  * Duplicated particles receive a Brownian jitter perturbation scaled to the
    current noise level sigma_{t_{k-1}}, ensuring distinct trajectories and
    restoring population diversity without mode collapse.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.schedulers.base import NoiseScheduler

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class BirthDeathDDPMConfig:
    """Hyperparameters for birth-death diffusion sampling acceleration."""

    strength: float = 1.0               # event rate multiplier (effective Delta t scaling)
    mode: str = "pairwise"              # "pairwise" (Algorithm 1) or "resample" (systematic SMC)
    jitter: float = 0.15                # fraction of sigma_t noise added to branched particles
    bandwidth: Union[float, str] = "silverman"  # "silverman" or explicit float
    bandwidth_scale: float = 1.0        # multiplier for kernel bandwidth
    start_frac: float = 0.0             # trajectory fraction to begin BD (0.0 = start at t=T)
    end_frac: float = 0.85              # trajectory fraction to stop BD (0.85 = stop near clean data)
    every: int = 1                      # apply BD every k-th diffusion step
    max_prob: float = 0.5               # cap on per-particle event probability per step
    beta_clip: float = 15.0             # numerical clamp on rate magnitude |beta|
    pool_size: int = 4                  # spatial pooling resolution for images (4x4 = 16 features/channel)
    seed: Optional[int] = None          # optional seed for reproducible BD event sampling

    def __post_init__(self) -> None:
        if self.mode not in ("pairwise", "resample"):
            raise ValueError(f"mode must be 'pairwise' or 'resample', got {self.mode!r}")
        if self.every < 1:
            raise ValueError("every must be >= 1")
        if not (0.0 <= self.start_frac <= self.end_frac <= 1.0):
            raise ValueError("need 0 <= start_frac <= end_frac <= 1")
        if isinstance(self.bandwidth, str) and self.bandwidth != "silverman":
            raise ValueError("bandwidth must be a float or 'silverman'")


# --------------------------------------------------------------------------- #
# Density Estimation Helpers
# --------------------------------------------------------------------------- #
def silverman_bandwidth(x_flat: Tensor) -> float:
    """Silverman's rule of thumb for isotropic Gaussian kernel bandwidth."""
    n, d = x_flat.shape
    if n < 2:
        return 0.3
    sigma = x_flat.float().std(dim=0, unbiased=False).mean().clamp_min(1e-5)
    factor = (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))
    return float((sigma * factor).clamp_min(1e-4).item())


def kde_log_density(x_flat: Tensor, bandwidth: float) -> Tensor:
    """Pairwise Gaussian KDE log-density (self-kernel included, normalizing constant dropped)."""
    flat = x_flat.float()
    n = flat.shape[0]
    inv_2h2 = 1.0 / (2.0 * max(bandwidth, 1e-4) ** 2)
    # Pairwise squared Euclidean distances
    dist_sq = torch.cdist(flat, flat).square()
    # Log-sum-exp over particles minus log population
    return torch.logsumexp(-dist_sq * inv_2h2, dim=1) - math.log(max(n, 1))


def project_features(samples: Tensor, pool_size: int = 4) -> Tensor:
    """Project samples into a compact feature representation for fast, stable KDE."""
    if samples.ndim == 4:
        # Spatial average pooling: (N, C, H, W) -> (N, C * pool_size * pool_size)
        # For CIFAR-10 (3, 32, 32) with pool_size=4: 3 * 16 = 48 dimensions.
        pooled = F.adaptive_avg_pool2d(samples, (pool_size, pool_size))
        return pooled.flatten(1)
    # Vectors / 1D / 2D data: flatten directly
    return samples.reshape(samples.shape[0], -1)


# --------------------------------------------------------------------------- #
# Birth-Death Event Kernels
# --------------------------------------------------------------------------- #
def _pairwise_birth_death(
    beta_bar: Tensor, prob: Tensor, generator: Optional[torch.Generator]
) -> Tuple[Tensor, int, int]:
    """Pairwise birth-death matching.

    Returns:
        (new_indices, num_births, num_deaths)
    """
    n = beta_bar.numel()
    device = beta_bar.device
    src = torch.arange(n, device=device)
    if n < 2:
        return src, 0, 0

    fire = torch.rand(n, generator=generator, device=device) < prob
    idx = fire.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return src, 0, 0

    # Draw uniformly random partner j != i
    offsets = torch.randint(1, n, (idx.numel(),), generator=generator, device=device)
    partner = (idx + offsets) % n

    kill = beta_bar[idx] > 0
    kill_idx = idx[kill]
    kill_partner = partner[kill]

    birth_idx = idx[~kill]
    birth_partner = partner[~kill]

    # Over-represented particle is killed; slot takes partner's state
    src[kill_idx] = kill_partner
    # Under-represented particle duplicates; partner's slot takes particle's state
    src[birth_partner] = birth_idx

    num_deaths = int(kill.sum().item())
    num_births = int((~kill).sum().item())
    return src, num_births, num_deaths


def _systematic_resample(log_w: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    """Low-variance systematic resampling from unnormalized log weights."""
    n = log_w.numel()
    dev = log_w.device
    weights = torch.softmax(log_w.double(), dim=0)
    cdf = torch.cumsum(weights, dim=0)
    cdf[-1] = 1.0
    u0 = torch.rand(1, generator=generator, device=dev, dtype=torch.float64)
    u = (u0 + torch.arange(n, device=dev, dtype=torch.float64)) / n
    return torch.searchsorted(cdf, u).clamp_(max=n - 1)


# --------------------------------------------------------------------------- #
# Main Modifier: BirthDeathDDPM
# --------------------------------------------------------------------------- #
class BirthDeathDDPM(SamplingModifier):
    """Birth-death particle modifier for reverse diffusion pipelines (DDIM / DDPM)."""

    def __init__(
        self,
        config: Optional[BirthDeathDDPMConfig] = None,
        target_evaluator: Optional[Callable[[Tensor, float], Tensor]] = None,
        **overrides,
    ) -> None:
        self.cfg = config if config is not None else BirthDeathDDPMConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise TypeError(f"Unknown BirthDeathDDPMConfig field: {k}")
            setattr(self.cfg, k, v)
        self.cfg.__post_init__()
        self.target_evaluator = target_evaluator
        self._generator: Optional[torch.Generator] = None
        self.history: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Reset internal history and generator for a new sampling trajectory."""
        self.history.clear()
        self._generator = None

    def _get_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.cfg.seed is None:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(int(self.cfg.seed))
        return self._generator

    def apply(
        self,
        state: SamplingState,
        *,
        model: Any = None,
        scheduler: Optional[NoiseScheduler] = None,
        generator: Optional[torch.Generator] = None,
    ) -> SamplingState:
        """Apply one birth-death population update between reverse diffusion steps."""
        samples = state.samples
        population = samples.shape[0]
        if population < 2:
            return state

        # Step information
        step_idx = state.metadata.get("step_idx", 0)
        num_steps = state.metadata.get(
            "num_steps",
            scheduler.num_train_timesteps if scheduler is not None else 1000
        )
        total_steps = max(num_steps, 1)

        # Check start_frac and end_frac
        frac = step_idx / max(total_steps - 1, 1)
        if not (self.cfg.start_frac <= frac <= self.cfg.end_frac):
            return state
        if (step_idx % self.cfg.every) != 0:
            return state

        # Current and previous timesteps
        t_current = state.timestep
        t_prev = state.previous_timestep if state.previous_timestep >= 0 else 0
        gen = self._get_generator(samples.device) or generator

        # Compute noise scale sigma_t at target timestep for jitter
        sigma_t = 0.0
        if scheduler is not None:
            alpha_bars = getattr(scheduler, "alpha_bars", None)
            if alpha_bars is None:
                alpha_bars = getattr(scheduler, "alphas_cumprod", None)
            if alpha_bars is not None and len(alpha_bars) > 0:
                idx = min(max(int(t_prev), 0), len(alpha_bars) - 1)
                alpha_bar = float(alpha_bars[idx].detach().cpu().item())
                sigma_t = math.sqrt(max(0.0, 1.0 - alpha_bar))

        # Effective step size Delta t for the BD event clock
        denom = float(getattr(scheduler, "num_train_timesteps", 1000))
        delta_tau = abs(float(t_current - t_prev)) / denom
        delta_t = self.cfg.strength * max(delta_tau, 0.02)

        # 1. Compute empirical population log-density log rho(x_i)
        features = project_features(samples, pool_size=self.cfg.pool_size)
        if isinstance(self.cfg.bandwidth, str):
            bw = silverman_bandwidth(features) * self.cfg.bandwidth_scale
        else:
            bw = float(self.cfg.bandwidth)
        log_rho = kde_log_density(features, bw)

        # 2. Compute target log-density log pi(x_i)
        if self.target_evaluator is not None:
            log_pi = self.target_evaluator(samples, sigma_t).to(samples.device)
        else:
            # Mode-diversity prior: uniform across feature space (log_pi = 0)
            log_pi = torch.zeros_like(log_rho)

        # 3. Calculate centered discrepancy rate beta_bar
        beta = log_rho - log_pi
        beta = torch.nan_to_num(beta, nan=0.0).clamp(-self.cfg.beta_clip, self.cfg.beta_clip)
        beta_bar = beta - beta.mean()

        # 4. Sample birth-death events
        if self.cfg.mode == "pairwise":
            prob = (-torch.expm1(-beta_bar.abs() * delta_t)).clamp(max=self.cfg.max_prob)
            src, births, deaths = _pairwise_birth_death(beta_bar, prob, gen)
            ess = float("nan")
        else:
            log_w = -beta_bar.double() * self.cfg.strength
            src = _systematic_resample(log_w, gen)
            births = deaths = population
            w = torch.softmax(log_w, dim=0)
            ess = float(1.0 / (w ** 2).sum().item())

        # 5. Apply ancestry re-indexing to population
        updated_samples = samples[src].clone()

        # 6. Apply diversity jitter to newly branched particles
        moved = src != torch.arange(population, device=samples.device)
        num_moved = int(moved.sum().item())
        if self.cfg.jitter > 0.0 and sigma_t > 0.0 and num_moved > 0:
            noise_shape = updated_samples[moved].shape
            noise = torch.randn(
                noise_shape, device=samples.device, dtype=updated_samples.dtype, generator=gen
            )
            updated_samples[moved] = updated_samples[moved] + (self.cfg.jitter * sigma_t) * noise

        # Update metadata
        event_info = {
            "step_idx": step_idx,
            "timestep": t_current,
            "previous_timestep": t_prev,
            "sigma_t": sigma_t,
            "delta_t": delta_t,
            "bandwidth": bw,
            "births": births,
            "deaths": deaths,
            "moved": num_moved,
            "beta_std": float(beta_bar.std().item()),
            "ess": ess,
        }
        self.history.append(event_info)

        metadata = dict(state.metadata)
        metadata["birth_death_ddpm"] = event_info
        return SamplingState(updated_samples, state.timestep, state.previous_timestep, metadata)


# --------------------------------------------------------------------------- #
# CLI Argument Registration Helpers
# --------------------------------------------------------------------------- #
def add_birth_death_ddpm_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI flags for reverse diffusion birth-death acceleration."""
    group = parser.add_argument_group("Birth-Death Diffusion Sampling (DDPM/DDIM) Arguments")
    group.add_argument(
        "--birth-death-ddpm", "--bd-ddpm", action="store_true", dest="birth_death_ddpm",
        help="enable birth-death acceleration during DDPM / DDIM reverse diffusion sampling"
    )
    group.add_argument(
        "--bd-ddpm-strength", type=float, default=1.0,
        help="effective strength multiplier for birth-death event probabilities (default: 1.0)"
    )
    group.add_argument(
        "--bd-ddpm-mode", choices=["pairwise", "resample"], default="pairwise",
        help="birth-death kernel mode: pairwise (Algorithm 1) or resample (SMC-style)"
    )
    group.add_argument(
        "--bd-ddpm-jitter", type=float, default=0.15,
        help="Brownian noise jitter added to branched particles (default: 0.15)"
    )
    group.add_argument(
        "--bd-ddpm-bandwidth", default="silverman",
        help="feature KDE bandwidth: 'silverman' or explicit float (default: silverman)"
    )
    group.add_argument(
        "--bd-ddpm-start-frac", type=float, default=0.0,
        help="trajectory progress fraction to begin birth-death (default: 0.0)"
    )
    group.add_argument(
        "--bd-ddpm-end-frac", type=float, default=0.85,
        help="trajectory progress fraction to stop birth-death (default: 0.85)"
    )
    group.add_argument(
        "--bd-ddpm-every", type=int, default=1,
        help="apply birth-death every k-th diffusion step (default: 1)"
    )


def create_birth_death_ddpm_from_args(
    args: argparse.Namespace,
    target_evaluator: Optional[Callable[[Tensor, float], Tensor]] = None,
) -> Optional[BirthDeathDDPM]:
    """Instantiate a BirthDeathDDPM modifier if enabled via CLI flags."""
    if not getattr(args, "birth_death_ddpm", False):
        return None

    bw = getattr(args, "bd_ddpm_bandwidth", "silverman")
    try:
        bw = float(bw)
    except (ValueError, TypeError):
        pass

    config = BirthDeathDDPMConfig(
        strength=getattr(args, "bd_ddpm_strength", 1.0),
        mode=getattr(args, "bd_ddpm_mode", "pairwise"),
        jitter=getattr(args, "bd_ddpm_jitter", 0.15),
        bandwidth=bw,
        start_frac=getattr(args, "bd_ddpm_start_frac", 0.0),
        end_frac=getattr(args, "bd_ddpm_end_frac", 0.85),
        every=getattr(args, "bd_ddpm_every", 1),
    )
    return BirthDeathDDPM(config=config, target_evaluator=target_evaluator)

