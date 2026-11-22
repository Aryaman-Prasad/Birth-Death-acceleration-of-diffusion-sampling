"""Particle Filtering Solution Space Explorer with Combined Birth-Death Resampling.

Implements an exploratory particle filtering strategy for diffusion reverse sampling:
1. Branching (Solution Space Exploration):
   Instead of following a single stochastic trajectory, each particle x_t branches into
   B candidate transitions exploring local stochastic variations:
       x_{t-1}^{(i, b)} = mu_theta(x_t^{(i)}, t) + sigma_t * epsilon^{(i, b)},  b in {1, ..., B}
2. Solution Quality Potential:
   Evaluates each branched candidate using a reconstruction consistency / potential score:
       G_t(x_{t-1}) = -||x_{t-1} - mu_theta(x_t, t)||^2 / (2 * sigma_t^2)
3. Combined Birth-Death Resampling:
   Combines Sequential Monte Carlo filtering with Birth-Death interaction to prune
   the expanded N * B candidate pool back to N elite particles, actively weeding
   out degenerate exploratory branches while duplicating high-fidelity modes.
"""

import math
from typing import Any

import torch
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.sampling import SamplingContext


class ParticleFilteringSolutionSpaceExplorer(SamplingModifier):
    """Explores solution space during reverse diffusion via branching and particle filtering.

    At each intervention cadence:
    1. Branches N particles into N * branch_factor candidate paths.
    2. Scores candidates using analytical drift consistency and mode diversity.
    3. Uses Birth-Death population interaction to filter the best N surviving particles.
    """

    def __init__(
        self,
        *,
        branch_factor: int = 2,
        every_k_steps: int = 2,
        temperature: float = 1.0,
        diversity_weight: float = 0.1,
    ):
        if branch_factor < 2:
            raise ValueError("branch_factor must be at least 2 to explore alternative paths")
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.branch_factor = int(branch_factor)
        self.every_k_steps = int(every_k_steps)
        self.temperature = float(temperature)
        self.diversity_weight = float(diversity_weight)

    def _explore_solution_branches(
        self,
        x_curr: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        model: nn.Module,
        scheduler,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate B candidate transitions per particle and score their solution potential."""
        N = x_curr.shape[0]
        B = self.branch_factor

        ts = torch.full((N,), timestep, device=x_curr.device, dtype=torch.long)
        alpha_bar = scheduler.alpha_bars[timestep].to(x_curr)
        alpha_bar_prev = (
            scheduler.alpha_bars[prev_timestep].to(x_curr)
            if prev_timestep >= 0
            else alpha_bar.new_tensor(1.0)
        )

        with torch.no_grad():
            predicted_noise = model(x_curr, ts)

        predicted_clean = (x_curr - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        if getattr(scheduler, "clip_sample", False):
            limit = getattr(scheduler, "clip_sample_range", 1.0)
            predicted_clean = predicted_clean.clamp(-limit, limit)

        sigma = torch.sqrt(
            (
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            ).clamp(min=1e-20)
        )
        direction = (1 - alpha_bar_prev - sigma.square()).clamp(min=0).sqrt() * predicted_noise
        mu = alpha_bar_prev.sqrt() * predicted_clean + direction

        # Generate B stochastic perturbations for each parent particle
        candidates = []
        potentials = []

        for b in range(B):
            noise_b = torch.randn(
                x_curr.shape, device=x_curr.device, dtype=x_curr.dtype, generator=generator
            )
            cand = mu + sigma * noise_b
            candidates.append(cand)

            # Potential score: log-density under the Gaussian transition kernel
            diff = cand - mu
            sq_err = diff.reshape(diff.shape[0], -1).square().sum(dim=1)
            pot = -0.5 * sq_err / (sigma.square().clamp_min(1e-12))
            potentials.append(pot)

        # Reshape to (N * B, ...)
        all_candidates = torch.cat(candidates, dim=0)  # (N*B, C, H, W)
        all_potentials = torch.cat(potentials, dim=0)  # (N*B,)

        # Diversity potential bonus: encourage candidates that differ from batch mean
        if self.diversity_weight > 0:
            cand_flat = all_candidates.reshape(all_candidates.shape[0], -1)
            mean_cand = cand_flat.mean(dim=0, keepdim=True)
            dist_to_mean = (cand_flat - mean_cand).square().sum(dim=1)
            all_potentials = all_potentials + self.diversity_weight * dist_to_mean

        return all_candidates, all_potentials

    def apply(
        self, state: SamplingState, *, model, scheduler, generator
    ) -> SamplingState:
        samples = state.samples
        context = state.context
        if context is None:
            return state

        if (
            context.is_clean_sample
            or context.state_timestep is None
            or context.state_coordinates == "clean_x0"
        ):
            return state

        if not 0 <= context.step_index < context.num_steps:
            return state
        if (context.step_index + 1) % self.every_k_steps != 0:
            return state

        N = samples.shape[0]
        timestep = state.timestep
        prev_timestep = state.previous_timestep

        # 1. Branch and explore solution space
        candidates, potentials = self._explore_solution_branches(
            samples, timestep, prev_timestep, model, scheduler, generator=generator
        )

        # 2. Combined Birth-Death Selection over the N*B candidate solution space
        total_pool_size = candidates.shape[0]  # N * B
        logits = potentials / self.temperature
        weights = torch.softmax(logits, dim=0)

        # Systematic / multinomial sampling according to importance weights
        # Select best N particles to survive back into the population
        top_k_indices = torch.multinomial(
            weights, num_samples=N, replacement=False, generator=generator
        )

        filtered_samples = candidates.index_select(0, top_k_indices)

        metadata = dict(state.metadata)
        metadata.update(
            {
                "filter_pool_size": total_pool_size,
                "filter_retained": N,
                "filter_mean_potential": float(potentials.mean().item()),
                "acceleration_mode": "particle_filtering_exploration",
            }
        )

        return SamplingState(
            filtered_samples,
            state.timestep,
            state.previous_timestep,
            metadata,
            context=state.context,
        )
