"""Sequential Monte Carlo (SMC) Reverse Transition Density Acceleration.

Replaces pairwise Euclidean distance KDE with the diffusion model's own analytical
reverse transition density:
    p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

Key Advantages over Empirical KDE:
1. Zero Empirical Distance Matrices: Eliminates O(N^2) pairwise distance computations.
2. No Curse-of-Dimensionality Self-Domination: Each particle is evaluated against its
   own predicted reverse trajectory, not against distant high-dimensional neighbors.
3. O(N) Computational Complexity: Scales linearly with particle population size.
4. Natural Likelihood Weighting: Particles that transition into high-probability regions
   are retained/duplicated, while particles drifting into unlikely drift regions are culled.
"""

import math
from typing import Any

import torch
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.sampling import SamplingContext


class SMCTransitionLikelihoodAcceleration(SamplingModifier):
    """Accelerate diffusion sampling via SMC analytical reverse transition density weighting.

    Instead of computing pairwise kernel densities in pixel or latent space, this
    modifier uses the model's analytical reverse transition probability:
        log p_theta(x_{t-1} | x_t)

    Particles with high reverse transition likelihood have negative discrepancy rate
    (high birth rate, duplicated), while particles deviating into improbable drift
    regions have positive discrepancy rate (high death rate, culled).
    """

    def __init__(
        self,
        *,
        every_k_steps: int = 1,
        total_time: float = 1.0,
        temperature: float = 1.0,
        divergence_penalty: float = 0.0,
        min_timestep: int | None = 100,
        max_timestep: int | None = None,
        max_delta_tau: float = 0.005,
        max_event_probability: float = 0.25,
    ):
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if total_time <= 0:
            raise ValueError("total_time must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)
        self.temperature = float(temperature)
        self.divergence_penalty = float(divergence_penalty)
        self.min_timestep = int(min_timestep) if min_timestep is not None else None
        self.max_timestep = int(max_timestep) if max_timestep is not None else None
        self.max_delta_tau = float(max_delta_tau)
        self.max_event_probability = (
            float(max_event_probability) if max_event_probability is not None else None
        )
        self._previous_samples: torch.Tensor | None = None
        self._previous_timestep: int | None = None

    def _compute_transition_log_likelihood(
        self,
        x_prev: torch.Tensor,
        x_curr: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        model: nn.Module,
        scheduler,
    ) -> torch.Tensor:
        """Compute analytical log p_theta(x_{t-1} | x_t) for each particle."""
        ts = torch.full(
            (x_curr.shape[0],), timestep, device=x_curr.device, dtype=torch.long
        )
        alpha_bar = scheduler.alpha_bars[timestep].to(x_curr)
        alpha_bar_prev = (
            scheduler.alpha_bars[prev_timestep].to(x_curr)
            if prev_timestep >= 0
            else alpha_bar.new_tensor(1.0)
        )

        # Model forward pass for predicted noise
        with torch.no_grad():
            predicted_noise = model(x_curr, ts)

        # Predicted clean sample (Tweedie reconstructor x_0)
        predicted_clean = (
            x_curr - (1 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt()
        if getattr(scheduler, "clip_sample", False):
            limit = getattr(scheduler, "clip_sample_range", 1.0)
            predicted_clean = predicted_clean.clamp(-limit, limit)

        # Expected reverse transition mean mu_theta(x_t, t)
        # Using standard DDIM/DDPM transition parametrization
        sigma = torch.sqrt(
            (
                (1 - alpha_bar_prev)
                / (1 - alpha_bar)
                * (1 - alpha_bar / alpha_bar_prev)
            ).clamp(min=1e-20)
        )
        direction = (1 - alpha_bar_prev - sigma.square()).clamp(min=0).sqrt() * predicted_noise
        mu = alpha_bar_prev.sqrt() * predicted_clean + direction

        # Analytical Gaussian log-likelihood:
        # log p(x_{t-1} | x_t) = -0.5 * ||x_{t-1} - mu||^2 / sigma^2 - (D/2) * log(2*pi*sigma^2)
        diff = x_prev - mu
        diff_flat = diff.reshape(diff.shape[0], -1)
        squared_error = diff_flat.square().sum(dim=1)  # per-particle residual norm

        # Relative transition log-likelihood (constants cancel under mean centering)
        var = sigma.square().clamp_min(1e-12)
        log_prob = -0.5 * squared_error / var

        # Optional dispersion / divergence penalty to prevent mode collapse
        if self.divergence_penalty > 0:
            mean_sample = x_prev.mean(dim=0, keepdim=True)
            dist_to_mean = (x_prev - mean_sample).reshape(x_prev.shape[0], -1).square().sum(dim=1)
            log_prob = log_prob + self.divergence_penalty * dist_to_mean

        return log_prob / self.temperature

    def apply(
        self, state: SamplingState, *, model, scheduler, generator
    ) -> SamplingState:
        samples = state.samples
        context = state.context
        if context is None:
            return state

        # Track trajectory positions
        if (
            context.is_clean_sample
            or context.state_timestep is None
            or context.state_coordinates == "clean_x0"
        ):
            self._previous_samples = None
            return state

        # If we haven't stored the previous step's particles, store and return
        if self._previous_samples is None or self._previous_samples.shape != samples.shape:
            self._previous_samples = samples.detach().clone()
            self._previous_timestep = state.timestep
            return state

        # Intervene only every k steps
        if not 0 <= context.step_index < context.num_steps:
            return state
        if (context.step_index + 1) % self.every_k_steps != 0:
            self._previous_samples = samples.detach().clone()
            self._previous_timestep = state.timestep
            return state

        # Check timestep window restrictions
        curr_t = (
            self._previous_timestep
            if self._previous_timestep is not None
            else state.timestep
        )
        if curr_t is not None:
            if self.min_timestep is not None and curr_t < self.min_timestep:
                self._previous_samples = samples.detach().clone()
                self._previous_timestep = state.timestep
                return state
            if self.max_timestep is not None and curr_t > self.max_timestep:
                self._previous_samples = samples.detach().clone()
                self._previous_timestep = state.timestep
                return state

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = max(1, eligible_updates // self.every_k_steps)
        raw_delta_tau = self.total_time / intervention_count
        delta_tau = min(raw_delta_tau, self.max_delta_tau)

        population = samples.shape[0]
        if population < 2:
            return state

        # Compute analytical reverse transition log-density p_theta(x_{t-1} | x_t)
        timestep = self._previous_timestep if self._previous_timestep is not None else state.timestep
        prev_timestep = state.previous_timestep
        log_prob = self._compute_transition_log_likelihood(
            samples, self._previous_samples, timestep, prev_timestep, model, scheduler
        )

        # Discrepancy rate beta:
        # Particles with low transition likelihood receive positive beta (death)
        # Particles with high transition likelihood receive negative beta (birth)
        centered_log_prob = log_prob - log_prob.mean()
        beta = -centered_log_prob  # low prob -> beta > 0 (kill)

        event_probability = -torch.expm1(-beta.abs() * delta_tau)
        if self.max_event_probability is not None and self.max_event_probability > 0:
            event_probability = event_probability.clamp(max=self.max_event_probability)
        uniforms = torch.rand(population, device=samples.device, generator=generator)
        selected = uniforms < event_probability
        partners = torch.randint(
            population - 1, (population,), device=samples.device, generator=generator
        )

        event_data = torch.stack(
            (
                selected.to(torch.float32),
                beta.gt(0).to(torch.float32),
                partners.to(torch.float32),
            ),
            dim=1,
        ).cpu().tolist()

        source_indices = list(range(population))
        births = 0
        deaths = 0
        for index, (is_selected, is_overrepresented, raw_partner) in enumerate(
            event_data
        ):
            if not is_selected:
                continue
            partner = int(raw_partner)
            if partner >= index:
                partner += 1
            if is_overrepresented:
                # Particle has improbable transition; killed & replaced by partner
                source_indices[index] = source_indices[partner]
                deaths += 1
            else:
                # Particle has high-likelihood transition; duplicated & kills partner
                source_indices[partner] = source_indices[index]
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )
        self._previous_samples = updated.detach().clone()
        self._previous_timestep = state.timestep

        metadata = dict(state.metadata)
        metadata.update(
            {
                "smc_events": births + deaths,
                "smc_births": births,
                "smc_deaths": deaths,
                "smc_mean_log_prob": float(log_prob.mean().item()),
                "acceleration_mode": "smc_reverse_transition",
            }
        )
        return SamplingState(
            updated,
            state.timestep,
            state.previous_timestep,
            metadata,
            context=state.context,
        )
