"""Transition-likelihood resampling as an exploratory diffusion modifier.

This is deliberately named a transition-likelihood heuristic rather than a
complete SMC method: it scores the realized reverse transition under the
sampler's Gaussian kernel, but does not form a target/proposal importance ratio.
"""

import math

import torch

from diffusion.acceleration.base import (
    SamplingModifier,
    SamplingState,
    coordinate_noise_std,
)
from diffusion.acceleration.reverse_transition import reverse_transition_parameters


class SMCTransitionLikelihoodAcceleration(SamplingModifier):
    """Copy/replace particles using likelihood under their actual reverse step.

    The method needs the pre-step parent ``x_t`` as well as the observed
    destination ``x_s``. The pipeline supplies the parent in
    ``SamplingState.source_samples``. DDIM uses its configured eta; DDPM uses
    its configured posterior variance. Deterministic DDIM has no transition
    density and is rejected.
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
        max_event_probability: float | None = 0.25,
        add_noise: float = 0.0,
    ):
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if total_time <= 0 or temperature <= 0:
            raise ValueError("total_time and temperature must be positive")
        if max_delta_tau <= 0:
            raise ValueError("max_delta_tau must be positive")
        if add_noise < 0:
            raise ValueError("add_noise must be non-negative")

        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)
        self.temperature = float(temperature)
        self.divergence_penalty = float(divergence_penalty)
        self.min_timestep = int(min_timestep) if min_timestep is not None else None
        self.max_timestep = int(max_timestep) if max_timestep is not None else None
        self.max_delta_tau = float(max_delta_tau)
        self.max_event_probability = max_event_probability
        self.add_noise = float(add_noise)

    def _transition_log_likelihood(self, source, destination, state, model, scheduler):
        mean, variance = reverse_transition_parameters(
            source,
            state.timestep,
            state.previous_timestep,
            model,
            scheduler,
            state.metadata,
        )
        if variance.item() <= 0:
            raise ValueError("the selected reverse transition has zero variance")
        residual = (destination - mean).reshape(destination.shape[0], -1)
        dimensions = residual.shape[1]
        log_prob = -0.5 * residual.square().sum(dim=1) / variance
        log_prob -= 0.5 * dimensions * torch.log(2 * math.pi * variance)
        if self.divergence_penalty > 0:
            flat = destination.reshape(destination.shape[0], -1)
            spread = (flat - flat.mean(dim=0, keepdim=True)).square().sum(dim=1)
            log_prob = log_prob + self.divergence_penalty * spread
        return log_prob / self.temperature

    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
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
        source_timestep = state.timestep
        if self.min_timestep is not None and source_timestep < self.min_timestep:
            return state
        if self.max_timestep is not None and source_timestep > self.max_timestep:
            return state

        source = state.source_samples
        if source is None:
            raise ValueError(
                "transition-likelihood acceleration requires source_samples; "
                "use SamplingPipeline or provide the pre-step state"
            )
        samples = state.samples
        if source.shape != samples.shape:
            raise ValueError("source and destination populations must have matching shapes")
        population = samples.shape[0]
        if population < 2:
            return state

        log_prob = self._transition_log_likelihood(
            source, samples, state, model, scheduler
        )
        # High transition likelihood is treated as a birth; low likelihood as
        # death. This is a heuristic resampling score, not SMC importance weight.
        beta = -(log_prob - log_prob.mean())
        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = max(1, eligible_updates // self.every_k_steps)
        delta_tau = min(self.total_time / intervention_count, self.max_delta_tau)
        event_probability = -torch.expm1(-beta.abs() * delta_tau)
        if self.max_event_probability is not None and self.max_event_probability > 0:
            event_probability = event_probability.clamp(max=float(self.max_event_probability))

        selected = torch.rand(population, device=samples.device, generator=generator) < event_probability
        partners = torch.randint(population - 1, (population,), device=samples.device, generator=generator)
        event_data = torch.stack(
            (selected.float(), beta.gt(0).float(), partners.float()), dim=1
        ).cpu().tolist()
        source_indices = list(range(population))
        cloned_positions = []
        births = deaths = 0
        for index, (is_selected, is_overrepresented, raw_partner) in enumerate(event_data):
            if not is_selected:
                continue
            partner = int(raw_partner)
            if partner >= index:
                partner += 1
            if is_overrepresented:
                source_indices[index] = source_indices[partner]
                cloned_positions.append(index)
                deaths += 1
            else:
                source_indices[partner] = source_indices[index]
                cloned_positions.append(partner)
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )
        if self.add_noise > 0 and cloned_positions:
            positions = sorted(set(cloned_positions))
            noise = torch.randn(
                (len(positions), *samples.shape[1:]), device=samples.device,
                dtype=samples.dtype, generator=generator,
            )
            scale = self.add_noise * coordinate_noise_std(
                samples, context.state_timestep, scheduler, context.state_coordinates
            )
            updated = updated.clone()
            updated[torch.tensor(positions, device=samples.device)] += scale * noise

        metadata = dict(state.metadata)
        metadata.update({
            "smc_events": births + deaths,
            "smc_births": births,
            "smc_deaths": deaths,
            "smc_mean_log_prob": float(log_prob.mean().item()),
            "acceleration_mode": "smc_reverse_transition_heuristic",
        })
        return SamplingState(
            updated, state.timestep, state.previous_timestep, metadata,
            context=state.context, source_samples=state.source_samples,
        )
