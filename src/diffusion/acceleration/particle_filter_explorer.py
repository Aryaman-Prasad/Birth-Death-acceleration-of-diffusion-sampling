"""Exploratory branching and resampling around stochastic DDPM/DDIM steps."""

import torch

from diffusion.acceleration.base import (
    SamplingModifier,
    SamplingState,
    coordinate_noise_std,
)
from diffusion.acceleration.reverse_transition import reverse_transition_parameters


class ParticleFilteringSolutionSpaceExplorer(SamplingModifier):
    """Branch from each pre-step parent, then select one candidate per parent.

    This is an exploratory branch/filter method, not a canonical particle
    filter: the candidate potential is the reverse-transition log density and
    an optional diversity bonus, followed by a separate BD-like exchange.
    It must run with stochastic DDIM (eta > 0) or DDPM so candidate branches
    have a non-degenerate transition distribution.
    """

    def __init__(
        self,
        *,
        branch_factor: int = 2,
        every_k_steps: int = 2,
        temperature: float = 1.0,
        diversity_weight: float = 0.1,
        min_timestep: int | None = 100,
        max_timestep: int | None = None,
        total_time: float = 1.0,
        max_delta_tau: float = 0.005,
        max_event_probability: float | None = 0.25,
        add_noise: float = 0.0,
    ):
        if branch_factor < 2:
            raise ValueError("branch_factor must be at least 2")
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if temperature <= 0 or total_time <= 0 or max_delta_tau <= 0:
            raise ValueError("temperature, total_time, and max_delta_tau must be positive")
        if add_noise < 0:
            raise ValueError("add_noise must be non-negative")

        self.branch_factor = int(branch_factor)
        self.every_k_steps = int(every_k_steps)
        self.temperature = float(temperature)
        self.diversity_weight = float(diversity_weight)
        self.min_timestep = int(min_timestep) if min_timestep is not None else None
        self.max_timestep = int(max_timestep) if max_timestep is not None else None
        self.total_time = float(total_time)
        self.max_delta_tau = float(max_delta_tau)
        self.max_event_probability = max_event_probability
        self.add_noise = float(add_noise)

    def _explore_solution_branches(self, source, timestep, previous_timestep,
                                    model, scheduler, metadata, generator):
        mean, variance = reverse_transition_parameters(
            source, timestep, previous_timestep, model, scheduler, metadata
        )
        if variance.item() <= 0:
            raise ValueError("the selected reverse transition has zero variance")
        count = source.shape[0]
        candidates = []
        potentials = []
        dimensions = source[0].numel()
        for _ in range(self.branch_factor):
            noise = torch.randn(
                source.shape, device=source.device, dtype=source.dtype,
                generator=generator,
            )
            candidate = mean + variance.sqrt() * noise
            residual = (candidate - mean).reshape(count, -1)
            log_probability = (
                -0.5 * residual.square().sum(dim=1) / variance
                -0.5 * dimensions * torch.log(2 * torch.pi * variance)
            )
            candidates.append(candidate)
            potentials.append(log_probability)

        all_candidates = torch.cat(candidates, dim=0)
        all_potentials = torch.cat(potentials, dim=0)
        if self.diversity_weight > 0:
            flat = all_candidates.reshape(all_candidates.shape[0], -1)
            distance_from_mean = (flat - flat.mean(dim=0, keepdim=True)).square().sum(dim=1)
            all_potentials = all_potentials + self.diversity_weight * distance_from_mean
        return all_candidates, all_potentials

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
        if self.min_timestep is not None and state.timestep < self.min_timestep:
            return state
        if self.max_timestep is not None and state.timestep > self.max_timestep:
            return state

        source = state.source_samples
        if source is None:
            raise ValueError(
                "particle filtering requires source_samples; use SamplingPipeline "
                "or provide the pre-step state"
            )
        samples = state.samples
        count = samples.shape[0]
        if source.shape != samples.shape:
            raise ValueError("source and destination populations must have matching shapes")
        if count < 2:
            return state

        candidates, potentials = self._explore_solution_branches(
            source, state.timestep, state.previous_timestep, model, scheduler,
            state.metadata, generator,
        )
        branches = potentials.reshape(self.branch_factor, count)
        branch_weights = torch.softmax(branches / self.temperature, dim=0)
        chosen_branches = torch.multinomial(
            branch_weights.T, num_samples=1, generator=generator
        ).squeeze(-1)
        chosen_indices = chosen_branches * count + torch.arange(count, device=samples.device)
        selected_samples = candidates.index_select(0, chosen_indices)
        selected_potentials = potentials.index_select(0, chosen_indices)

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = max(1, eligible_updates // self.every_k_steps)
        delta_tau = min(self.total_time / intervention_count, self.max_delta_tau)
        births = deaths = 0
        cloned_positions = []
        if self.max_event_probability is not None and self.max_event_probability > 0:
            beta = -(selected_potentials - selected_potentials.mean())
            event_probability = -torch.expm1(-beta.abs() * delta_tau)
            event_probability = event_probability.clamp(max=float(self.max_event_probability))
            selected = torch.rand(count, device=samples.device, generator=generator) < event_probability
            partners = torch.randint(count - 1, (count,), device=samples.device, generator=generator)
            event_data = torch.stack(
                (selected.float(), beta.gt(0).float(), partners.float()), dim=1
            ).cpu().tolist()
            source_indices = list(range(count))
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
            selected_samples = selected_samples.index_select(
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
            selected_samples = selected_samples.clone()
            selected_samples[torch.tensor(positions, device=samples.device)] += scale * noise

        metadata = dict(state.metadata)
        metadata.update({
            "filter_pool_size": int(candidates.shape[0]),
            "filter_retained": count,
            "filter_mean_potential": float(potentials.mean().item()),
            "filter_bd_events": births + deaths,
            "filter_bd_births": births,
            "filter_bd_deaths": deaths,
            "acceleration_mode": "particle_filtering_exploration",
        })
        return SamplingState(
            selected_samples, state.timestep, state.previous_timestep, metadata,
            context=state.context, source_samples=state.source_samples,
        )
