"""Continuous normalizing-flow densities and a BD modifier that consumes them."""

import math
from typing import Callable

import torch
from torch import nn

from diffusion.acceleration.base import (
    SamplingModifier,
    SamplingState,
    coordinate_noise_std,
)
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


def compute_divergence_exact(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Compute ``Tr(dv/dx)`` for samplewise vector fields."""
    x = x if x.requires_grad else x.detach().requires_grad_(True)
    velocity = v_fn(x, t)
    flat_velocity = velocity.reshape(velocity.shape[0], -1)
    divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if not flat_velocity.requires_grad:
        return divergence
    for index in range(flat_velocity.shape[1]):
        component = flat_velocity[:, index].sum()
        if not component.requires_grad:
            continue
        gradient = torch.autograd.grad(
            component, x,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is not None:
            divergence = divergence + gradient.reshape(x.shape[0], -1)[:, index]
    return divergence if create_graph else divergence.detach()


def compute_divergence_hutchinson(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_samples: int = 1,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Unbiased Hutchinson trace estimate ``E[e^T J_v e]``."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    x = x if x.requires_grad else x.detach().requires_grad_(True)
    velocity = v_fn(x, t)
    divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if not velocity.requires_grad:
        return divergence
    for _ in range(num_samples):
        probe = torch.randn_like(x)
        probe_flat = probe.reshape(probe.shape[0], -1)
        product = (velocity.reshape(velocity.shape[0], -1) * probe_flat).sum()
        if not product.requires_grad:
            continue
        gradient = torch.autograd.grad(
            product, x, create_graph=create_graph, retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is not None:
            divergence = divergence + (
                gradient.reshape(gradient.shape[0], -1) * probe_flat
            ).sum(dim=1)
    divergence = divergence / num_samples
    return divergence if create_graph else divergence.detach()


class ContinuousNormalizingFlowTarget(nn.Module, TargetDistribution):
    """A CNF density obtained by integrating the change-of-variables ODE.

    The velocity convention is ``dx/dt = v(x,t)``. From a queried state at
    normalized time ``t`` the state is integrated forward to the Gaussian base
    at time 1, while accumulating divergence:

    ``log p_t(x_t) = log p_1(x_1) + integral_t^1 div(v(x_s,s)) ds``.

    Exact divergence is suitable for low-dimensional toy tests. Hutchinson
    mode supports larger states with a stochastic trace estimate; it is not an
    exact density evaluation in that mode.
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        prior_std: float = 1.0,
        divergence_method: str = "exact",
        hutchinson_samples: int = 2,
        integration_steps: int = 32,
        state_coordinates: str = "x_t",
    ):
        super().__init__()
        if prior_std <= 0:
            raise ValueError("prior_std must be positive")
        if divergence_method not in {"exact", "hutchinson"}:
            raise ValueError("divergence_method must be 'exact' or 'hutchinson'")
        if hutchinson_samples < 1 or integration_steps < 1:
            raise ValueError("hutchinson_samples and integration_steps must be positive")
        if state_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError("state_coordinates must be 'x_t' or 'normalized_x_t'")
        self.velocity_net = velocity_net
        self.prior_std = float(prior_std)
        self.divergence_method = divergence_method
        self.hutchinson_samples = int(hutchinson_samples)
        self.integration_steps = int(integration_steps)
        self.state_coordinates = state_coordinates

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(x, t)

    def _field_and_divergence(self, x, t, *, create_graph):
        x = x if x.requires_grad else x.detach().requires_grad_(True)
        velocity = self.velocity(x, t)
        flat_velocity = velocity.reshape(velocity.shape[0], -1)
        if self.divergence_method == "exact":
            divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            if flat_velocity.requires_grad:
                for index in range(flat_velocity.shape[1]):
                    component = flat_velocity[:, index].sum()
                    if not component.requires_grad:
                        continue
                    gradient = torch.autograd.grad(
                        component, x,
                        create_graph=create_graph,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if gradient is not None:
                        divergence = divergence + gradient.reshape(x.shape[0], -1)[:, index]
        else:
            divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            if flat_velocity.requires_grad:
                for _ in range(self.hutchinson_samples):
                    probe = torch.randn_like(x)
                    probe_flat = probe.reshape(probe.shape[0], -1)
                    product = (flat_velocity * probe_flat).sum()
                    if not product.requires_grad:
                        continue
                    gradient = torch.autograd.grad(
                        product, x, create_graph=create_graph, retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if gradient is not None:
                        divergence = divergence + (
                            gradient.reshape(gradient.shape[0], -1) * probe_flat
                        ).sum(dim=1)
                divergence = divergence / self.hutchinson_samples
        if not create_graph:
            velocity = velocity.detach()
            divergence = divergence.detach()
        return velocity, divergence

    def _integrated_log_prob(self, samples, timestep, scheduler, *, create_graph):
        if not 0 <= int(timestep) < scheduler.num_train_timesteps:
            raise ValueError("timestep is outside the scheduler range")
        terminal_index = max(1, scheduler.num_train_timesteps - 1)
        start_time = float(timestep) / terminal_index
        if create_graph:
            x = samples if samples.requires_grad else samples.detach().requires_grad_(True)
        else:
            x = samples.detach()
        accumulated_divergence = torch.zeros(
            x.shape[0], device=x.device, dtype=x.dtype
        )

        if start_time < 1.0:
            step_size = (1.0 - start_time) / self.integration_steps

            def rhs(position, time_value):
                times = torch.full(
                    (position.shape[0],), time_value,
                    device=position.device, dtype=position.dtype,
                )
                return self._field_and_divergence(
                    position, times, create_graph=create_graph
                )

            for step in range(self.integration_steps):
                time_value = start_time + step * step_size
                k1_x, k1_d = rhs(x, time_value)
                k2_x, k2_d = rhs(
                    x + 0.5 * step_size * k1_x,
                    time_value + 0.5 * step_size,
                )
                k3_x, k3_d = rhs(
                    x + 0.5 * step_size * k2_x,
                    time_value + 0.5 * step_size,
                )
                k4_x, k4_d = rhs(x + step_size * k3_x, time_value + step_size)
                x = x + (step_size / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
                accumulated_divergence = accumulated_divergence + (step_size / 6.0) * (
                    k1_d + 2 * k2_d + 2 * k3_d + k4_d
                )

        dimensions = x[0].numel()
        variance = self.prior_std**2
        flat = x.reshape(x.shape[0], -1)
        log_prior = (
            -0.5 * dimensions * math.log(2 * math.pi * variance)
            -0.5 * flat.square().sum(dim=1) / variance
        )
        return log_prior + accumulated_divergence

    def log_prob(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        # SamplingPipeline runs under no_grad; divergence evaluation still
        # requires local autograd even when the caller only needs a scalar.
        with torch.enable_grad():
            return self._integrated_log_prob(
                samples, timestep, scheduler, create_graph=False
            ).detach()

    def score(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        """Differentiate the integrated log density with respect to its input."""
        with torch.enable_grad():
            inputs = samples.detach().requires_grad_(True)
            log_density = self._integrated_log_prob(
                inputs, timestep, scheduler, create_graph=True
            )
            score = torch.autograd.grad(log_density.sum(), inputs)[0]
        return score.detach()


class FlowBirthDeathAcceleration(SamplingModifier):
    """Apply the common population-KDE BD rule using a supplied exact target."""

    def __init__(
        self,
        flow_target: TargetDistribution,
        *,
        every_k_steps: int = 1,
        total_time: float = 1.0,
        bandwidth: float = 0.3,
        min_timestep: int | None = 100,
        max_timestep: int | None = None,
        max_delta_tau: float = 0.005,
        max_event_probability: float | None = 0.25,
        add_noise: float = 0.0,
    ):
        if every_k_steps < 1 or total_time <= 0 or bandwidth <= 0:
            raise ValueError("every_k_steps, total_time, and bandwidth must be positive")
        if max_delta_tau <= 0 or add_noise < 0:
            raise ValueError("max_delta_tau must be positive and add_noise non-negative")
        self.flow_target = flow_target
        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)
        self.bandwidth = float(bandwidth)
        self.min_timestep = int(min_timestep) if min_timestep is not None else None
        self.max_timestep = int(max_timestep) if max_timestep is not None else None
        self.max_delta_tau = float(max_delta_tau)
        self.max_event_probability = max_event_probability
        self.add_noise = float(add_noise)

    def _target_coordinates(self, samples, context, scheduler):
        target_coordinates = getattr(self.flow_target, "state_coordinates", None)
        if target_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError("flow target must declare x_t or normalized_x_t coordinates")
        if context.state_coordinates == target_coordinates:
            return samples
        if context.state_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError(f"unsupported sampler coordinates {context.state_coordinates!r}")
        if context.state_timestep is None:
            raise ValueError("coordinate conversion requires a noisy-state timestep")
        alpha_bar = scheduler.alpha_bars[context.state_timestep].to(samples)
        scale = alpha_bar.sqrt()
        return samples / scale if context.state_coordinates == "x_t" else samples * scale

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
        timestep = context.state_timestep
        if self.min_timestep is not None and timestep < self.min_timestep:
            return state
        if self.max_timestep is not None and timestep > self.max_timestep:
            return state
        samples = state.samples
        population = samples.shape[0]
        if population < 2:
            return state

        target_samples = self._target_coordinates(samples, context, scheduler)
        log_target = self.flow_target.log_prob(target_samples, timestep, scheduler)
        flat = target_samples.reshape(population, -1)
        dimensions = flat.shape[1]
        distances = torch.cdist(flat, flat).square()
        log_kernel = -0.5 * distances / self.bandwidth**2
        log_population = (
            torch.logsumexp(log_kernel, dim=1) - math.log(population)
            - 0.5 * dimensions * math.log(2 * math.pi * self.bandwidth**2)
        )
        beta = log_population - log_target
        beta = beta - beta.mean()
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
            "flow_bd_events": births + deaths,
            "flow_bd_births": births,
            "flow_bd_deaths": deaths,
            "flow_mean_exact_logp": float(log_target.mean().item()),
            "acceleration_mode": "flow_target_birth_death",
        })
        return SamplingState(
            updated, state.timestep, state.previous_timestep, metadata,
            context=state.context, source_samples=state.source_samples,
        )
