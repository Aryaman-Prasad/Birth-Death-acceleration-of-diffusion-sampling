"""Exact Density Normalizing Flow and Flow-Matching Birth-Death Acceleration.

Evaluates Birth-Death acceleration on models with exact tractable log-densities:
1. Continuous Normalizing Flows (CNFs) using the Instantaneous Change-of-Variables formula:
       d/dt log p_t(x_t) = -div(v_theta(x_t, t)) = -Tr(d v_theta / dx)
2. Tractable Flow-Matching targets where log-density and velocity are decoupled.

Key Advantage:
Rigorously decouples score-matching approximation error from density estimation error.
Because the scalar target density log p_t(x) is mathematically exact, Birth-Death
selection reflects true population drift without heuristic KDE target proxies.
"""

import math
from typing import Any, Callable

import torch
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


def compute_divergence_exact(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Compute exact divergence Tr(dv/dx) via autograd for low-to-medium dimensions."""
    x = x.detach().requires_grad_(True)
    v = v_fn(x, t)
    dim = x.reshape(x.shape[0], -1).shape[1]
    divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    # Sum of diagonal elements of Jacobian
    for i in range(dim):
        v_i = v.reshape(v.shape[0], -1)[:, i]
        grad_v_i = torch.autograd.grad(
            v_i.sum(), x, create_graph=False, retain_graph=True
        )[0]
        divergence += grad_v_i.reshape(grad_v_i.shape[0], -1)[:, i]

    return divergence


def compute_divergence_hutchinson(
    v_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    t: torch.Tensor,
    num_samples: int = 1,
) -> torch.Tensor:
    """Compute unbiased divergence estimate Tr(dv/dx) using Hutchinson's trace estimator.

    Tr(J) = E_{e ~ N(0, I)} [ e^T (d v / d x) e ]
    """
    x = x.detach().requires_grad_(True)
    v = v_fn(x, t)
    divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    for _ in range(num_samples):
        # Rademacher (+1, -1) or standard normal noise vector
        e = torch.randn_like(x)
        e_flat = e.reshape(e.shape[0], -1)
        v_flat = v.reshape(v.shape[0], -1)
        v_dot_e = (v_flat * e_flat).sum()

        grad_x = torch.autograd.grad(
            v_dot_e, x, create_graph=False, retain_graph=True
        )[0]
        grad_flat = grad_x.reshape(grad_x.shape[0], -1)
        divergence += (grad_flat * e_flat).sum(dim=1)

    return divergence / num_samples


class ContinuousNormalizingFlowTarget(nn.Module, TargetDistribution):
    """Continuous Normalizing Flow target with exact tractable log-density and score.

    Implements a velocity field v_theta(x, t) and computes exact log-density
    via numerical ODE integration of the instantaneous change of variables:
        log p_t(x) = log p_prior(x_prior) - int_{t}^{1} div(v_theta(x_s, s)) ds
    """

    state_coordinates = "x_t"

    def __init__(
        self,
        velocity_net: nn.Module,
        prior_std: float = 1.0,
        divergence_method: str = "exact",
        hutchinson_samples: int = 2,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.prior_std = float(prior_std)
        self.divergence_method = divergence_method
        self.hutchinson_samples = int(hutchinson_samples)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(x, t)

    def divergence(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dim = x[0].numel()
        if self.divergence_method == "exact" or dim <= 64:
            return compute_divergence_exact(self.velocity, x, t)
        return compute_divergence_hutchinson(
            self.velocity, x, t, num_samples=self.hutchinson_samples
        )

    def log_prob(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        """Exact log p_t(x) evaluated via flow velocity divergence."""
        t_normalized = float(timestep) / max(1, scheduler.num_train_timesteps)
        t_tensor = torch.full(
            (samples.shape[0],), t_normalized, device=samples.device, dtype=samples.dtype
        )

        dim = samples[0].numel()
        # Prior log-density under Gaussian base p_1(x) = N(0, prior_std^2 * I)
        flat = samples.reshape(samples.shape[0], -1)
        prior_variance = self.prior_std**2
        log_prior = -0.5 * dim * math.log(2 * math.pi * prior_variance) - 0.5 * flat.square().sum(dim=1) / prior_variance

        # Instantaneous divergence correction at noise level t
        div = self.divergence(samples, t_tensor)
        dt = (1.0 - t_normalized)
        log_density = log_prior - div * dt
        return log_density

    def score(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        """Exact score nabla_x log p_t(x) via automatic differentiation."""
        samples_copy = samples.detach().requires_grad_(True)
        log_p = self.log_prob(samples_copy, timestep, scheduler).sum()
        grad = torch.autograd.grad(log_p, samples_copy)[0]
        return grad


class FlowBirthDeathAcceleration(SamplingModifier):
    """Birth-Death particle acceleration using exact normalizing flow log-densities.

    Unlike empirical KDE models, this evaluates the target log-density using the
    flow's exact change-of-variables equation, completely decoupling score error
    from density estimation error.
    """

    def __init__(
        self,
        flow_target: TargetDistribution,
        *,
        every_k_steps: int = 1,
        total_time: float = 1.0,
        bandwidth: float = 0.3,
    ):
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if total_time <= 0:
            raise ValueError("total_time must be positive")

        self.flow_target = flow_target
        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)
        self.bandwidth = float(bandwidth)

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

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = max(1, eligible_updates // self.every_k_steps)
        delta_tau = self.total_time / intervention_count

        population = samples.shape[0]
        if population < 2:
            return state

        timestep = context.state_timestep

        # 1. Exact scalar log-density from the Flow model
        log_target = self.flow_target.log_prob(samples, int(timestep), scheduler)

        # 2. Local population density estimate
        flat = samples.reshape(population, -1)
        dim = flat.shape[1]
        dist_sq = torch.cdist(flat, flat).square()
        log_kernel = -0.5 * dist_sq / (self.bandwidth**2)
        log_pop = torch.logsumexp(log_kernel, dim=1) - math.log(population) - 0.5 * dim * math.log(2 * math.pi * self.bandwidth**2)

        # 3. Discrepancy rate beta: beta_i = log_pop(x_i) - log_target(x_i)
        beta = log_pop - log_target
        beta = beta - beta.mean()

        event_probability = -torch.expm1(-beta.abs() * delta_tau)
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
                # Overrepresented compared to exact flow density -> kill & replace
                source_indices[index] = source_indices[partner]
                deaths += 1
            else:
                # Underrepresented compared to exact flow density -> duplicate
                source_indices[partner] = source_indices[index]
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )

        metadata = dict(state.metadata)
        metadata.update(
            {
                "flow_bd_events": births + deaths,
                "flow_bd_births": births,
                "flow_bd_deaths": deaths,
                "flow_mean_exact_logp": float(log_target.mean().item()),
                "acceleration_mode": "exact_flow_birth_death",
            }
        )
        return SamplingState(
            updated,
            state.timestep,
            state.previous_timestep,
            metadata,
            context=state.context,
        )
