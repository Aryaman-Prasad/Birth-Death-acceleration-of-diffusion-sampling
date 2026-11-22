"""Unit tests for the 3 novel acceleration paradigms:
1. Sequential Monte Carlo (SMC) Reverse Transition Density Weighting
2. Exact Density Continuous Normalizing Flows (CNFs)
3. Particle Filtering Solution Space Exploration
"""

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion.acceleration import (
    ContinuousNormalizingFlowTarget,
    FlowBirthDeathAcceleration,
    ParticleFilteringSolutionSpaceExplorer,
    SMCTransitionLikelihoodAcceleration,
)
from diffusion.acceleration.base import SamplingState
from diffusion.sampling import SamplingContext
from diffusion.schedulers.linear import LinearScheduler


class SimpleMockModel(nn.Module):
    """Simple linear-decay mock model for reverse noise prediction."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(x)) * 0.1


class VelocityNet(nn.Module):
    """Simple velocity network v(x, t) for 2D flow target."""

    def __init__(self, dim: int = 2):
        super().__init__()
        self.linear1 = nn.Linear(dim + 1, 32)
        self.linear2 = nn.Linear(32, dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        xt = torch.cat([x, t], dim=1)
        return self.linear2(torch.relu(self.linear1(xt)))


def test_smc_transition_likelihood_step():
    """Verify that SMC reverse transition density acceleration runs in O(N) and updates state."""
    scheduler = LinearScheduler(num_train_timesteps=100)
    model = SimpleMockModel(channels=3)

    modifier = SMCTransitionLikelihoodAcceleration(
        every_k_steps=1,
        total_time=1.0,
        temperature=1.0,
    )

    particles_step1 = torch.randn(8, 3, 16, 16)
    context_step1 = SamplingContext(
        state_coordinates="x_t",
        state_timestep=80,
        is_clean_sample=False,
        step_index=0,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state1 = SamplingState(
        samples=particles_step1, timestep=90, previous_timestep=80, context=context_step1
    )
    gen = torch.Generator().manual_seed(42)

    # First step: caches previous state
    state2 = modifier.apply(state1, model=model, scheduler=scheduler, generator=gen)
    assert state2.samples.shape == particles_step1.shape

    # Second step: calculates p_theta(x_{t-1} | x_t) and applies birth-death
    particles_step2 = particles_step1 + 0.05 * torch.randn_like(particles_step1)
    context_step2 = SamplingContext(
        state_coordinates="x_t",
        state_timestep=70,
        is_clean_sample=False,
        step_index=1,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state3 = SamplingState(
        samples=particles_step2, timestep=80, previous_timestep=70, context=context_step2
    )

    result_state = modifier.apply(state3, model=model, scheduler=scheduler, generator=gen)
    assert result_state.samples.shape == (8, 3, 16, 16)
    assert "smc_events" in result_state.metadata
    assert "smc_mean_log_prob" in result_state.metadata
    assert result_state.metadata["acceleration_mode"] == "smc_reverse_transition"


def test_continuous_normalizing_flow_target():
    """Verify exact log-density evaluation via instantaneous divergence in CNF."""
    v_net = VelocityNet(dim=2)
    flow_target = ContinuousNormalizingFlowTarget(
        velocity_net=v_net,
        prior_std=1.0,
        divergence_method="exact",
    )
    scheduler = LinearScheduler(num_train_timesteps=100)

    samples = torch.randn(10, 2)
    log_probs = flow_target.log_prob(samples, timestep=50, scheduler=scheduler)
    assert log_probs.shape == (10,)
    assert torch.isfinite(log_probs).all(), "Log-densities must be finite"

    # Verify score nabla_x log p_t(x)
    scores = flow_target.score(samples, timestep=50, scheduler=scheduler)
    assert scores.shape == (10, 2)
    assert torch.isfinite(scores).all(), "Scores must be finite"


def test_flow_birth_death_step():
    """Verify Birth-Death acceleration using exact flow log-density."""
    v_net = VelocityNet(dim=2)
    flow_target = ContinuousNormalizingFlowTarget(
        velocity_net=v_net, prior_std=1.0, divergence_method="exact"
    )
    scheduler = LinearScheduler(num_train_timesteps=100)

    modifier = FlowBirthDeathAcceleration(
        flow_target=flow_target,
        every_k_steps=1,
        total_time=1.0,
        bandwidth=0.3,
    )

    particles = torch.randn(16, 2)
    context = SamplingContext(
        state_coordinates="x_t",
        state_timestep=50,
        is_clean_sample=False,
        step_index=0,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state = SamplingState(
        samples=particles, timestep=60, previous_timestep=50, context=context
    )
    gen = torch.Generator().manual_seed(42)

    new_state = modifier.apply(state, model=None, scheduler=scheduler, generator=gen)
    assert new_state.samples.shape == particles.shape
    assert "flow_bd_events" in new_state.metadata
    assert "flow_mean_exact_logp" in new_state.metadata
    assert new_state.metadata["acceleration_mode"] == "exact_flow_birth_death"


def test_particle_filtering_solution_explorer():
    """Verify branching N -> N*B candidates and filtering back to N elite particles."""
    scheduler = LinearScheduler(num_train_timesteps=100)
    model = SimpleMockModel(channels=3)

    modifier = ParticleFilteringSolutionSpaceExplorer(
        branch_factor=3,
        every_k_steps=1,
        temperature=1.0,
    )

    particles = torch.randn(8, 3, 16, 16)
    context = SamplingContext(
        state_coordinates="x_t",
        state_timestep=50,
        is_clean_sample=False,
        step_index=0,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state = SamplingState(
        samples=particles, timestep=60, previous_timestep=50, context=context
    )
    gen = torch.Generator().manual_seed(42)

    new_state = modifier.apply(state, model=model, scheduler=scheduler, generator=gen)
    assert new_state.samples.shape == particles.shape, "Must filter back to original population N=8"
    assert new_state.metadata["filter_pool_size"] == 24, "8 * 3 = 24 candidates"
    assert new_state.metadata["filter_retained"] == 8
    assert "filter_mean_potential" in new_state.metadata
    assert new_state.metadata["acceleration_mode"] == "particle_filtering_exploration"
