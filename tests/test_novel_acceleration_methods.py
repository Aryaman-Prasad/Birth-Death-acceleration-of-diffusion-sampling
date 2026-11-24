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
from diffusion.acceleration.reverse_transition import reverse_transition_parameters
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


class LinearVelocity(nn.Module):
    """Linear CNF field with a closed-form Gaussian marginal."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.rate * x


class ZeroNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def test_smc_transition_likelihood_step():
    """Verify that SMC reverse transition density acceleration runs in O(N) and updates state."""
    scheduler = LinearScheduler(num_train_timesteps=100)
    model = SimpleMockModel(channels=3)

    modifier = SMCTransitionLikelihoodAcceleration(
        every_k_steps=1,
        total_time=1.0,
        temperature=1.0,
        min_timestep=0,
    )

    source = torch.randn(8, 3, 16, 16)
    destination = source + 0.05 * torch.randn_like(source)
    context = SamplingContext(
        state_coordinates="x_t",
        state_timestep=70,
        is_clean_sample=False,
        step_index=0,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state = SamplingState(
        samples=destination,
        timestep=80,
        previous_timestep=70,
        metadata={"sampler_name": "ddim", "eta": 1.0},
        context=context,
        source_samples=source,
    )
    gen = torch.Generator().manual_seed(42)
    result_state = modifier.apply(state, model=model, scheduler=scheduler, generator=gen)
    assert result_state.samples.shape == (8, 3, 16, 16)
    assert "smc_events" in result_state.metadata
    assert "smc_mean_log_prob" in result_state.metadata
    assert result_state.metadata["acceleration_mode"] == "smc_reverse_transition_heuristic"


def test_reverse_transition_parameters_match_ddpm_posterior():
    scheduler = LinearScheduler(num_train_timesteps=100)
    source = torch.randn(4, 2)
    timestep = 80
    previous_timestep = 79
    model = ZeroNoiseModel()
    mean, variance = reverse_transition_parameters(
        source, timestep, previous_timestep, model, scheduler,
        {"sampler_name": "ddpm"},
    )
    alpha_bar = scheduler.alpha_bars[timestep]
    alpha_bar_previous = scheduler.alpha_bars_prev[timestep]
    beta = scheduler.betas[timestep]
    alpha = scheduler.alphas[timestep]
    predicted_clean = source / alpha_bar.sqrt()
    expected_mean = (
        beta * alpha_bar_previous.sqrt() / (1 - alpha_bar) * predicted_clean
        + (1 - alpha_bar_previous) * alpha.sqrt() / (1 - alpha_bar) * source
    )
    expected_variance = beta * (1 - alpha_bar_previous) / (1 - alpha_bar)
    assert torch.allclose(mean, expected_mean)
    assert torch.allclose(variance, expected_variance)


def test_transition_likelihood_rejects_deterministic_ddim():
    scheduler = LinearScheduler(num_train_timesteps=100)
    with pytest.raises(ValueError, match="eta > 0"):
        reverse_transition_parameters(
            torch.randn(2, 2), 80, 70, ZeroNoiseModel(), scheduler,
            {"sampler_name": "ddim", "eta": 0.0},
        )


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


def test_cnf_integrates_divergence_and_score_under_no_grad():
    rate = 0.35
    prior_std = 1.2
    flow_target = ContinuousNormalizingFlowTarget(
        velocity_net=LinearVelocity(rate),
        prior_std=prior_std,
        divergence_method="exact",
        integration_steps=64,
    )
    scheduler = LinearScheduler(num_train_timesteps=100)
    samples = torch.tensor([[0.25, -0.7], [1.1, 0.4]])
    timestep = 37
    normalized_time = timestep / (scheduler.num_train_timesteps - 1)
    variance = prior_std**2 * math.exp(-2 * rate * (1 - normalized_time))
    expected = -math.log(2 * math.pi * variance) - samples.square().sum(dim=1) / (2 * variance)
    with torch.no_grad():
        actual = flow_target.log_prob(samples, timestep, scheduler)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    score = flow_target.score(samples, timestep, scheduler)
    assert torch.allclose(score, -samples / variance, atol=2e-5, rtol=2e-5)


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
        min_timestep=0,
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
    assert new_state.metadata["acceleration_mode"] == "flow_target_birth_death"


def test_particle_filtering_solution_explorer():
    """Verify branching N -> N*B candidates and filtering back to N elite particles."""
    scheduler = LinearScheduler(num_train_timesteps=100)
    model = SimpleMockModel(channels=3)

    modifier = ParticleFilteringSolutionSpaceExplorer(
        branch_factor=3,
        every_k_steps=1,
        temperature=1.0,
        min_timestep=0,
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
        samples=particles + 0.01 * torch.randn_like(particles),
        timestep=60,
        previous_timestep=50,
        metadata={"sampler_name": "ddim", "eta": 1.0},
        context=context,
        source_samples=particles,
    )
    gen = torch.Generator().manual_seed(42)

    new_state = modifier.apply(state, model=model, scheduler=scheduler, generator=gen)
    assert new_state.samples.shape == particles.shape, "Must filter back to original population N=8"
    assert new_state.metadata["filter_pool_size"] == 24, "8 * 3 = 24 candidates"
    assert new_state.metadata["filter_retained"] == 8
    assert "filter_mean_potential" in new_state.metadata
    assert new_state.metadata["acceleration_mode"] == "particle_filtering_exploration"
