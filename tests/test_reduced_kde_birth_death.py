"""Tests for Reduced-KDE Birth-Death Acceleration and DDPM U-Net FeatureReducer."""

import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from diffusion.acceleration.base import SamplingState
from diffusion.acceleration.reduced_kde_birth_death import (
    DDPMUNetEncoderTrunk,
    FeatureReducer,
    ReducedKDEBirthDeathAcceleration,
)
from diffusion.sampling import SamplingContext
from diffusion.schedulers.linear import LinearScheduler
from diffusion.targets.toy_mixture import ToyGaussianMixtureTarget
from sample_reduced_kde import parse_args


class MockEmpiricalTarget(nn.Module):
    state_coordinates = "normalized_x_t"

    def __init__(self, reference_samples):
        super().__init__()
        self.register_buffer("reference_samples", reference_samples)

    def noise_std(self, timestep, scheduler, *, device, dtype):
        return torch.tensor(0.5, device=device, dtype=dtype)

    def log_prob(self, samples, timestep, scheduler):
        return torch.zeros(samples.shape[0], device=samples.device)


def test_ddpm_unet_encoder_trunk_dimensions():
    """Verify that the DDPM U-Net bottleneck features have dimension strictly < 1024."""
    trunk = DDPMUNetEncoderTrunk(base_channels=64, layer="bottleneck")
    assert trunk.out_dim == 128, f"Expected 128 dims, got {trunk.out_dim}"
    assert trunk.out_dim < 1024, "Output dimension must be strictly less than 1024"

    x = torch.randn(4, 3, 32, 32)
    t = torch.tensor([50, 100, 200, 500])
    out = trunk(x, timesteps=t)
    assert out.shape == (4, 128), f"Expected shape (4, 128), got {out.shape}"
    assert torch.isfinite(out).all()


def test_ddpm_unet_encoder_layer_downblock1():
    """Verify that earlier encoder layers also have dimension < 1024."""
    trunk = DDPMUNetEncoderTrunk(base_channels=64, layer="down_block1")
    assert trunk.out_dim == 64
    assert trunk.out_dim < 1024

    x = torch.randn(4, 3, 32, 32)
    out = trunk(x)
    assert out.shape == (4, 64)


def test_ddpm_unet_encoder_time_conditioning():
    """Verify that timestep conditioning alters the extracted representations."""
    trunk = DDPMUNetEncoderTrunk(base_channels=64, layer="bottleneck")
    x = torch.randn(2, 3, 32, 32)
    out_t10 = trunk(x, timesteps=10)
    out_t900 = trunk(x, timesteps=900)
    assert not torch.allclose(out_t10, out_t900), "Timestep conditioning must modulate features"


def test_feature_reducer_image_projection():
    reducer = FeatureReducer(reduced_dim=25, backbone="bottleneck", projection_method="orthogonal")
    x = torch.randn(8, 3, 32, 32)
    out = reducer(x)
    assert out.shape == (8, 25), f"Expected shape (8, 25), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_feature_reducer_pca_fitting():
    reducer = FeatureReducer(reduced_dim=10, backbone="bottleneck", projection_method="pca")
    ref_samples = torch.randn(64, 3, 32, 32)
    reducer.fit_pca(ref_samples)
    assert reducer.is_pca_fitted

    out = reducer(ref_samples)
    assert out.shape == (64, 10)
    # Check mean is close to 0 and std close to 1
    mean = out.mean(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-2)


def test_feature_reducer_vector_passthrough():
    reducer = FeatureReducer(reduced_dim=25, backbone="bottleneck")
    # 2D vector data (toy mixture)
    x = torch.randn(16, 2)
    out = reducer(x)
    assert out.shape == (16, 2), "Low-dimensional vector data should pass through intact"


def test_reduced_kde_birth_death_step():
    device = torch.device("cpu")
    scheduler = LinearScheduler(num_train_timesteps=100)
    ref_samples = torch.randn(32, 3, 32, 32)
    target = MockEmpiricalTarget(ref_samples)

    modifier = ReducedKDEBirthDeathAcceleration(
        target=target,
        kernel_bandwidth=0.5,
        kde_reduce=25,
        every_k_steps=1,
        total_time=1.0,
        backbone="bottleneck",
    )

    particles = torch.randn(16, 3, 32, 32)
    context = SamplingContext(
        state_coordinates="normalized_x_t",
        state_timestep=50,
        is_clean_sample=False,
        step_index=0,
        num_steps=10,
        has_clean_endpoint=False,
    )
    state = SamplingState(samples=particles, timestep=50, previous_timestep=40, context=context)
    gen = torch.Generator().manual_seed(42)

    new_state = modifier.apply(state, model=None, scheduler=scheduler, generator=gen)
    assert new_state.samples.shape == particles.shape
    assert "reduced_kde_dim" in new_state.metadata
    assert new_state.metadata["reduced_kde_dim"] == 25
    assert new_state.metadata["reduced_kde_feature_type"] == "ddpm_unet_encoder"
    assert "birth_death_events" in new_state.metadata


def test_cli_kde_reduce_arguments():
    # Test --kde_reduce
    args = parse_args(["--toy-mixture-oracle", "--kde_reduce", "25"])
    assert args.kde_reduce == 25

    # Test --kde-reduce alias
    args = parse_args(["--toy-mixture-oracle", "--kde-reduce", "16"])
    assert args.kde_reduce == 16
