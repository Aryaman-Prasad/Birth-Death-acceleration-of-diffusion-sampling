"""Maximum mean discrepancy metrics for generated samples and references."""

from __future__ import annotations

import torch


@torch.no_grad()
def rbf_mmd_squared(samples: torch.Tensor, references: torch.Tensor,
                    bandwidth_factors: tuple[float, ...] = (0.5, 1.0, 2.0)
                    ) -> tuple[dict[str, object], dict[str, object]]:
    """Compute biased, multi-scale RBF MMD-squared in flattened sample space.

    A median-distance heuristic sets the central RBF bandwidth from the
    reference subset. The biased estimator includes diagonal kernel terms and
    is non-negative up to floating point error. On images, this is pixel-space
    MMD; it should not be interpreted as a perceptual similarity metric.

    Returns metric values separately from metadata such as the actual
    bandwidths, so callers can store both in their run record.
    """
    if samples.ndim < 2 or references.ndim < 2:
        raise ValueError("samples and references must have shape (N, *sample_shape)")
    if samples.shape[0] < 1 or references.shape[0] < 1:
        raise ValueError("MMD requires at least one generated and one reference sample")
    if tuple(samples.shape[1:]) != tuple(references.shape[1:]):
        raise ValueError(
            f"sample shapes differ: {tuple(samples.shape[1:])} vs "
            f"{tuple(references.shape[1:])}"
        )
    if not bandwidth_factors or any(factor <= 0 for factor in bandwidth_factors):
        raise ValueError("bandwidth_factors must contain positive values")

    # Float32 makes this practical for image tensors while avoiding accidental
    # half-precision underflow in high-dimensional pairwise distances.
    x = samples.detach().reshape(samples.shape[0], -1).to(dtype=torch.float32)
    y = references.detach().reshape(references.shape[0], -1).to(
        device=x.device, dtype=torch.float32
    )
    reference_distances = torch.pdist(y, p=2)
    positive_distances = reference_distances[reference_distances > 0]
    if positive_distances.numel() == 0:
        base_bandwidth = 1.0
    else:
        base_bandwidth = float(positive_distances.median().item())
        # Identical references are ignored above; keep a finite kernel scale
        # for near-identical references too.
        base_bandwidth = max(base_bandwidth, 1e-6)

    d_xx = torch.cdist(x, x).square()
    d_yy = torch.cdist(y, y).square()
    d_xy = torch.cdist(x, y).square()
    by_bandwidth: dict[str, float] = {}
    actual_bandwidths: dict[str, float] = {}
    for factor in bandwidth_factors:
        bandwidth = base_bandwidth * factor
        denominator = 2.0 * bandwidth * bandwidth
        k_xx = torch.exp(-d_xx / denominator)
        k_yy = torch.exp(-d_yy / denominator)
        k_xy = torch.exp(-d_xy / denominator)
        value = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
        key = f"{factor:g}x_median"
        by_bandwidth[key] = max(0.0, float(value.item()))
        actual_bandwidths[key] = bandwidth

    metrics = {
        "empirical_target_mmd2_rbf_multiscale": sum(by_bandwidth.values()) / len(by_bandwidth),
        "empirical_target_mmd2_rbf_by_bandwidth": by_bandwidth,
    }
    metadata = {
        "empirical_target_mmd_space": "flattened sample coordinates (pixels for images)",
        "empirical_target_mmd_estimator": "biased RBF MMD-squared; multiscale mean",
        "empirical_target_mmd_bandwidth_rule": (
            "median nonzero reference pairwise distance; fallback 1 for a single "
            "or identical reference set"
        ),
        "empirical_target_mmd_median_reference_distance": base_bandwidth,
        "empirical_target_mmd_actual_bandwidths": actual_bandwidths,
        "empirical_target_mmd_sample_count": int(samples.shape[0]),
        "empirical_target_mmd_reference_count": int(references.shape[0]),
    }
    return metrics, metadata
