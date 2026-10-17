"""
Comprehensive Evaluation Suite for Diffusion Samplers.
Computes:
1. Efficiency metrics: Total Latency, Latency/sample, Throughput, NFEs, Speedup ratio.
2. Trajectory dynamics: Total path length, Path straightness efficiency.
3. Distribution fidelity:
   - Sliced Wasserstein Distance (SWD)
   - Multi-scale RBF Maximum Mean Discrepancy (MMD)
   - Multimodal Mode Coverage & Mode Collapse Detection
   - Image distribution statistics
"""

import math
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple


def compute_rbf_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0]
) -> float:
    """
    Compute unbiased Maximum Mean Discrepancy (MMD) with multi-scale RBF kernels.
    """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    
    # Subsample if too large to compute pairwise distances efficiently
    max_pts = 2000
    if x.shape[0] > max_pts:
        x = x[torch.randperm(x.shape[0])[:max_pts]]
    if y.shape[0] > max_pts:
        y = y[torch.randperm(y.shape[0])[:max_pts]]

    # Compute pairwise squared distances
    xx_dist = torch.cdist(x, x, p=2) ** 2
    yy_dist = torch.cdist(y, y, p=2) ** 2
    xy_dist = torch.cdist(x, y, p=2) ** 2

    mmd_total = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma ** 2)
        k_xx = torch.exp(-gamma * xx_dist)
        k_yy = torch.exp(-gamma * yy_dist)
        k_xy = torch.exp(-gamma * xy_dist)

        # Unbiased estimator: exclude diagonal for xx and yy
        n_x = x.shape[0]
        n_y = y.shape[0]
        
        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n_x * (n_x - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n_y * (n_y - 1))
        sum_xy = k_xy.mean()

        mmd_total += (sum_xx + sum_yy - 2.0 * sum_xy).item()

    return max(0.0, mmd_total / len(sigmas))


def compute_sliced_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    num_projections: int = 64
) -> float:
    """
    Compute Sliced Wasserstein Distance (SWD) between two sets of empirical samples.
    Projects samples onto random 1D unit directions and computes analytical 1D Wasserstein-1.
    """
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    dim = x.shape[1]

    # Sample random projections on the unit sphere
    projections = torch.randn(dim, num_projections, device=x.device)
    projections = projections / torch.norm(projections, dim=0, keepdim=True)

    # Project points
    proj_x = torch.matmul(x, projections) # (N, num_proj)
    proj_y = torch.matmul(y, projections) # (M, num_proj)

    # Sort projected coordinates along each projection
    proj_x_sorted, _ = torch.sort(proj_x, dim=0)
    proj_y_sorted, _ = torch.sort(proj_y, dim=0)

    # If sample sizes differ, interpolate to match sizes
    if proj_x_sorted.shape[0] != proj_y_sorted.shape[0]:
        min_len = min(proj_x_sorted.shape[0], proj_y_sorted.shape[0])
        idx_x = torch.linspace(0, proj_x_sorted.shape[0] - 1, min_len).long()
        idx_y = torch.linspace(0, proj_y_sorted.shape[0] - 1, min_len).long()
        proj_x_sorted = proj_x_sorted[idx_x]
        proj_y_sorted = proj_y_sorted[idx_y]

    # Analytical 1D Wasserstein-1 is L1 distance between sorted order statistics
    swd = torch.mean(torch.abs(proj_x_sorted - proj_y_sorted)).item()
    return swd


def compute_multimodal_mode_coverage(
    generated_samples: torch.Tensor,
    mode_centers: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Quantifies mode coverage, missing modes, and sample allocation across Gaussian modes.
    Used for 8-Gaussians and 25-Gaussians benchmark validation.
    """
    gen = generated_samples.reshape(generated_samples.shape[0], -1)
    centers = mode_centers.to(gen.device)
    
    # Compute distances from each sample to each mode center
    dists = torch.cdist(gen, centers) # (N, num_modes)
    min_dists, nearest_modes = torch.min(dists, dim=1) # (N,)

    # Samples within threshold distance of a mode are assigned to that mode
    valid_mask = min_dists <= threshold
    valid_samples = nearest_modes[valid_mask]

    num_modes = centers.shape[0]
    mode_counts = torch.bincount(valid_samples, minlength=num_modes).float()
    
    modes_covered = int((mode_counts > 0).sum().item())
    mode_coverage_pct = (modes_covered / num_modes) * 100.0
    
    # Mode balance / entropy
    probs = mode_counts / max(1.0, mode_counts.sum().item())
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    max_entropy = math.log(num_modes)
    mode_balance = (entropy / max_entropy) * 100.0 if max_entropy > 0 else 100.0

    in_mode_fraction = (valid_mask.sum().item() / max(1, len(gen))) * 100.0

    return {
        "modes_covered": modes_covered,
        "total_modes": num_modes,
        "mode_coverage_pct": mode_coverage_pct,
        "mode_balance_pct": mode_balance,
        "in_mode_fraction_pct": in_mode_fraction,
        "mode_counts": mode_counts.cpu().tolist(),
    }


class SamplingBenchmarkEvaluator:
    """
    Unified evaluator comparing baseline DDPM against accelerated methods.
    """
    def __init__(self, ground_truth_samples: torch.Tensor, mode_centers: Optional[torch.Tensor] = None):
        self.gt_samples = ground_truth_samples.cpu()
        self.mode_centers = mode_centers.cpu() if mode_centers is not None else None

    def evaluate_sampler(
        self,
        sampler_name: str,
        generated_samples: torch.Tensor,
        info: Dict[str, Any],
        baseline_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate generated samples against ground truth for efficiency and fidelity.
        """
        gen_cpu = generated_samples.cpu()
        
        # 1. Efficiency metrics
        inference_time = info.get("inference_time", 0.0)
        nfe = info.get("nfe", 0)
        path_length = info.get("path_length", 0.0)
        num_samples = gen_cpu.shape[0]
        latency_per_sample_ms = (inference_time / num_samples) * 1000.0 if num_samples > 0 else 0.0
        throughput = num_samples / inference_time if inference_time > 0 else 0.0
        
        speedup = (baseline_time / inference_time) if baseline_time and inference_time > 0 else 1.0

        # 2. Fidelity metrics
        mmd_val = compute_rbf_mmd(gen_cpu, self.gt_samples)
        swd_val = compute_sliced_wasserstein(gen_cpu, self.gt_samples)

        result = {
            "sampler": sampler_name,
            "nfe": nfe,
            "inference_time_sec": round(inference_time, 4),
            "latency_per_sample_ms": round(latency_per_sample_ms, 2),
            "throughput_samples_per_sec": round(throughput, 2),
            "speedup_vs_baseline": round(speedup, 2),
            "path_length": round(path_length, 4),
            "mmd": round(mmd_val, 6),
            "sliced_wasserstein": round(swd_val, 6),
        }

        # 3. Multimodal coverage if mode centers provided
        if self.mode_centers is not None and gen_cpu.shape[-1] == self.mode_centers.shape[-1]:
            mode_eval = compute_multimodal_mode_coverage(gen_cpu, self.mode_centers)
            result.update({
                "modes_covered": f"{mode_eval['modes_covered']}/{mode_eval['total_modes']}",
                "mode_coverage_pct": round(mode_eval["mode_coverage_pct"], 1),
                "mode_balance_pct": round(mode_eval["mode_balance_pct"], 1),
                "in_mode_fraction_pct": round(mode_eval["in_mode_fraction_pct"], 1),
            })

        return result

