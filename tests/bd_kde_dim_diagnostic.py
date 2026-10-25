"""Explain BD behaviour in high dimension via KDE self-domination.

The paper's birth-death rate compares a kernel density estimate (KDE) of the
current particle population with the target log density.  In high dimension the
Gaussian kernel mass is dominated by each particle's own kernel, so the KDE
ordering carries almost no information about the target and BD degenerates into
"copy high-density particles".

This diagnostic draws N iid samples from a standard Gaussian (so the population
is already exactly correct and any BD change is pure added variance), applies
one or more BD interventions, and reports:

* the mean self-kernel fraction of the KDE,
* the effective number of *other* neighbours in the kernel,
* the rank correlation between KDE and target log density,
* the exact-duplicate fraction, and MMD/NLL before vs after BD.

Both small and median-heuristic bandwidths fail, for different reasons: small
``h`` makes the KDE a sum of delta functions (self-dominated), while the median
heuristic gives ``h ~ sigma * sqrt(d)``, which over-smooths the density so that
``beta`` becomes a monotone function of radius and BD copies central particles.

Example:
    python3 tests/bd_kde_dim_diagnostic.py --dims 2,8,32,128,768 \
        --counts 256,1024 --total-times 0.1,1 \
        --output samples/bd-kde-dim/metrics.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration.base import SamplingState  # noqa: E402
from diffusion.acceleration.birth_death import BirthDeathAcceleration  # noqa: E402
from diffusion.sampling import SamplingContext  # noqa: E402
from diffusion.targets.base import TargetDistribution  # noqa: E402


class StandardGaussian(TargetDistribution, nn.Module):
    state_coordinates = "normalized_x_t"

    def __init__(self, dim):
        nn.Module.__init__(self)
        self.dim = dim

    def log_prob(self, samples, timestep, scheduler):
        flat = samples.reshape(samples.shape[0], -1)
        return -0.5 * flat.square().sum(1) - 0.5 * self.dim * math.log(2 * math.pi)

    def score(self, samples, timestep, scheduler):
        return -samples


class FakeScheduler:
    """BD wants alpha_bars; the target is already in normalized coordinates."""

    def __init__(self):
        self.alpha_bars = torch.ones(1)


def mmd2(x, y, bandwidths):
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    values = [
        (torch.exp(-xx / (2 * h * h)).mean()
         + torch.exp(-yy / (2 * h * h)).mean()
         - 2 * torch.exp(-xy / (2 * h * h)).mean()).item()
        for h in bandwidths
    ]
    return sum(values) / len(values)


def median_pairwise(x):
    distances = torch.cdist(x, x)
    n = distances.shape[0]
    values = distances[torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)]
    return float(values.median().item())


def kde_diagnostics(x, h, target):
    n, dim = x.shape
    distances_squared = torch.cdist(x, x).square()
    log_kernel = -distances_squared / (2 * h * h)
    # sum_j exp(-||x_i - x_j||^2 / (2 h^2)), divided by n.
    log_kde = torch.logsumexp(log_kernel, dim=1) - math.log(n)
    # The self kernel contributes exp(0) = 1 to the sum S_i, and the normalised
    # KDE at x_i is S_i / n, so the self fraction is (1/n) / (S_i/n) = 1/S_i.
    # Since log_kde = log(S_i/n), we have 1/S_i = exp(-log_kde) / n.
    inverse_density = torch.exp(-log_kde)
    self_fraction = inverse_density / n
    other_neighbours = n / inverse_density - 1.0
    log_target = target.log_prob(x, 0, None)
    beta = log_kde - log_target
    beta = beta - beta.mean()
    rank_kde = torch.argsort(torch.argsort(log_kde)).float()
    rank_target = torch.argsort(torch.argsort(log_target)).float()
    rank_kde = rank_kde - rank_kde.mean()
    rank_target = rank_target - rank_target.mean()
    rank_correlation = float(
        (rank_kde * rank_target).sum()
        / (rank_kde.square().sum() * rank_target.square().sum()).sqrt()
    )
    return {
        "self_kde_fraction": float(self_fraction.mean().item()),
        "effective_other_neighbors": float(other_neighbours.mean().item()),
        "kde_target_rank_corr": rank_correlation,
        "beta_abs_mean": float(beta.abs().mean().item()),
        "beta_abs_max": float(beta.abs().max().item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="2,8,32,128,768")
    parser.add_argument("--counts", default="256,1024")
    parser.add_argument("--seeds", default="7,17,29")
    parser.add_argument("--bandwidths", default="0.1,0.3,0.7",
                        help="absolute bandwidths; median-based values are added automatically")
    parser.add_argument("--total-times", default="0.1,1")
    parser.add_argument("--output", default="samples/bd-kde-dim/metrics.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    dims = [int(value) for value in args.dims.split(",")]
    counts = [int(value) for value in args.counts.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    h_absolute = [float(value) for value in args.bandwidths.split(",")]
    total_times = [float(value) for value in args.total_times.split(",")]

    rows = []
    for dim in dims:
        target = StandardGaussian(dim).to(device)
        scheduler = FakeScheduler()
        for count in counts:
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                x = torch.randn(count, dim, generator=generator)
                reference_generator = torch.Generator().manual_seed(seed + 100_003)
                fresh = torch.randn(4 * count, dim, generator=reference_generator)
                median = median_pairwise(x)
                reference_bandwidth = median_pairwise(fresh[:count])
                bandwidths = h_absolute + [0.5 * median, median, 2 * median]
                for h in bandwidths:
                    diagnostics = kde_diagnostics(x, h, target)
                    for total_time in total_times:
                        modifier = BirthDeathAcceleration(
                            target, h, every_k_steps=1, total_time=total_time)
                        context = SamplingContext(
                            0, "normalized_x_t", step_index=0, num_steps=1,
                            has_clean_endpoint=False)
                        modifier_generator = torch.Generator().manual_seed(seed + 100_003)
                        updated_state = modifier.apply(
                            SamplingState(x.clone(), 0, 0, context=context),
                            model=None, scheduler=scheduler,
                            generator=modifier_generator)
                        updated = updated_state.samples
                        updated_metadata = updated_state.metadata
                        unique = torch.unique(updated, dim=0).shape[0] / count
                        mmd_bandwidths = [0.5 * reference_bandwidth,
                                          reference_bandwidth,
                                          2 * reference_bandwidth]
                        rows.append({
                            "dim": dim, "count": count, "seed": seed, "h": h,
                            "h_over_median": h / median if median > 0 else None,
                            "total_time": total_time,
                            "events": (updated_metadata["birth_death_events"] / count
                                       if updated_metadata else 0.0),
                            "unique_fraction": unique,
                            "mmd_after": mmd2(updated, fresh, mmd_bandwidths),
                            "mmd_before": mmd2(x, fresh, mmd_bandwidths),
                            "nll_before": float(-target.log_prob(x, 0, None).mean().item()),
                            "nll_after": float(-target.log_prob(updated, 0, None).mean().item()),
                            **diagnostics,
                        })
            print(f"done dim={dim} count={count}", flush=True)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved {output.resolve()}")

    print(f"{'dim':>4} {'N':>5} {'h':>8} {'self%':>6} {'neigh':>7} "
          f"{'rank':>6} {'uniq':>6} {'MMD b/a':>17} {'NLL b/a':>14}")
    for row in rows:
        if row["total_time"] != total_times[0]:
            continue
        print(f"{row['dim']:>4} {row['count']:>5} {row['h']:>8.3f} "
              f"{100 * row['self_kde_fraction']:>6.1f} "
              f"{row['effective_other_neighbors']:>7.2f} "
              f"{row['kde_target_rank_corr']:>6.2f} "
              f"{row['unique_fraction']:>6.2f} "
              f"{row['mmd_before']:>8.4f}/{row['mmd_after']:<8.4f} "
              f"{row['nll_before']:>6.2f}/{row['nll_after']:<6.2f}")


if __name__ == "__main__":
    main()
