"""Run a paper-inspired, known-density BD-vs-ULA toy experiment.

This diagnostic uses the four-component, two-dimensional Gaussian mixture and
single-mode initialization from Section 6.2 of Lu, Lu, and Nolen (2019). It is
intended to test whether BD improves finite-particle mixing on a case with
actual mode exploration headroom; it is not a complete reproduction of their
larger experiment.

Example:
    python3 tests/bd_paper_toy_diagnostic.py --steps 5000 --count 128 \
        --bandwidths 0.3,0.7,1.2 --output samples/bd-paper-toy-diagnostic/metrics.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration.base import SamplingState
from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


class PaperGaussianMixture(TargetDistribution, nn.Module):
    """The four anisotropic Gaussian components used in the paper's Example 2."""

    state_coordinates = "normalized_x_t"

    def __init__(self):
        nn.Module.__init__(self)
        self.register_buffer("means", torch.tensor([
            [0.0, 8.0], [0.0, 2.0], [-3.0, 5.0], [3.0, 5.0],
        ]))
        # Diagonal entries of Sigma_1,...,Sigma_4.
        self.register_buffer("variances", torch.tensor([
            [1.2, 0.01], [1.2, 0.01], [0.01, 2.0], [0.01, 2.0],
        ]))

    def component_log_probs(self, samples: torch.Tensor) -> torch.Tensor:
        values = samples[:, None, :] - self.means.to(samples)[None, :, :]
        variances = self.variances.to(samples)[None, :, :]
        return (
            -0.5 * (values.square() / variances).sum(dim=-1)
            -0.5 * torch.log(2 * math.pi * variances).sum(dim=-1)
            -math.log(self.means.shape[0])
        )

    def log_prob(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        return torch.logsumexp(self.component_log_probs(samples), dim=1)

    def score(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        log_components = self.component_log_probs(samples)
        weights = torch.softmax(log_components, dim=1)
        variances = self.variances.to(samples)[None, :, :]
        component_scores = (self.means.to(samples)[None, :, :] - samples[:, None, :]) / variances
        return (weights[:, :, None] * component_scores).sum(dim=1)


def _metrics(samples: torch.Tensor, target: PaperGaussianMixture) -> dict:
    responsibilities = torch.softmax(target.component_log_probs(samples), dim=1).mean(dim=0)
    target_weights = torch.full_like(responsibilities, 1.0 / responsibilities.numel())
    mean = samples.mean(dim=0)
    target_mean = target.means.mean(dim=0)
    return {
        "mean_nll": float(-target.log_prob(samples, 0, None).mean().item()),
        "component_responsibility_tv": float(
            0.5 * (responsibilities - target_weights).abs().sum().item()
        ),
        "component_responsibilities": responsibilities.tolist(),
        "mean_squared_error": float((mean - target_mean).square().sum().item()),
        "sample_mean": mean.tolist(),
    }


def _run(target: PaperGaussianMixture, count: int, steps: int, step_size: float,
         bandwidth: float | None, every_k_steps: int, seed: int,
         checkpoints: set[int]) -> list[dict]:
    sampler_generator = torch.Generator().manual_seed(seed)
    modifier_generator = torch.Generator().manual_seed(seed + 100_003)
    initial_mean = torch.tensor([0.0, 8.0])
    samples = initial_mean + (0.3 ** 0.5) * torch.randn(
        (count, 2), generator=sampler_generator
    )
    modifier = None if bandwidth is None else BirthDeathAcceleration(
        target,
        kernel_bandwidth=bandwidth,
        every_k_steps=every_k_steps,
        # Match the paper's birth-death Delta t to the ULA integration step.
        total_time=steps * step_size,
    )
    records = []
    for step_index in range(steps):
        noise = torch.randn(samples.shape, generator=sampler_generator)
        samples = (
            samples + step_size * target.score(samples, 0, None)
            + math.sqrt(2 * step_size) * noise
        )
        if modifier is not None:
            context = SamplingContext(
                state_timestep=0,
                state_coordinates="normalized_x_t",
                step_index=step_index,
                num_steps=steps,
                has_clean_endpoint=False,
            )
            state = SamplingState(samples, 0, 0, context=context)
            samples = modifier.apply(
                state,
                model=None,
                # This target does not depend on scheduler noise, so scheduler is unused.
                scheduler=None,
                generator=modifier_generator,
            ).samples
        if step_index + 1 in checkpoints:
            records.append({"step": step_index + 1, **_metrics(samples, target)})
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=1e-3)
    parser.add_argument("--bd-every-k", type=int, default=1,
                        help="number of ULA steps between BD interventions")
    parser.add_argument("--bandwidths", default="0.3,0.7,1.2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="samples/bd-paper-toy-diagnostic/metrics.json")
    args = parser.parse_args(argv)
    if args.count < 2 or args.steps < 1 or args.step_size <= 0 or args.bd_every_k < 1:
        parser.error("count must be >= 2; steps, step-size, and bd-every-k must be positive")
    try:
        bandwidths = [float(value) for value in args.bandwidths.split(",")]
    except ValueError as error:
        parser.error(f"invalid --bandwidths: {error}")
    if any(value <= 0 for value in bandwidths):
        parser.error("all bandwidths must be positive")

    target = PaperGaussianMixture()
    checkpoint_values = sorted({
        min(args.steps, value) for value in (100, 500, 1000, 2500, 5000, 10000)
    })
    checkpoints = set(checkpoint_values)
    runs = {}
    elapsed_seconds = {}
    methods = [("ula", None)] + [(f"bd_h{bandwidth:g}", bandwidth) for bandwidth in bandwidths]
    for name, bandwidth in methods:
        start = time.perf_counter()
        runs[name] = _run(
            target, args.count, args.steps, args.step_size, bandwidth,
            args.bd_every_k, args.seed, checkpoints,
        )
        elapsed_seconds[name] = time.perf_counter() - start

    report = {
        "description": "Paper-inspired fixed-target Gaussian-mixture mixing diagnostic",
        "paper_reference": "Lu, Lu, Nolen (2019), Section 6.2, arXiv:1905.09863",
        "count": args.count,
        "steps": args.steps,
        "step_size": args.step_size,
        "bd_every_k_steps": args.bd_every_k,
        "bd_total_time": args.steps * args.step_size,
        "seed": args.seed,
        "elapsed_seconds": elapsed_seconds,
        "target_means": target.means.tolist(),
        "target_diagonal_variances": target.variances.tolist(),
        "runs": runs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("method    time(s)   step  NLL       component-TV  mean-squared-error  component responsibilities")
    for name, records in runs.items():
        print(f"{name:<9} {elapsed_seconds[name]:>7.3f}")
        for record in records:
            print(
                f"{name:<9} {record['step']:>5}  {record['mean_nll']:>8.3f}  "
                f"{record['component_responsibility_tv']:>12.3f}  "
                f"{record['mean_squared_error']:>18.3f}  "
                f"{[round(value, 2) for value in record['component_responsibilities']]}"
            )
    print(f"Saved metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
