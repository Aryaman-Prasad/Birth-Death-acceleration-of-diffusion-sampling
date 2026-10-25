"""Compare toy-mixture diffusion sampling with and without birth-death.

Run from the repository root, for example:

    python3 tests/toy_bd_experiment.py --sampler ddim --steps 50 --count 256

This is a small end-to-end experiment runner, not a unit-test suite. It uses
the analytic toy target so generated samples can be scored against its exact
clean density and its eight known modes.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.metrics import measure_sampling
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import ToyGaussianMixtureTarget


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _sample_target(target: ToyGaussianMixtureTarget, count: int,
                   generator: torch.Generator) -> torch.Tensor:
    mode_indices = torch.randint(
        target.centers.shape[0], (count,), device=target.centers.device, generator=generator
    )
    noise = torch.randn(
        (count, target.centers.shape[1]), device=target.centers.device,
        dtype=target.centers.dtype, generator=generator,
    )
    return target.centers[mode_indices] + target.standard_deviation.to(target.centers) * noise


@torch.no_grad()
def _rbf_mmd_squared(samples: torch.Tensor, references: torch.Tensor,
                     bandwidth: float) -> float:
    """Biased (non-negative) squared MMD for one RBF bandwidth."""
    distances_xx = torch.cdist(samples, samples).square()
    distances_yy = torch.cdist(references, references).square()
    distances_xy = torch.cdist(samples, references).square()
    scale = 2 * bandwidth ** 2
    kernel_xx = torch.exp(-distances_xx / scale)
    kernel_yy = torch.exp(-distances_yy / scale)
    kernel_xy = torch.exp(-distances_xy / scale)

    xx = kernel_xx.mean()
    yy = kernel_yy.mean()
    return float((xx + yy - 2 * kernel_xy.mean()).item())


@torch.no_grad()
def _toy_quality(samples: torch.Tensor, target: ToyGaussianMixtureTarget,
                 references: torch.Tensor) -> dict:
    """Score clean samples under p_0 and report occupancy of the known modes."""
    samples = samples.to(target.centers)
    clean_log_prob = target.log_prob_at_noise(samples, sigma=0.0)
    distances_squared = torch.cdist(samples, target.centers).square()
    nearest_modes = distances_squared.argmin(dim=1)
    counts = torch.bincount(nearest_modes, minlength=target.centers.shape[0])
    frequencies = counts.to(samples.dtype) / samples.shape[0]
    uniform_frequency = 1.0 / target.centers.shape[0]
    mmd_by_bandwidth = {
        str(bandwidth): _rbf_mmd_squared(samples, references, bandwidth)
        for bandwidth in (0.2, 0.5, 1.0, 2.0, 4.0)
    }
    return {
        "clean_target_nll": float(-clean_log_prob.mean().item()),
        "clean_target_mmd2_rbf_multiscale": sum(mmd_by_bandwidth.values()) / len(mmd_by_bandwidth),
        "clean_target_mmd2_rbf_by_bandwidth": mmd_by_bandwidth,
        "mode_coverage": float((counts > 0).float().mean().item()),
        "mode_balance_tv": float(
            0.5 * (frequencies - uniform_frequency).abs().sum().item()
        ),
        "mode_counts": [int(count) for count in counts.cpu().tolist()],
    }


def _run_case(model, target, references, scheduler, device, args,
              sampler_name: str, birth_death: bool) -> tuple[dict, torch.Tensor]:
    sampler = make_sampler(sampler_name, eta=args.eta)
    inference_steps = None if sampler_name == "ddpm" else args.steps
    sampler_steps = len(sampler.inference_timesteps(scheduler, inference_steps, device))
    modifiers = []
    if birth_death:
        modifiers.append(BirthDeathAcceleration(
            target,
            args.bd_kernel_bandwidth,
            every_k_steps=args.bd_every_k,
            total_time=args.bd_total_time,
        ))
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.count, *target.centers.shape[1:])
    samples, sampling_metrics = measure_sampling(
        model,
        device,
        args.count,
        sampler_steps,
        lambda: pipeline.sample(
            model, shape, device, inference_steps, generator
        ),
    )
    record = sampling_metrics.to_dict()
    record.update(_toy_quality(samples, target, references))
    record.update({
        "sampler": sampler_name,
        "birth_death_enabled": birth_death,
        "seed": args.seed,
        "bd_kernel_bandwidth": args.bd_kernel_bandwidth if birth_death else None,
        "bd_every_k_steps": args.bd_every_k if birth_death else None,
        "bd_total_time": args.bd_total_time if birth_death else None,
    })
    return record, samples.detach().cpu()


def _print_results(results: list[dict]) -> None:
    print("\nToy Gaussian-mixture comparison (lower MMD^2 / mode-balance TV is better):")
    print(
        f"{'sampler':<8} {'method':<10} {'steps':>6} {'NFE':>6} "
        f"{'time(s)':>10} {'sec/sample':>12} {'clean NLL':>12} {'MMD^2':>10} "
        f"{'coverage':>10} {'mode TV':>10}"
    )
    for result in results:
        method = "BD" if result["birth_death_enabled"] else "baseline"
        print(
            f"{result['sampler']:<8} {method:<10} {result['sampler_steps']:>6} "
            f"{result['nfe']:>6} {result['elapsed_seconds']:>10.3f} "
            f"{result['seconds_per_sample']:>12.5f} "
            f"{result['clean_target_nll']:>12.4f} "
            f"{result['clean_target_mmd2_rbf_multiscale']:>10.5f} "
            f"{result['mode_coverage']:>10.3f} "
            f"{result['mode_balance_tv']:>10.3f}"
        )
        print(f"  mode counts: {result['mode_counts']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare toy-mixture DDPM/DDIM sampling with and without birth-death"
    )
    parser.add_argument("--sampler", choices=("ddpm", "ddim", "both"), default="ddim")
    parser.add_argument("--steps", type=int, default=50,
                        help="DDIM updates; DDPM always uses all scheduler timesteps")
    parser.add_argument("--timesteps", type=int, default=1000,
                        help="training noise levels in the local scheduler")
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity; zero keeps the sampler deterministic")
    parser.add_argument("--count", type=int, default=256,
                        help="particle count for each sampler trajectory; must be at least 2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or another torch device")
    parser.add_argument("--bd-kernel-bandwidth", type=float, default=0.3)
    parser.add_argument("--bd-every-k", type=int, default=1)
    parser.add_argument("--bd-total-time", type=float, default=1.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--mode-standard-deviation", type=float, default=0.15)
    parser.add_argument("--output-dir", default="samples/toy-bd-experiment",
                        help="directory for samples and the JSON metrics report")
    args = parser.parse_args(argv)

    if args.count < 2:
        parser.error("--count must be at least 2 for birth-death")
    if args.steps < 1 or args.timesteps < 2:
        parser.error("--steps must be positive and --timesteps must be at least 2")

    device = _device(args.device)
    target = ToyGaussianMixtureTarget(
        radius=args.radius,
        standard_deviation=args.mode_standard_deviation,
    ).to(device)
    scheduler = make_scheduler(
        args.scheduler, args.timesteps, args.beta_start, args.beta_end
    ).to(device)
    model = make_model(
        "toy-mixture-oracle", target=target, alpha_bars=scheduler.alpha_bars
    ).to(device)
    model.eval()
    reference_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    references = _sample_target(target, max(args.count, 512), reference_generator)

    sampler_names = ("ddpm", "ddim") if args.sampler == "both" else (args.sampler,)
    results = []
    samples_by_run = {}
    for sampler_name in sampler_names:
        for birth_death in (False, True):
            result, samples = _run_case(
                model, target, references, scheduler, device, args,
                sampler_name, birth_death,
            )
            results.append(result)
            run_name = f"{sampler_name}_{'bd' if birth_death else 'baseline'}"
            samples_by_run[run_name] = samples

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for run_name, samples in samples_by_run.items():
        torch.save(samples, output_dir / f"{run_name}.pt")
    report = {
        "experiment": "toy_gaussian_mixture_birth_death_comparison",
        "device": str(device),
        "count": args.count,
        "seed": args.seed,
        "training_timesteps": args.timesteps,
        "ddim_steps": args.steps,
        "toy_target": {
            "radius": args.radius,
            "mode_standard_deviation": args.mode_standard_deviation,
            "mode_count": target.centers.shape[0],
        },
        "results": results,
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_results(results)
    print(f"\nSaved samples and metrics to {output_dir.resolve()}")
    print(f"Metrics JSON: {report_path.resolve()}")


if __name__ == "__main__":
    main()
