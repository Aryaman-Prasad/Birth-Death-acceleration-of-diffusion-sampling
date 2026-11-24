"""Run all current acceleration extensions on analytic toy targets.

Run from the repository root:

    python3 tests/toy_acceleration_extensions.py --device cpu

The DDIM block compares the same exact 2D Gaussian-mixture score sampler with
each extension. The CNF block is a separate fixed-target ULA comparison: an
identity CNF represents an exact standard Gaussian, and its log density is
used by Flow-BD. These results are toy diagnostics, not evidence of image
quality or of the extensions' theoretical convergence claims.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from diffusion.acceleration import (
    BirthDeathAcceleration,
    ContinuousNormalizingFlowTarget,
    FlowBirthDeathAcceleration,
    ParticleFilteringSolutionSpaceExplorer,
    ReducedKDEBirthDeathAcceleration,
    SMCTransitionLikelihoodAcceleration,
)
from diffusion.metrics import measure_sampling
from diffusion.models.toy_mixture_oracle import ToyMixtureOracleModel
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import DDIMSampler, LangevinSampler
from diffusion.schedulers.linear import LinearScheduler
from diffusion.targets.toy_mixture import ToyGaussianMixtureTarget
from toy_bd_experiment import _sample_target, _toy_quality


class ZeroVelocity(nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class StandardGaussianEpsilonModel(nn.Module):
    """Epsilon predictor whose normalized-state score is standard Gaussian."""

    def __init__(self, alpha_bars: torch.Tensor):
        super().__init__()
        self.register_buffer("alpha_bars", alpha_bars.detach().clone())

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].to(x)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        alpha_bar = alpha_bar.reshape(shape)
        noise_std = ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()
        normalized = x / alpha_bar.sqrt()
        return noise_std * normalized


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _toy_mixture_cases(target, model, scheduler, device, args, references):
    ddim = DDIMSampler(eta=args.eta)
    method_builders = {
        "baseline": lambda: None,
        "birth_death_kde": lambda: BirthDeathAcceleration(
            target, args.bandwidth, every_k_steps=args.every_k,
            total_time=args.bd_total_time, min_timestep=0,
        ),
        # In 2D, this exercises the common reduced-KDE BD logic and exact toy
        # density, but the feature reducer correctly passes vectors through.
        "reduced_kde_2d_passthrough": lambda: ReducedKDEBirthDeathAcceleration(
            target, args.bandwidth, kde_reduce=2, backbone="pca",
            every_k_steps=args.every_k, total_time=args.bd_total_time,
            min_timestep=0, noise_references=False,
        ),
        "transition_likelihood_heuristic": lambda: SMCTransitionLikelihoodAcceleration(
            every_k_steps=args.every_k, total_time=args.bd_total_time,
            min_timestep=0, max_event_probability=0.15,
        ),
        "particle_filter_explorer": lambda: ParticleFilteringSolutionSpaceExplorer(
            branch_factor=args.branch_factor, every_k_steps=args.every_k,
            temperature=1.0, diversity_weight=args.diversity_weight,
            total_time=args.bd_total_time,
            min_timestep=0, max_event_probability=0.15,
        ),
        # The modifier consumes an exact target log density. Here the exact
        # target is the toy mixture, not a CNF; CNF itself is exercised below.
        "flow_bd_exact_toy_density": lambda: FlowBirthDeathAcceleration(
            target, every_k_steps=args.every_k, total_time=args.bd_total_time,
            bandwidth=args.bandwidth, min_timestep=0,
        ),
    }
    results = []
    saved_samples = {}
    for method_name, build_modifier in method_builders.items():
        modifier = build_modifier()
        pipeline = SamplingPipeline(ddim, scheduler, modifiers=[] if modifier is None else [modifier])
        sampler_generator = torch.Generator(device=device).manual_seed(args.seed)
        modifier_generator = torch.Generator(device=device).manual_seed(args.seed + 100_003)
        shape = (args.count, 2)
        steps = len(ddim.inference_timesteps(scheduler, args.steps, device))
        samples, timed = measure_sampling(
            model, device, args.count, steps,
            lambda: pipeline.sample(
                model, shape, device, args.steps,
                sampler_generator, modifier_generator,
            ),
        )
        result = timed.to_dict()
        result.update(_toy_quality(samples, target, references))
        result.update({
            "sampler": "ddim",
            "method": method_name,
            "eta": args.eta,
            "seed": args.seed,
            "bd_every_k": args.every_k if modifier is not None else None,
            "bd_total_time": args.bd_total_time if modifier is not None else None,
            "bandwidth": args.bandwidth if modifier is not None else None,
        })
        results.append(result)
        saved_samples[method_name] = samples.detach().cpu()
    return results, saved_samples


@torch.no_grad()
def _gaussian_quality(samples, references):
    samples = samples.float()
    references = references.to(samples)
    mean = samples.mean(dim=0)
    covariance = torch.cov(samples.T)
    identity = torch.eye(samples.shape[1], device=samples.device, dtype=samples.dtype)
    log_prob = -0.5 * (
        samples.shape[1] * math.log(2 * math.pi) + samples.square().sum(dim=1)
    )
    # Reuse an RBF MMD implementation through the toy diagnostic helper's
    # formula, evaluated directly against the standard-normal references.
    mmd_by_bandwidth = {}
    for bandwidth in (0.5, 1.0, 2.0):
        d_xx = torch.cdist(samples, samples).square()
        d_yy = torch.cdist(references, references).square()
        d_xy = torch.cdist(samples, references).square()
        mmd = (
            torch.exp(-d_xx / (2 * bandwidth**2)).mean()
            + torch.exp(-d_yy / (2 * bandwidth**2)).mean()
            - 2 * torch.exp(-d_xy / (2 * bandwidth**2)).mean()
        )
        mmd_by_bandwidth[str(bandwidth)] = float(mmd.item())
    return {
        "target_nll": float(-log_prob.mean().item()),
        "mean_norm": float(mean.norm().item()),
        "covariance_frobenius_error": float(torch.linalg.matrix_norm(covariance - identity).item()),
        "mmd2_rbf_multiscale": sum(mmd_by_bandwidth.values()) / len(mmd_by_bandwidth),
        "mmd2_rbf_by_bandwidth": mmd_by_bandwidth,
    }


def _cnf_gaussian_cases(scheduler, device, args):
    target = ContinuousNormalizingFlowTarget(
        ZeroVelocity(), prior_std=1.0, divergence_method="exact",
        integration_steps=8, state_coordinates="normalized_x_t",
    ).to(device)
    model = StandardGaussianEpsilonModel(scheduler.alpha_bars).to(device)
    sampler = LangevinSampler(
        timestep=0, step_size=args.gaussian_step_size,
        initial_scale=args.gaussian_initial_scale,
    )
    references_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    references = torch.randn((max(args.count, 512), 2), device=device, generator=references_generator)
    results = []
    saved_samples = {}
    for use_bd in (False, True):
        modifier = None
        method_name = "ula_baseline"
        if use_bd:
            modifier = FlowBirthDeathAcceleration(
                target, every_k_steps=args.gaussian_every_k,
                total_time=args.gaussian_bd_total_time,
                bandwidth=args.gaussian_bandwidth,
                min_timestep=0,
                max_event_probability=0.15,
            )
            method_name = "ula_with_cnf_flow_bd"
        pipeline = SamplingPipeline(sampler, scheduler, modifiers=[] if modifier is None else [modifier])
        sampler_generator = torch.Generator(device=device).manual_seed(args.seed + 17)
        modifier_generator = torch.Generator(device=device).manual_seed(args.seed + 100_017)
        samples, timed = measure_sampling(
            model, device, args.count, args.gaussian_steps,
            lambda: pipeline.sample(
                model, (args.count, 2), device, args.gaussian_steps,
                sampler_generator, modifier_generator,
            ),
        )
        result = timed.to_dict()
        result.update(_gaussian_quality(samples, references))
        result.update({
            "sampler": "fixed_target_langevin",
            "method": method_name,
            "target": "standard_gaussian_from_identity_cnf",
            "cnf_divergence_method": "exact",
            "langevin_step_size": args.gaussian_step_size,
            "langevin_initial_scale": args.gaussian_initial_scale,
            "bd_every_k": args.gaussian_every_k if use_bd else None,
            "bd_total_time": args.gaussian_bd_total_time if use_bd else None,
        })
        results.append(result)
        saved_samples[method_name] = samples.detach().cpu()
    return results, saved_samples


def _print_table(results):
    print("\nToy extension comparisons (lower NLL/MMD/TV is generally better):")
    print(
        f"{'sampler':<22} {'method':<38} {'NFE':>6} {'time(s)':>9} "
        f"{'NLL':>10} {'MMD^2':>10} {'coverage':>9} {'mode TV':>9}"
    )
    for result in results:
        nll = result.get("clean_target_nll", result.get("target_nll", float("nan")))
        mmd = result.get(
            "clean_target_mmd2_rbf_multiscale",
            result.get("mmd2_rbf_multiscale", float("nan")),
        )
        coverage = result.get("mode_coverage", float("nan"))
        mode_tv = result.get("mode_balance_tv", float("nan"))
        print(
            f"{result['sampler']:<22} {result['method']:<38} "
            f"{result['nfe']:>6} {result['elapsed_seconds']:>9.3f} "
            f"{nll:>10.4f} {mmd:>10.5f} {coverage:>9.3f} {mode_tv:>9.3f}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--steps", type=int, default=25, help="DDIM steps on the mixture target")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=1.0,
                        help="must be positive for transition scoring and branching")
    parser.add_argument("--every-k", type=int, default=2)
    parser.add_argument("--bd-total-time", type=float, default=0.2)
    parser.add_argument("--bandwidth", type=float, default=0.7)
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--diversity-weight", type=float, default=0.1,
                        help="particle-filter candidate dispersion bonus")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--mode-standard-deviation", type=float, default=0.15)
    parser.add_argument("--gaussian-steps", type=int, default=300)
    parser.add_argument("--gaussian-step-size", type=float, default=0.01)
    parser.add_argument("--gaussian-initial-scale", type=float, default=3.0)
    parser.add_argument("--gaussian-every-k", type=int, default=5)
    parser.add_argument("--gaussian-bd-total-time", type=float, default=0.5)
    parser.add_argument("--gaussian-bandwidth", type=float, default=0.8)
    parser.add_argument("--output-dir", default="samples/toy-acceleration-extensions")
    args = parser.parse_args(argv)
    if args.count < 2 or args.steps < 1 or args.timesteps < 2:
        parser.error("count must be >= 2, steps >= 1, and timesteps >= 2")
    if args.eta <= 0:
        parser.error("--eta must be positive for transition likelihood and branching")
    if args.every_k < 1 or args.gaussian_every_k < 1 or args.gaussian_steps < 1:
        parser.error("intervention cadence and Gaussian ULA steps must be positive")

    device = _device(args.device)
    scheduler = LinearScheduler(num_train_timesteps=args.timesteps).to(device)
    mixture_target = ToyGaussianMixtureTarget(
        radius=args.radius, standard_deviation=args.mode_standard_deviation
    ).to(device)
    mixture_model = ToyMixtureOracleModel(
        mixture_target, scheduler.alpha_bars
    ).to(device).eval()
    reference_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    mixture_references = _sample_target(
        mixture_target, max(args.count, 512), reference_generator
    )
    mixture_results, mixture_samples = _toy_mixture_cases(
        mixture_target, mixture_model, scheduler, device, args, mixture_references
    )
    gaussian_results, gaussian_samples = _cnf_gaussian_cases(scheduler, device, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_samples = {**mixture_samples, **gaussian_samples}
    for name, samples in all_samples.items():
        torch.save(samples, output_dir / f"{name}.pt")
    report = {
        "experiment": "toy_acceleration_extensions",
        "device": str(device),
        "count": args.count,
        "seed": args.seed,
        "mixture_target": {
            "kind": "eight_component_isotropic_gaussian_mixture",
            "radius": args.radius,
            "component_standard_deviation": args.mode_standard_deviation,
            "sampler": "DDIM",
            "steps": args.steps,
            "eta": args.eta,
            "methods": mixture_results,
        },
        "cnf_gaussian_target": {
            "kind": "standard_gaussian_from_identity_CNF",
            "sampler": "fixed_target_Langevin",
            "steps": args.gaussian_steps,
            "methods": gaussian_results,
        },
        "interpretation_notes": [
            "Reduced-KDE on 2D vectors passes the vectors through; this does not test image feature reduction.",
            "The Flow-BD mixture case uses the exact analytic mixture density; the separate Gaussian case tests the integrated CNF density itself.",
            "Transition-likelihood resampling is a heuristic score, not a complete target/proposal SMC importance correction.",
            "Particle filtering is a branch/filter exploration heuristic, not canonical SMC.",
            "NFE counts model forwards made during sampling, including extra forwards from transition methods; elapsed time includes modifier work.",
        ],
    }
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _print_table(mixture_results + gaussian_results)
    print(f"\nSaved samples and metrics under {output_dir.resolve()}")
    print(f"Metrics JSON: {report_path.resolve()}")


if __name__ == "__main__":
    main()
