"""Sweep birth-death (BD) against plain DDIM/DDPM on the 2D toy mixture.

This is the diffusion analogue of ``bd_paper_toy_diagnostic.py``: it uses the
project's own pipeline, oracle score model, exact time-dependent target density,
and quality metrics, and reports NLL, MMD, mode coverage/TV, exact-duplicate
fraction, NFE, and wall-clock.

The sweep is deliberately matched on sampler steps (and therefore NFE) so that
any BD gain is not confused with simply taking more reverse steps.  Use
``--counts``, ``--steps``, ``--etas``, ``--bandwidths`` and ``--total-times`` to
choose the grid; ``bandwidth=None`` (the default entry added automatically) is
the no-BD baseline.

Example:
    python3 tests/bd_diffusion_sweep.py --counts 256 --steps 3,5,10,20,50 \
        --etas 0,1 --bandwidths 0.3,0.7,1.2 --total-times 0.1,1,10 \
        --output samples/bd-diffusion-sweep/metrics.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import toy_bd_experiment as tbe  # noqa: E402
from diffusion.acceleration.birth_death import BirthDeathAcceleration  # noqa: E402
from diffusion.metrics import measure_sampling  # noqa: E402
from diffusion.models import make_model  # noqa: E402
from diffusion.pipeline import SamplingPipeline  # noqa: E402
from diffusion.samplers import make_sampler  # noqa: E402
from diffusion.schedulers import make_scheduler  # noqa: E402
from diffusion.targets import ToyGaussianMixtureTarget  # noqa: E402


def run_case(model, target, refs, scheduler, *, count, steps, eta, sampler_name,
             bandwidth, total_time, every_k, seed, device):
    sampler = make_sampler(sampler_name, eta=eta)
    inference_steps = None if sampler_name == "ddpm" else steps
    sampler_steps = len(sampler.inference_timesteps(scheduler, inference_steps, device))
    modifiers = []
    if bandwidth is not None:
        modifiers.append(BirthDeathAcceleration(
            target, bandwidth, every_k_steps=every_k, total_time=total_time))
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)
    gen = torch.Generator(device=device).manual_seed(seed)
    mgen = torch.Generator(device=device).manual_seed(seed + 100_003)
    start = time.perf_counter()
    samples, sm = measure_sampling(
        model, device, count, sampler_steps,
        lambda: pipeline.sample(model, (count, 2), device, inference_steps, gen, mgen))
    elapsed = time.perf_counter() - start
    record = sm.to_dict()
    record.update(tbe._toy_quality(samples.detach(), target, refs))
    # DDIM eta=0 is deterministic, so an exact BD duplicate stays a duplicate.
    unique = torch.unique(samples.reshape(count, -1), dim=0).shape[0]
    record["unique_fraction"] = unique / count
    record.update(dict(sampler=sampler_name, steps=steps, eta=eta,
                       bandwidth=bandwidth, total_time=total_time,
                       every_k=every_k, seed=seed, elapsed_recomputed=elapsed))
    return record


def summarize(rows, keys):
    out = {}
    for key in keys:
        values = [row[key] for row in rows]
        out[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default="256")
    parser.add_argument("--steps", default="3,5,10,20,50")
    parser.add_argument("--etas", default="0,1")
    parser.add_argument("--bandwidths", default="0.3,0.7,1.2")
    parser.add_argument("--total-times", default="0.1,1,10")
    parser.add_argument("--every-k", type=int, default=1)
    parser.add_argument("--seeds", default="7,17,29,41")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--output", default="samples/bd-diffusion-sweep/metrics.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    target = ToyGaussianMixtureTarget().to(device)
    scheduler = make_scheduler("linear", args.timesteps, 1e-4, 2e-2).to(device)
    model = make_model("toy-mixture-oracle", target=target,
                       alpha_bars=scheduler.alpha_bars).to(device).eval()
    ref_gen = torch.Generator(device=device).manual_seed(12_345)
    refs = tbe._sample_target(target, 8192, ref_gen)

    counts = [int(value) for value in args.counts.split(",")]
    steps_list = [int(value) for value in args.steps.split(",")]
    etas = [float(value) for value in args.etas.split(",")]
    # None is the no-BD baseline; it is always included exactly once.
    bandwidths = [None] + [float(value) for value in args.bandwidths.split(",")]
    total_times = [float(value) for value in args.total_times.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]

    rows = []
    for count in counts:
        for steps in steps_list:
            for eta in etas:
                for bandwidth in bandwidths:
                    for total_time in ([1.0] if bandwidth is None else total_times):
                        for seed in seeds:
                            row = run_case(
                                model, target, refs, scheduler, count=count, steps=steps,
                                eta=eta, sampler_name=args.sampler, bandwidth=bandwidth,
                                total_time=total_time, every_k=args.every_k,
                                seed=seed, device=device,
                            )
                            rows.append(row)
                            print(
                                f"count={count} steps={steps} eta={eta} h={bandwidth} "
                                f"T={total_time} seed={seed} "
                                f"NLL={row['clean_target_nll']:.4f} "
                                f"MMD={row['clean_target_mmd2_rbf_multiscale']:.5f} "
                                f"cov={row['mode_coverage']:.3f} "
                                f"TV={row['mode_balance_tv']:.3f} "
                                f"uniq={row['unique_fraction']:.3f} "
                                f"t={row['elapsed_seconds']:.3f}",
                                flush=True,
                            )

    groups = {}
    for row in rows:
        key = (row["sample_count"], row["steps"], row["eta"],
               row["bandwidth"], row["total_time"])
        groups.setdefault(key, []).append(row)
    keys = ["nfe", "clean_target_nll", "sampling_target_nll",
            "clean_target_mmd2_rbf_multiscale", "mode_coverage",
            "mode_balance_tv", "unique_fraction", "elapsed_seconds"]
    summary = []
    for key, values in groups.items():
        entry = {"count": key[0], "steps": key[1], "eta": key[2],
                 "bandwidth": key[3], "total_time": key[4]}
        entry.update(summarize(values, keys))
        summary.append(entry)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2) + "\n")
    print(f"\nSaved {output.resolve()}")

    print(f"\n{'count':>5} {'steps':>5} {'eta':>4} {'h':>5} {'T':>5} {'NLL':>8} "
          f"{'MMD':>9} {'cov':>6} {'TV':>6} {'uniq':>6} {'time':>7}")
    for entry in summary:
        print(f"{entry['count']:>5} {entry['steps']:>5} {entry['eta']:>4.0f} "
              f"{str(entry['bandwidth']):>5} {entry['total_time']:>5.1f} "
              f"{entry['clean_target_nll']['mean']:>8.4f} "
              f"{entry['clean_target_mmd2_rbf_multiscale']['mean']:>9.5f} "
              f"{entry['mode_coverage']['mean']:>6.3f} "
              f"{entry['mode_balance_tv']['mean']:>6.3f} "
              f"{entry['unique_fraction']['mean']:>6.3f} "
              f"{entry['elapsed_seconds']['mean']:>7.3f}")


if __name__ == "__main__":
    main()
