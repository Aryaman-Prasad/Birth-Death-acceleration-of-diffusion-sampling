"""Standalone Evaluation Suite for Sequential Monte Carlo (SMC) Reverse Transition Likelihood Weighting.

Compares:
1. Baseline DDIM (No Birth-Death)
2. SMC Reverse Transition Weighting (O(N) analytical reverse kernel likelihood, 0 distance matrices)

Usage:
    python3 scripts/run_smc_eval.py --dataset cifar10 --download --device cuda --steps 10 25 50
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration import SMCTransitionLikelihoodAcceleration
from diffusion.datasets import make_data_module
from diffusion.metrics import FIDEvaluator, measure_sampling, rbf_mmd_squared
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import EmpiricalNoisedDataTarget


def _unique_fraction(samples: torch.Tensor) -> float:
    flat = samples.reshape(samples.shape[0], -1)
    unique_count = torch.unique(flat, dim=0).shape[0]
    return float(unique_count / samples.shape[0])


def run_smc_case(
    *,
    model,
    scheduler,
    steps: int,
    eta: float,
    sample_shape: tuple[int, ...],
    count: int,
    device: torch.device,
    seed: int,
    target=None,
    use_smc: bool = False,
    every_k: int = 2,
    total_time: float = 0.01,
    temperature: float = 1.0,
    real_images: torch.Tensor | None = None,
    fid_samples: int = 256,
    fid_batch_size: int = 32,
):
    """Executes a single sampling case with latency, NFE, diversity, MMD, and FID."""
    sampler = make_sampler("ddim", eta=eta)
    sampler_steps = len(sampler.inference_timesteps(scheduler, steps, device))

    modifiers = []
    if use_smc:
        modifiers.append(
            SMCTransitionLikelihoodAcceleration(
                every_k_steps=every_k,
                total_time=total_time,
                temperature=temperature,
            )
        )

    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)

    # 1. Preview generation and diversity
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (count, *sample_shape)
    preview_samples, sampling_metrics = measure_sampling(
        model,
        device,
        count,
        sampler_steps,
        lambda: pipeline.sample(model, shape, device, steps, gen),
    )
    unique_ratio = _unique_fraction(preview_samples)

    # 2. Quality Evaluation: MMD and NLL
    mmd_score = None
    nll_score = None
    ref_tensor = getattr(target, "reference_samples", getattr(target, "references", None))
    if target is not None and ref_tensor is not None:
        try:
            n_refs = min(512, ref_tensor.shape[0])
            ref_subset = ref_tensor[:n_refs].to(device=device, dtype=torch.float32)
            eval_samples = preview_samples.to(device=device, dtype=torch.float32)
            mmd_metrics, _ = rbf_mmd_squared(eval_samples, ref_subset)
            mmd_score = float(mmd_metrics["empirical_target_mmd2_rbf_multiscale"])
        except Exception:
            mmd_score = None

        try:
            alpha_bar = scheduler.alpha_bars[0].to(device)
            density_samples = preview_samples.to(device) / alpha_bar.sqrt()
            log_prob = target.log_prob(density_samples, 0, scheduler)
            if torch.isfinite(log_prob).any():
                nll_score = float((-log_prob[torch.isfinite(log_prob)].mean()).item())
            else:
                log_prob_s = target.log_prob(density_samples, 50, scheduler)
                if torch.isfinite(log_prob_s).any():
                    nll_score = float((-log_prob_s[torch.isfinite(log_prob_s)].mean()).item())
        except Exception:
            nll_score = None

    # 3. FID evaluation
    fid_score = None
    if real_images is not None and fid_samples > 0:
        fid_evaluator = FIDEvaluator(device)
        for start in range(0, real_images.shape[0], fid_batch_size):
            chunk = real_images[start : start + fid_batch_size].to(device)
            fid_evaluator.update_real(chunk)

        remaining = fid_samples
        fid_gen = torch.Generator(device=device).manual_seed(seed + 999)
        while remaining > 0:
            b_count = min(fid_batch_size, remaining)
            b_shape = (b_count, *sample_shape)
            fake_batch = pipeline.sample(model, b_shape, device, steps, fid_gen)
            fid_evaluator.update_fake(fake_batch)
            remaining -= b_count
        fid_score = float(fid_evaluator.compute())

    record = {
        "sampler": "ddim",
        "steps": steps,
        "method": "smc_transition" if use_smc else "baseline",
        "every_k": every_k if use_smc else None,
        "total_time": total_time if use_smc else None,
        "temperature": temperature if use_smc else None,
        "nfe": sampling_metrics.nfe,
        "elapsed_seconds": round(sampling_metrics.elapsed_seconds, 4),
        "samples_per_second": round(sampling_metrics.samples_per_second, 2),
        "unique_fraction": round(unique_ratio, 4),
        "mmd": round(mmd_score, 6) if mmd_score is not None else None,
        "nll": round(nll_score, 2) if nll_score is not None else None,
        "fid": round(fid_score, 3) if fid_score is not None else None,
    }
    return record, preview_samples


def main():
    parser = argparse.ArgumentParser(description="Standalone SMC Transition Likelihood Evaluation")
    parser.add_argument("--dataset", choices=("cifar10", "imagefolder"), default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--reference-count", type=int, default=512)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--smc-every-k", type=int, default=2)
    parser.add_argument("--smc-total-time", type=float, default=0.01)
    parser.add_argument("--smc-temp", type=float, default=1.0)
    parser.add_argument("--output-dir", default="smc_benchmark_results")

    args = parser.parse_args()
    if getattr(args, "no_fid", False):
        args.fid_samples = 0

    device = torch.device(
        "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"[+] Initializing SMC Transition Evaluation Suite on {device}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Reference Target
    data_module = make_data_module(
        args.dataset, args.data_dir, args.image_size, args.download, train=True
    )
    sample_shape = tuple(data_module.sample_shape)
    dataset = data_module.dataset()
    ref_count = min(args.reference_count, len(dataset))
    target = EmpiricalNoisedDataTarget.from_dataset(
        dataset, num_reference_samples=ref_count, seed=args.seed, reference_chunk_size=64
    )

    # 2. Scheduler and Target-Score Model
    scheduler = make_scheduler("linear", 1000, 1e-4, 2e-2).to(device)
    model = make_model("target-score", target=target, scheduler=scheduler).to(device)
    model.eval()

    # 3. Reference images for FID
    real_images = None
    if args.fid_samples > 0:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.fid_batch_size, shuffle=True)
        img_list = []
        accumulated = 0
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(args.fid_samples - accumulated, imgs.shape[0])
            img_list.append(imgs[:take])
            accumulated += take
            if accumulated >= args.fid_samples:
                break
        real_images = torch.cat(img_list, dim=0)

    # 4. Run Cases across Steps
    results = []
    for st in args.steps:
        # A) Baseline DDIM
        case_name_base = f"ddim_{st}_baseline"
        print(f"\n[Case: {case_name_base}] Running Baseline DDIM at {st} steps...")
        rec_base, prev_base = run_smc_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            use_smc=False,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_base["case_name"] = case_name_base
        results.append(rec_base)
        grid_base = (prev_base.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_base, samples_dir / f"{case_name_base}.png", nrow=4)
        print(f"   -> Samples/s: {rec_base['samples_per_second']} | Unique: {rec_base['unique_fraction']} | MMD: {rec_base.get('mmd')} | FID: {rec_base.get('fid')}")

        # B) SMC Reverse Transition
        case_name_smc = f"ddim_{st}_smc_transition"
        print(f"\n[Case: {case_name_smc}] Running SMC Transition Weighting at {st} steps...")
        rec_smc, prev_smc = run_smc_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            use_smc=True,
            every_k=args.smc_every_k,
            total_time=args.smc_total_time,
            temperature=args.smc_temp,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_smc["case_name"] = case_name_smc
        results.append(rec_smc)
        grid_smc = (prev_smc.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_smc, samples_dir / f"{case_name_smc}.png", nrow=4)
        print(f"   -> Samples/s: {rec_smc['samples_per_second']} | Unique: {rec_smc['unique_fraction']} | MMD: {rec_smc.get('mmd')} | FID: {rec_smc.get('fid')}")

    # Save CSV and JSON
    csv_file = out_dir / "smc_benchmark_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_file = out_dir / "smc_benchmark_metrics.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Presentation Plots
    try:
        import matplotlib.pyplot as plt
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: FID across steps
        plt.figure(figsize=(7, 4.5), dpi=150)
        for meth, label, color, style in [
            ("baseline", "Baseline DDIM", "#1f77b4", "o-"),
            ("smc_transition", "SMC Transition Weighting (O(N))", "#ff7f0e", "v-"),
        ]:
            pts = [r for r in results if r["method"] == meth and r.get("fid") is not None]
            if pts:
                pts_sorted = sorted(pts, key=lambda x: x["steps"])
                plt.plot(
                    [p["steps"] for p in pts_sorted],
                    [p["fid"] for p in pts_sorted],
                    style,
                    label=label,
                    color=color,
                    linewidth=2,
                    markersize=7,
                )
        plt.xlabel("Inference Steps", fontsize=11)
        plt.ylabel("FID Score (Lower is Better)", fontsize=11)
        plt.title(f"SMC Reverse Transition Quality ({args.dataset.upper()})", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig1_smc_steps_vs_fid.png")
        plt.close()

        # Plot 2: Throughput
        plt.figure(figsize=(7, 4.5), dpi=150)
        st_ref = 25 if 25 in args.steps else args.steps[0]
        pts_sp = [r for r in results if r["steps"] == st_ref]
        plt.bar(
            [p["method"] for p in pts_sp],
            [p["samples_per_second"] for p in pts_sp],
            color=["#1f77b4", "#ff7f0e"],
            width=0.4,
        )
        plt.ylabel("Throughput (Samples/sec)", fontsize=11)
        plt.title(f"Throughput Comparison at {st_ref} Steps", fontsize=12, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig2_smc_throughput.png")
        plt.close()
    except Exception as e:
        print(f"[!] Plot note: {e}")

    print("\n" + "=" * 80)
    print(f"[+] SMC Benchmark Complete! Results saved to:")
    print(f"    - CSV:   {csv_file}")
    print(f"    - JSON:  {json_file}")
    print(f"    - Grids: {samples_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
