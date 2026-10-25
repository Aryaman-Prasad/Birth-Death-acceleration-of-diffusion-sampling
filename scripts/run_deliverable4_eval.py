"""Deliverable 4 Evaluation & Ablation Study Runner for Non-Toy Datasets (CIFAR-10 / CelebA).

This script automates the complete evaluation suite required for Deliverable 4:
1. Quality & Efficiency benchmarking:
   - NFE (Model Forward Calls)
   - Inference latency (sec/sample) and throughput (samples/sec)
   - Generation quality via FID (Frechet Inception Distance)
   - Population diversity via Unique Particle Fraction (diagnosing mode/particle collapse)
2. Ablation & Robustness studies:
   - Sampler step counts (10, 25, 50)
   - BD intervention cadence (--bd-every-k: 1, 2, 5)
   - BD pseudo-time budget (--bd-total-time: 0.01, 0.1, 1.0)
   - Sampler stochasticity (DDIM eta: 0.0 vs 1.0)
3. Automated compilation of presentation artifacts:
   - CSV summary table (deliverable4_results.csv)
   - Metrics JSON (deliverable4_metrics.json)
   - Matplotlib visualization figures (Speed vs Quality, Diversity vs Budget, Cadence ablation)
   - Side-by-side qualitative image grids

Usage on Kaggle:
    python3 scripts/run_deliverable4_eval.py --dataset cifar10 --device cuda
    python3 scripts/run_deliverable4_eval.py --dataset imagefolder --data-dir /kaggle/input/celeba-dataset/img_align_celeba --image-size 64 --device cuda
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

# Insert project src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.datasets import make_data_module
from diffusion.engine import load_checkpoint
from diffusion.metrics import FIDEvaluator, measure_sampling
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import EmpiricalNoisedDataTarget


def _unique_fraction(samples: torch.Tensor) -> float:
    """Calculate the ratio of distinct particles in the batch."""
    flat = samples.reshape(samples.shape[0], -1)
    unique_count = torch.unique(flat, dim=0).shape[0]
    return float(unique_count / samples.shape[0])


def run_single_experiment(
    *,
    model,
    scheduler,
    sampler_name: str,
    steps: int,
    eta: float,
    sample_shape: tuple[int, ...],
    count: int,
    device: torch.device,
    seed: int,
    target=None,
    birth_death: bool = False,
    bd_every_k: int = 2,
    bd_total_time: float = 1.0,
    bd_bandwidth: float = 0.3,
    fid_evaluator: FIDEvaluator | None = None,
    fid_samples: int = 500,
    fid_batch_size: int = 64,
):
    """Executes a single sampling run, measuring latency, NFE, diversity, and FID."""
    sampler = make_sampler(sampler_name, eta=eta)
    inference_steps = None if sampler_name == "ddpm" else steps
    sampler_steps = len(sampler.inference_timesteps(scheduler, inference_steps, device))

    modifiers = []
    if birth_death and target is not None:
        modifiers.append(
            BirthDeathAcceleration(
                target,
                bd_bandwidth,
                every_k_steps=bd_every_k,
                total_time=bd_total_time,
            )
        )

    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)

    # 1. Preview generation (for visuals and diversity metric)
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (count, *sample_shape)
    preview_samples, sampling_metrics = measure_sampling(
        model,
        device,
        count,
        sampler_steps,
        lambda: pipeline.sample(model, shape, device, inference_steps, gen),
    )

    unique_ratio = _unique_fraction(preview_samples)

    # 2. FID evaluation (if requested)
    fid_score = None
    if fid_evaluator is not None and fid_samples > 0:
        fid_evaluator.reset_fake()
        remaining = fid_samples
        fid_gen = torch.Generator(device=device).manual_seed(seed + 999)
        while remaining > 0:
            b_count = min(fid_batch_size, remaining)
            b_shape = (b_count, *sample_shape)
            fake_batch = pipeline.sample(model, b_shape, device, inference_steps, fid_gen)
            fid_evaluator.update_fake(fake_batch)
            remaining -= b_count
        fid_score = float(fid_evaluator.compute())

    record = {
        "sampler": sampler_name,
        "steps": steps,
        "eta": eta,
        "birth_death": birth_death,
        "bd_every_k": bd_every_k if birth_death else None,
        "bd_total_time": bd_total_time if birth_death else None,
        "bd_bandwidth": bd_bandwidth if birth_death else None,
        "nfe": sampling_metrics.nfe,
        "elapsed_seconds": round(sampling_metrics.elapsed_seconds, 4),
        "seconds_per_sample": round(sampling_metrics.seconds_per_sample, 4),
        "samples_per_second": round(sampling_metrics.samples_per_second, 2),
        "unique_fraction": round(unique_ratio, 4),
        "fid": round(fid_score, 3) if fid_score is not None else None,
    }
    return record, preview_samples.detach().cpu()


def generate_presentation_plots(results: list[dict], output_dir: Path):
    """Generates publication-quality charts for the presentation slides."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] matplotlib not installed. Skipping plot generation.")
        return

    out_plots = output_dir / "plots"
    out_plots.mkdir(parents=True, exist_ok=True)

    # Plot 1: Speed vs Quality (Steps vs FID)
    step_runs = [r for r in results if r.get("bd_every_k") in (None, 2) and r.get("bd_total_time") in (None, 0.01) and r.get("fid") is not None]
    if len(step_runs) >= 4:
        plt.figure(figsize=(7, 4.5), dpi=150)
        baseline_steps = sorted([r for r in step_runs if not r["birth_death"]], key=lambda x: x["steps"])
        bd_steps = sorted([r for r in step_runs if r["birth_death"]], key=lambda x: x["steps"])

        if baseline_steps:
            plt.plot([r["steps"] for r in baseline_steps], [r["fid"] for r in baseline_steps], "o-", label="Baseline DDIM (No BD)", color="#1f77b4", linewidth=2)
        if bd_steps:
            plt.plot([r["steps"] for r in bd_steps], [r["fid"] for r in bd_steps], "s--", label="DDIM + Birth-Death", color="#d62728", linewidth=2)

        plt.xlabel("Inference Steps (NFE)", fontsize=11)
        plt.ylabel("FID Score (Lower is Better)", fontsize=11)
        plt.title("Sample Quality vs. Sampling Steps", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_plots / "fig1_steps_vs_fid.png")
        plt.close()

    # Plot 2: Diversity Collapse vs. BD Time Budget
    budget_runs = [r for r in results if r["birth_death"] and r.get("bd_total_time") is not None]
    if len(budget_runs) >= 3:
        plt.figure(figsize=(7, 4.5), dpi=150)
        budget_sorted = sorted(budget_runs, key=lambda x: x["bd_total_time"])
        times = [r["bd_total_time"] for r in budget_sorted]
        uniques = [r["unique_fraction"] for r in budget_sorted]

        plt.plot(times, uniques, "o-", color="#2ca02c", linewidth=2.5, markersize=8)
        plt.axhline(1.0, linestyle=":", color="gray", label="Ideal Diversity (No duplicates)")
        plt.xlabel("BD Pseudo-Time Budget (total_time)", fontsize=11)
        plt.ylabel("Unique Particle Fraction", fontsize=11)
        plt.title("Particle Duplication / Collapse with Increasing BD Time", fontsize=13, fontweight="bold")
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(out_plots / "fig2_diversity_collapse.png")
        plt.close()

    # Plot 3: Wall-Clock Latency Comparison
    plt.figure(figsize=(7, 4.5), dpi=150)
    b_lat = [r for r in results if not r["birth_death"]]
    bd_lat = [r for r in results if r["birth_death"]]
    if b_lat and bd_lat:
        b_avg = sum(r["seconds_per_sample"] for r in b_lat) / len(b_lat)
        bd_avg = sum(r["seconds_per_sample"] for r in bd_lat) / len(bd_lat)
        plt.bar(["Baseline DDIM", "DDIM + Birth-Death"], [b_avg, bd_avg], color=["#1f77b4", "#d62728"], width=0.5)
        plt.ylabel("Seconds per Sample", fontsize=11)
        plt.title("Sampling Wall-Clock Overhead of Birth-Death", fontsize=13, fontweight="bold")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_plots / "fig3_latency_overhead.png")
        plt.close()

    print(f"[+] Presentation figures saved to: {out_plots.resolve()}/")


def main():
    parser = argparse.ArgumentParser(description="Deliverable 4 Evaluation Suite for Non-Toy Datasets")
    parser.add_argument("--dataset", choices=("cifar10", "imagefolder"), default="cifar10")
    parser.add_argument("--data-dir", default="data", help="Dataset directory")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--reference-count", type=int, default=256,
                        help="Number of real images used for the empirical target density")
    parser.add_argument("--fid-samples", type=int, default=500,
                        help="Number of generated & real images for FID (e.g. 500 for fast eval, 5000+ for final)")
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--count", type=int, default=16, help="Preview batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="deliverable4_results")
    parser.add_argument("--quick", action="store_true", help="Run a quick 6-case subset instead of full sweep")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing Deliverable 4 benchmark on {device}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Data Module & Reference Target
    data_module = make_data_module(
        args.dataset, args.data_dir, args.image_size, args.download,
        train=True,
    )
    sample_shape = tuple(data_module.sample_shape)
    dataset = data_module.dataset()
    ref_count = min(args.reference_count, len(dataset))
    print(f"[+] Building empirical target density over {ref_count} reference images...")
    target = EmpiricalNoisedDataTarget.from_dataset(
        dataset, num_reference_samples=ref_count, seed=args.seed, reference_chunk_size=64
    )

    # Scheduler and Target-Score Model
    scheduler = make_scheduler("linear", 1000, 1e-4, 2e-2).to(device)
    model = make_model("target-score", target=target, scheduler=scheduler).to(device)
    model.eval()

    # Setup FID Evaluator with real images
    fid_eval = None
    if args.fid_samples > 0:
        print(f"[+] Initializing FID evaluator with {args.fid_samples} real reference images...")
        fid_eval = FIDEvaluator(device)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.fid_batch_size, shuffle=True)
        accumulated = 0
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(args.fid_samples - accumulated, imgs.shape[0])
            fid_eval.update_real(imgs[:take])
            accumulated += take
            if accumulated >= args.fid_samples:
                break

    # 2. Experiment Grid Configurations
    # We test:
    # A) Step reduction: DDIM at 10, 25, 50 steps (Baseline vs BD)
    # B) Pseudo-time budget ablation: BD total_time in [0.01, 0.1, 1.0]
    # C) Cadence ablation: BD every_k in [1, 2, 5]
    # D) Stochasticity: eta in [0.0, 1.0]
    if args.quick:
        cases = [
            # Standard Step Comparison
            {"name": "ddim_10_nobd", "steps": 10, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
            {"name": "ddim_10_bd",   "steps": 10, "bd": True,  "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ddim_25_nobd", "steps": 25, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
            {"name": "ddim_25_bd",   "steps": 25, "bd": True,  "k": 2, "tau": 0.01, "eta": 0.0},
            # Heavy BD collapse test
            {"name": "ddim_25_bd_heavy", "steps": 25, "bd": True, "k": 2, "tau": 1.0, "eta": 0.0},
            # Stochasticity test
            {"name": "ddim_25_bd_stochastic", "steps": 25, "bd": True, "k": 2, "tau": 0.01, "eta": 1.0},
        ]
    else:
        cases = [
            # A. Step comparison (10, 25, 50 steps) Baseline vs BD
            {"name": "ddim_10_baseline", "steps": 10, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
            {"name": "ddim_10_bd",       "steps": 10, "bd": True,  "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ddim_25_baseline", "steps": 25, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
            {"name": "ddim_25_bd",       "steps": 25, "bd": True,  "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ddim_50_baseline", "steps": 50, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
            {"name": "ddim_50_bd",       "steps": 50, "bd": True,  "k": 2, "tau": 0.01, "eta": 0.0},

            # B. BD Pseudo-time Budget Ablation (steps=25, k=2)
            {"name": "ablation_time_0.01", "steps": 25, "bd": True, "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ablation_time_0.10", "steps": 25, "bd": True, "k": 2, "tau": 0.10, "eta": 0.0},
            {"name": "ablation_time_1.00", "steps": 25, "bd": True, "k": 2, "tau": 1.00, "eta": 0.0},

            # C. Cadence Ablation (steps=25, tau=0.01)
            {"name": "ablation_k_1", "steps": 25, "bd": True, "k": 1, "tau": 0.01, "eta": 0.0},
            {"name": "ablation_k_2", "steps": 25, "bd": True, "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ablation_k_5", "steps": 25, "bd": True, "k": 5, "tau": 0.01, "eta": 0.0},

            # D. Stochasticity Ablation
            {"name": "ablation_eta_0.0", "steps": 25, "bd": True, "k": 2, "tau": 0.01, "eta": 0.0},
            {"name": "ablation_eta_1.0", "steps": 25, "bd": True, "k": 2, "tau": 0.01, "eta": 1.0},
        ]

    # Deduplicate cases by parameters
    seen_params = set()
    unique_cases = []
    for c in cases:
        key = (c["steps"], c["bd"], c["k"], c["tau"], c["eta"])
        if key not in seen_params:
            seen_params.add(key)
            unique_cases.append(c)

    print(f"\n[+] Executing {len(unique_cases)} evaluation cases for Deliverable 4...")
    print("=" * 85)
    print(f"{'Case Name':<24} | {'Steps':<5} | {'BD':<5} | {'Time(s)':<8} | {'Sec/Samp':<9} | {'Unique':<8} | {'FID':<7}")
    print("-" * 85)

    results = []
    for c in unique_cases:
        rec, preview_imgs = run_single_experiment(
            model=model,
            scheduler=scheduler,
            sampler_name="ddim",
            steps=c["steps"],
            eta=c["eta"],
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            birth_death=c["bd"],
            bd_every_k=c["k"],
            bd_total_time=c["tau"],
            bd_bandwidth=0.3,
            fid_evaluator=fid_eval,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec["case_name"] = c["name"]
        results.append(rec)

        # Save preview grid
        img_path = samples_dir / f"{c['name']}.png"
        save_image((preview_imgs + 1) / 2, img_path, nrow=int(args.count ** 0.5) or 4)

        fid_str = f"{rec['fid']:.2f}" if rec['fid'] is not None else "N/A"
        bd_str = "YES" if rec["birth_death"] else "NO"
        print(f"{c['name']:<24} | {rec['steps']:<5} | {bd_str:<5} | {rec['elapsed_seconds']:<8.2f} | {rec['seconds_per_sample']:<9.4f} | {rec['unique_fraction']:<8.2f} | {fid_str:<7}")

    print("=" * 85)

    # 3. Export CSV Table
    csv_file = out_dir / "deliverable4_results.csv"
    with open(csv_file, "w", newline="") as f:
        fieldnames = ["case_name", "sampler", "steps", "eta", "birth_death", "bd_every_k",
                      "bd_total_time", "bd_bandwidth", "nfe", "elapsed_seconds",
                      "seconds_per_sample", "samples_per_second", "unique_fraction", "fid"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Export JSON
    json_file = out_dir / "deliverable4_metrics.json"
    with open(json_file, "w") as f:
        json.dump({"dataset": args.dataset, "device": str(device), "results": results}, f, indent=2)

    print(f"\n[+] Results successfully exported to:")
    print(f"    - CSV Table: {csv_file.resolve()}")
    print(f"    - JSON File: {json_file.resolve()}")

    # 4. Generate presentation charts
    generate_presentation_plots(results, out_dir)


if __name__ == "__main__":
    main()

