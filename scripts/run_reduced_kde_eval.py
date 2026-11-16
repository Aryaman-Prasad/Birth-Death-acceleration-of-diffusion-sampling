"""Evaluation & Benchmark Suite for Pretrained-CNN Reduced-KDE Birth-Death Acceleration.

Compares:
1. Baseline DDIM (No Birth-Death)
2. High-Dimensional Pixel-KDE Birth-Death (dim=3072 for CIFAR-10 / 196608 for CelebA)
3. Reduced-KDE Birth-Death (dim=25 via Pretrained CNN + PCA)

Usage:
    python3 scripts/run_reduced_kde_eval.py --dataset cifar10 --kde_reduce 25 --device cuda
    python3 scripts/run_reduced_kde_eval.py --dataset imagefolder --data-dir data/celeba --kde_reduce 25
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

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.acceleration.reduced_kde_birth_death import (
    ReducedKDEBirthDeathAcceleration,
)
from diffusion.datasets import make_data_module
from diffusion.metrics import FIDEvaluator, measure_sampling
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import EmpiricalNoisedDataTarget


def _unique_fraction(samples: torch.Tensor) -> float:
    flat = samples.reshape(samples.shape[0], -1)
    unique_count = torch.unique(flat, dim=0).shape[0]
    return float(unique_count / samples.shape[0])


def run_experiment_case(
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
    bd_mode: str = "none",  # "none", "pixel_kde", or "reduced_kde"
    kde_reduce: int = 25,
    cnn_backbone: str = "auto",
    projection_method: str = "pca",
    noise_references: bool = True,
    bd_every_k: int = 2,
    bd_total_time: float = 0.01,
    bd_bandwidth: float = 0.3,
    real_images: torch.Tensor | None = None,
    fid_samples: int = 256,
    fid_batch_size: int = 32,
):
    """Executes a single sampling case with latency, NFE, diversity, and FID."""
    sampler = make_sampler(sampler_name, eta=eta)
    inference_steps = None if sampler_name == "ddpm" else steps
    sampler_steps = len(
        sampler.inference_timesteps(scheduler, inference_steps, device)
    )

    modifiers = []
    if bd_mode == "pixel_kde" and target is not None:
        modifiers.append(
            BirthDeathAcceleration(
                target,
                bd_bandwidth,
                every_k_steps=bd_every_k,
                total_time=bd_total_time,
            )
        )
    elif bd_mode == "reduced_kde" and target is not None:
        modifiers.append(
            ReducedKDEBirthDeathAcceleration(
                target,
                bd_bandwidth,
                kde_reduce=kde_reduce,
                every_k_steps=bd_every_k,
                total_time=bd_total_time,
                backbone=cnn_backbone,
                projection_method=projection_method,
                noise_references=noise_references,
                seed=seed,
            )
        )

    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)

    # 1. Preview generation (diversity measurement)
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

    # 2. FID evaluation
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
            fake_batch = pipeline.sample(
                model, b_shape, device, inference_steps, fid_gen
            )
            fid_evaluator.update_fake(fake_batch)
            remaining -= b_count
        fid_score = float(fid_evaluator.compute())

    record = {
        "sampler": sampler_name,
        "steps": steps,
        "eta": eta,
        "bd_mode": bd_mode,
        "kde_reduce": kde_reduce if bd_mode == "reduced_kde" else None,
        "bd_every_k": bd_every_k if bd_mode != "none" else None,
        "bd_total_time": bd_total_time if bd_mode != "none" else None,
        "bd_bandwidth": bd_bandwidth if bd_mode != "none" else None,
        "nfe": sampling_metrics.nfe,
        "elapsed_seconds": round(sampling_metrics.elapsed_seconds, 4),
        "seconds_per_sample": round(sampling_metrics.seconds_per_sample, 4),
        "samples_per_second": round(sampling_metrics.samples_per_second, 2),
        "unique_fraction": round(unique_ratio, 4),
        "fid": round(fid_score, 3) if fid_score is not None else None,
    }
    return record, preview_samples


def main():
    parser = argparse.ArgumentParser(
        description="Deliverable 4 Reduced-KDE Benchmark Suite"
    )
    parser.add_argument(
        "--dataset", choices=("cifar10", "imagefolder"), default="cifar10"
    )
    parser.add_argument(
        "--data-dir", default="data", help="Dataset directory"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download dataset if missing"
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--reference-count",
        type=int,
        default=256,
        help="Number of real images used for empirical target density",
    )
    parser.add_argument(
        "--fid-samples",
        type=int,
        default=256,
        help="Number of real & generated images for FID evaluation",
    )
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument(
        "--count", type=int, default=16, help="Preview batch size"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="reduced_kde_deliverable4_results"
    )

    # Dimensionality Reduction Flags
    parser.add_argument(
        "--kde_reduce",
        "--kde-reduce",
        type=int,
        dest="kde_reduce",
        default=25,
        help="Compressed feature dimension for Reduced-KDE (default: 25)",
    )
    parser.add_argument(
        "--cnn-backbone",
        choices=("auto", "vgg16", "resnet18", "mobilenet", "convnet"),
        default="auto",
        help="Pretrained CNN backbone: auto, vgg16, resnet18, mobilenet, convnet (default: auto)",
    )
    parser.add_argument(
        "--projection-method",
        choices=("pca", "orthogonal", "linear", "mlp"),
        default="pca",
        help="compression method: pca, orthogonal, linear (nn.Linear), or mlp (nn.Sequential)",
    )
    parser.add_argument(
        "--noise-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to re-noise references at each step (pass --no-noise-references for 10x faster caching)",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if args.device in ("auto", "cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"[+] Initializing Reduced-KDE benchmark suite on {device}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset & Reference Target
    data_module = make_data_module(
        args.dataset,
        args.data_dir,
        args.image_size,
        args.download,
        train=True,
    )
    sample_shape = tuple(data_module.sample_shape)
    dataset = data_module.dataset()
    ref_count = min(args.reference_count, len(dataset))
    print(
        f"[+] Building empirical target density over {ref_count} reference images..."
    )
    target = EmpiricalNoisedDataTarget.from_dataset(
        dataset,
        num_reference_samples=ref_count,
        seed=args.seed,
        reference_chunk_size=64,
    )

    # 2. Scheduler and Target-Score Model
    scheduler = make_scheduler("linear", 1000, 1e-4, 2e-2).to(device)
    model = make_model("target-score", target=target, scheduler=scheduler).to(
        device
    )
    model.eval()

    # 3. Reference images for FID
    real_images = None
    if args.fid_samples > 0:
        from torch.utils.data import DataLoader

        loader = DataLoader(
            dataset, batch_size=args.fid_batch_size, shuffle=True
        )
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

    # 4. Three-way comparison cases:
    # A) Baseline DDIM
    # B) Standard High-Dim Pixel-KDE (dim 3072/196608)
    # C) Reduced-KDE (dim 25 via Pretrained CNN)
    cases = [
        # Baseline
        {
            "name": "ddim_25_baseline",
            "steps": 25,
            "mode": "none",
            "k": 1,
            "tau": 0.0,
            "eta": 1.0,
        },
        # High-Dim Pixel KDE BD (old)
        {
            "name": "ddim_25_bd_pixel_highdim",
            "steps": 25,
            "mode": "pixel_kde",
            "k": 2,
            "tau": 0.01,
            "eta": 1.0,
        },
        # Reduced-KDE BD (new dim=25)
        {
            "name": f"ddim_25_bd_reduced_kde_dim{args.kde_reduce}",
            "steps": 25,
            "mode": "reduced_kde",
            "k": 2,
            "tau": 0.01,
            "eta": 1.0,
        },
        # 50-step comparison
        {
            "name": "ddim_50_baseline",
            "steps": 50,
            "mode": "none",
            "k": 1,
            "tau": 0.0,
            "eta": 1.0,
        },
        {
            "name": "ddim_50_bd_pixel_highdim",
            "steps": 50,
            "mode": "pixel_kde",
            "k": 2,
            "tau": 0.01,
            "eta": 1.0,
        },
        {
            "name": f"ddim_50_bd_reduced_kde_dim{args.kde_reduce}",
            "steps": 50,
            "mode": "reduced_kde",
            "k": 2,
            "tau": 0.01,
            "eta": 1.0,
        },
    ]

    results = []
    print("\n" + "=" * 80)
    print(f"RUNNING REDUCED-KDE BENCHMARK SWEEP (--kde_reduce={args.kde_reduce})")
    print("=" * 80)

    for case in cases:
        c_name = case["name"]
        print(f"\n[Case: {c_name}] Running {case['mode']} at {case['steps']} steps...")
        rec, preview = run_experiment_case(
            model=model,
            scheduler=scheduler,
            sampler_name="ddim",
            steps=case["steps"],
            eta=case["eta"],
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            bd_mode=case["mode"],
            kde_reduce=args.kde_reduce,
            cnn_backbone=args.cnn_backbone,
            projection_method=args.projection_method,
            noise_references=args.noise_references,
            bd_every_k=case["k"],
            bd_total_time=case["tau"],
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec["case_name"] = c_name
        results.append(rec)

        # Save preview grid
        grid_norm = (preview.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_norm, samples_dir / f"{c_name}.png", nrow=4)
        print(
            f"   -> NFE: {rec['nfe']} | Time: {rec['elapsed_seconds']}s | "
            f"Unique: {rec['unique_fraction']} | FID: {rec['fid']}"
        )

    # Save CSV
    csv_file = out_dir / "reduced_kde_benchmark_results.csv"
    keys = list(results[0].keys())
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # Save JSON
    json_file = out_dir / "reduced_kde_benchmark_metrics.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # 5. Generate Matplotlib Presentation Plots
    try:
        import matplotlib.pyplot as plt

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: FID Comparison across Steps
        plt.figure(figsize=(7.5, 4.8), dpi=150)
        for mode, label, color, style in [
            ("none", "Baseline DDIM (No BD)", "#1f77b4", "o-"),
            ("pixel_kde", "High-Dim Pixel-KDE BD", "#d62728", "s--"),
            ("reduced_kde", f"Reduced-KDE BD (dim={args.kde_reduce})", "#2ca02c", "^-"),
        ]:
            pts = [r for r in results if r["bd_mode"] == mode and r.get("fid") is not None]
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
        plt.title(
            f"Sample Quality: Baseline vs. Pixel-KDE vs. Reduced-KDE ({args.dataset.upper()})",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig1_steps_vs_fid.png")
        plt.close()

        # Plot 2: Unique Particle Fraction (Diversity)
        plt.figure(figsize=(7.5, 4.5), dpi=150)
        modes = ["Baseline", "Pixel-KDE", f"Reduced-KDE (d={args.kde_reduce})"]
        uniques = []
        for mode in ("none", "pixel_kde", "reduced_kde"):
            matching = [
                r["unique_fraction"]
                for r in results
                if r["bd_mode"] == mode and r["steps"] == 25
            ]
            uniques.append(matching[0] if matching else 1.0)

        plt.bar(modes, uniques, color=["#1f77b4", "#d62728", "#2ca02c"], width=0.45)
        plt.axhline(1.0, linestyle=":", color="gray", label="Ideal Diversity (1.0)")
        plt.ylabel("Unique Particle Fraction", fontsize=11)
        plt.title(
            "Population Diversity (Unique Fraction at 25 Steps)",
            fontsize=12,
            fontweight="bold",
        )
        plt.ylim(0, 1.15)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig2_diversity_comparison.png")
        plt.close()

        print(f"[+] Saved presentation plots to: {plots_dir}/")
    except Exception as e:
        print(f"[!] Plot generation note: {e}")

    print("\n" + "=" * 80)
    print(f"[+] Benchmark Complete! Results saved to:")
    print(f"    - CSV:  {csv_file}")
    print(f"    - JSON: {json_file}")
    print(f"    - Grids: {samples_dir}/")
    if (out_dir / "plots").exists():
        print(f"    - Plots: {out_dir / 'plots'}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
