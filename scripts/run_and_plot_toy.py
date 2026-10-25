"""Run and plot toy 2D Gaussian mixture experiments (Baseline vs Birth-Death).

This script compares diffusion sampling with and without Birth-Death on an
analytic 8-mode 2D Gaussian mixture ring target, and generates publication-grade
visualizations:
1. Scatter plots of generated 2D particles overlaid on true target density contours.
2. Mode occupancy and duplicate particle clustering (visually demonstrating collapse).
3. Quantitative comparison bars (MMD^2, NLL, Mode Coverage, Unique Ratio).

Usage:
    python3 scripts/run_and_plot_toy.py --count 512 --device auto --output-dir toy_results
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

import toy_bd_experiment as tbe
from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.metrics import measure_sampling
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import ToyGaussianMixtureTarget


def _unique_fraction(samples: torch.Tensor) -> float:
    flat = samples.reshape(samples.shape[0], -1)
    return float(torch.unique(flat, dim=0).shape[0] / samples.shape[0])


def run_toy_case(
    *,
    model,
    target,
    references,
    scheduler,
    sampler_name: str,
    steps: int,
    eta: float,
    count: int,
    birth_death: bool,
    bd_every_k: int,
    bd_total_time: float,
    bd_bandwidth: float,
    device: torch.device,
    seed: int,
):
    sampler = make_sampler(sampler_name, eta=eta)
    inference_steps = None if sampler_name == "ddpm" else steps
    sampler_steps = len(sampler.inference_timesteps(scheduler, inference_steps, device))

    modifiers = []
    if birth_death:
        modifiers.append(
            BirthDeathAcceleration(
                target,
                bd_bandwidth,
                every_k_steps=bd_every_k,
                total_time=bd_total_time,
            )
        )

    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (count, 2)

    samples, metrics = measure_sampling(
        model,
        device,
        count,
        sampler_steps,
        lambda: pipeline.sample(model, shape, device, inference_steps, gen),
    )

    quality = tbe._toy_quality(samples.detach(), target, references)
    unique_ratio = _unique_fraction(samples)

    record = {
        "sampler": sampler_name,
        "steps": steps,
        "eta": eta,
        "birth_death": birth_death,
        "bd_every_k": bd_every_k if birth_death else None,
        "bd_total_time": bd_total_time if birth_death else None,
        "bd_bandwidth": bd_bandwidth if birth_death else None,
        "nfe": metrics.nfe,
        "elapsed_seconds": round(metrics.elapsed_seconds, 4),
        "clean_target_nll": round(quality["clean_target_nll"], 4),
        "mmd2": round(quality["clean_target_mmd2_rbf_multiscale"], 5),
        "mode_coverage": round(quality["mode_coverage"], 3),
        "mode_balance_tv": round(quality["mode_balance_tv"], 4),
        "unique_fraction": round(unique_ratio, 4),
        "mode_counts": quality["mode_counts"],
    }
    return record, samples.detach().cpu().numpy()


def plot_toy_comparison(cases_data, target, output_path: Path):
    """Generates 2D scatter plots of particles overlaid on target density contours."""
    fig, axes = plt.subplots(1, len(cases_data), figsize=(4.5 * len(cases_data), 4.5), dpi=180)
    if len(cases_data) == 1:
        axes = [axes]

    # Precompute target density grid for background contour
    grid_range = 3.2
    grid_res = 150
    x = np.linspace(-grid_range, grid_range, grid_res)
    y = np.linspace(-grid_range, grid_range, grid_res)
    xx, yy = np.meshgrid(x, y)
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)

    with torch.no_grad():
        target_cpu = target.to("cpu")
        log_prob = target_cpu.log_prob_at_noise(pts, sigma=0.0)
        density = torch.exp(log_prob).reshape(grid_res, grid_res).numpy()

    centers = target_cpu.centers.numpy()

    for ax, (title, record, samples) in zip(axes, cases_data):
        # 1. Background target contour
        ax.contour(xx, yy, density, levels=7, cmap="Greys", alpha=0.45, linewidths=1.2)
        # 2. Mode centers
        ax.scatter(centers[:, 0], centers[:, 1], color="red", marker="+", s=70, linewidths=2, label="True Mode Centers", zorder=3)
        # 3. Generated particles
        ax.scatter(samples[:, 0], samples[:, 1], color="#1f77b4", alpha=0.6, s=18, edgecolors="none", label="Generated Samples", zorder=2)

        ax.set_xlim(-grid_range, grid_range)
        ax.set_ylim(-grid_range, grid_range)
        ax.set_aspect("equal", "box")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.4)

        # Annotations box
        info_text = (
            f"Steps: {record['steps']} | BD: {'Yes' if record['birth_death'] else 'No'}\n"
            f"MMD²: {record['mmd2']:.4f}\n"
            f"Coverage: {int(record['mode_coverage']*8)}/8 modes\n"
            f"Unique: {record['unique_fraction']*100:.1f}%"
        )
        ax.text(
            0.04, 0.04, info_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#cccccc")
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved 2D scatter comparison to: {output_path.resolve()}")


def plot_toy_metrics_bar(results: list[dict], output_path: Path):
    """Bar chart comparing MMD^2 and Unique Particle Ratio."""
    labels = [r["name"] for r in results]
    mmds = [r["mmd2"] for r in results]
    uniques = [r["unique_fraction"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=160)

    # Left axis: MMD^2 (lower is better)
    color1 = "#1f77b4"
    ax1.set_xlabel("Experiment Configuration", fontweight="bold", labelpad=10)
    ax1.set_ylabel("MMD² (Lower is Better)", color=color1, fontweight="bold")
    bars1 = ax1.bar(x - width/2, mmds, width, label="MMD²", color=color1, alpha=0.85)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5)

    # Right axis: Unique Fraction (higher is better)
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Unique Particle Fraction (Higher is Better)", color=color2, fontweight="bold")
    bars2 = ax2.bar(x + width/2, uniques, width, label="Unique Fraction", color=color2, alpha=0.85)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    plt.title("Toy Mixture: Quality (MMD²) vs. Particle Diversity (Unique %)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved metrics bar chart to: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Run and plot toy Gaussian mixture BD comparisons")
    parser.add_argument("--count", type=int, default=512, help="Number of particles to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="toy_results")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    print(f"[+] Running Toy 2D Gaussian Mixture benchmark on {device}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup 8-mode target and scheduler
    target = ToyGaussianMixtureTarget(radius=2.0, standard_deviation=0.15).to(device)
    scheduler = make_scheduler("linear", 1000, 1e-4, 2e-2).to(device)
    model = make_model("toy-mixture-oracle", target=target, alpha_bars=scheduler.alpha_bars).to(device)
    model.eval()

    ref_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    references = tbe._sample_target(target, max(args.count, 1024), ref_gen)

    # 4 Decisive comparison cases
    configs = [
        {"name": "DDIM 10 (Baseline)",    "steps": 10, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
        {"name": "DDIM 10 + Light BD",   "steps": 10, "bd": True,  "k": 2, "tau": 0.1, "eta": 0.0},
        {"name": "DDIM 10 + Heavy BD",   "steps": 10, "bd": True,  "k": 1, "tau": 1.0, "eta": 0.0},
        {"name": "DDIM 50 (Baseline)",    "steps": 50, "bd": False, "k": 1, "tau": 0.0, "eta": 0.0},
    ]

    print("\n" + "=" * 80)
    print(f"{'Configuration':<22} | {'Steps':<5} | {'BD':<5} | {'MMD²':<8} | {'Cov':<5} | {'Unique':<8} | {'Time(s)':<7}")
    print("-" * 80)

    cases_plot_data = []
    results = []

    for c in configs:
        rec, samples = run_toy_case(
            model=model,
            target=target,
            references=references,
            scheduler=scheduler,
            sampler_name="ddim",
            steps=c["steps"],
            eta=c["eta"],
            count=args.count,
            birth_death=c["bd"],
            bd_every_k=c["k"],
            bd_total_time=c["tau"],
            bd_bandwidth=0.3,
            device=device,
            seed=args.seed,
        )
        rec["name"] = c["name"]
        results.append(rec)
        cases_plot_data.append((c["name"], rec, samples))

        bd_label = "YES" if rec["birth_death"] else "NO"
        cov_label = f"{int(rec['mode_coverage']*8)}/8"
        print(f"{c['name']:<22} | {rec['steps']:<5} | {bd_label:<5} | {rec['mmd2']:<8.5f} | {cov_label:<5} | {rec['unique_fraction']:<8.2f} | {rec['elapsed_seconds']:<7.3f}")

    print("=" * 80)

    # Export CSV & JSON
    csv_path = out_dir / "toy_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = out_dir / "toy_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Plots
    scatter_fig = out_dir / "toy_scatter_comparison.png"
    plot_toy_comparison(cases_plot_data, target, scatter_fig)

    barchart_fig = out_dir / "toy_metrics_barchart.png"
    plot_toy_metrics_bar(results, barchart_fig)

    print(f"\n[+] All toy results & figures generated in: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()

