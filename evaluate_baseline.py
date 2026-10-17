"""
Deliverable 1 Evaluation Runner: Diffusion Sampling Baseline Benchmark.
Compares standard DDPM ancestral sampling with DDIM across different step budgets.
Measures wall-clock latency, throughput, NFEs, trajectory lengths, and fidelity metrics (SWD, MMD, Mode Coverage).
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch

from src.models.toy_mlp import ToyScoreMLP
from src.diffusion.ddpm import GaussianDiffusion
from src.samplers.ddpm_sampler import DDPMSampler
from src.samplers.ddim_sampler import DDIMSampler
from src.diffusion.birth_death_sampler import BirthDeathDiffusionSampler
from src.metrics.evaluator import SamplingBenchmarkEvaluator
from src.utils.datasets import Synthetic2DDataset
from src.utils.visualization import (
    plot_2d_distribution_comparison,
    plot_2d_trajectory,
    plot_speed_vs_quality,
)


def get_mode_centers(dataset_name: str) -> torch.Tensor:
    """Helper to return ground truth Gaussian mode centers for coverage evaluation."""
    if dataset_name == "8gaussians":
        scale = 4.0 / 2.0
        centers = [
            (scale * math.cos(2 * math.pi * i / 8), scale * math.sin(2 * math.pi * i / 8))
            for i in range(8)
        ]
        return torch.tensor(centers, dtype=torch.float32)
    elif dataset_name == "25gaussians":
        range_val = 4.0 / 2.0
        coords = np.linspace(-range_val, range_val, 5)
        centers = [[x, y] for x in coords for y in coords]
        return torch.tensor(centers, dtype=torch.float32)
    return None


def run_baseline_evaluation(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("=" * 80)
    print(" " * 20 + "DIFFUSION SAMPLING BASELINE EVALUATION (DELIVERABLE 1)")
    print("=" * 80)
    print(f"[*] Target Device: {device}")
    print(f"[*] Dataset: {args.dataset}")
    print(f"[*] Total Diffusion Steps (T): {args.timesteps}")
    print(f"[*] Evaluation Batch Size: {args.num_samples}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Ground truth test dataset
    gt_dataset = Synthetic2DDataset(name=args.dataset, num_samples=args.num_samples, seed=123)
    gt_data = gt_dataset.data
    mode_centers = get_mode_centers(args.dataset)

    evaluator = SamplingBenchmarkEvaluator(ground_truth_samples=gt_data, mode_centers=mode_centers)

    # 2. Initialize Model
    score_model = ToyScoreMLP(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_emb_dim=64
    ).to(device)

    diffusion = GaussianDiffusion(
        model=score_model,
        timesteps=args.timesteps,
        beta_schedule=args.schedule
    ).to(device)

    # Check for checkpoint or auto-train a fast model
    ckpt_path = os.path.join(args.checkpoint_dir, f"ddpm_{args.dataset}.pt")
    if os.path.exists(ckpt_path):
        print(f"[*] Loading pretrained model weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("ema_state", checkpoint.get("model_state"))
        score_model.load_state_dict(state_dict)
    else:
        print(f"[!] No checkpoint found at '{ckpt_path}'. Auto-training a fast baseline model ({args.auto_train_epochs} epochs)...")
        from train import train_2d
        train_args = argparse.Namespace(
            dataset=args.dataset,
            epochs=args.auto_train_epochs,
            batch_size=512,
            lr=2e-3,
            timesteps=args.timesteps,
            schedule=args.schedule,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_samples=30000,
            device=args.device,
            seed=42,
            log_interval=5,
            eval_interval=100,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir
        )
        train_2d(train_args)
        checkpoint = torch.load(ckpt_path, map_location=device)
        score_model.load_state_dict(checkpoint.get("ema_state", checkpoint.get("model_state")))

    score_model.eval()

    # 3. Configure benchmark sampler configurations
    sampler_configs = [
        {"type": "DDPM", "steps": args.timesteps, "label": f"DDPM-{args.timesteps} (Baseline)"},
        {"type": "DDIM", "steps": 50, "eta": 0.0, "label": "DDIM-50"},
        {"type": "BDLS", "steps": 50, "eta": 0.0, "label": "BDLS-Diff-50 (Proposed)"},
        {"type": "DDIM", "steps": 20, "eta": 0.0, "label": "DDIM-20"},
        {"type": "BDLS", "steps": 20, "eta": 0.0, "label": "BDLS-Diff-20 (Proposed)"},
        {"type": "DDIM", "steps": 10, "eta": 0.0, "label": "DDIM-10"},
    ]

    benchmark_results = []
    generated_dict = {}
    baseline_time = None
    trajectory_for_viz = None

    shape = (args.num_samples, 2)

    for cfg in sampler_configs:
        label = cfg["label"]
        print(f"\n[>] Running {label} ...")

        if cfg["type"] == "DDPM":
            sampler = DDPMSampler(diffusion)
            samples, info = sampler.sample(
                shape=shape,
                return_intermediates=True,
                verbose=True
            )
            baseline_time = info["inference_time"]
            trajectory_for_viz = info["intermediates"]

        elif cfg["type"] == "DDIM":
            sampler = DDIMSampler(diffusion)
            samples, info = sampler.sample(
                shape=shape,
                num_inference_steps=cfg["steps"],
                eta=cfg.get("eta", 0.0),
                return_intermediates=(trajectory_for_viz is None),
                verbose=True
            )
            if trajectory_for_viz is None:
                trajectory_for_viz = info["intermediates"]

        elif cfg["type"] == "BDLS":
            sampler = BirthDeathDiffusionSampler(
                diffusion,
                birth_rate=0.8,
                langevin_steps=1,
                langevin_lr=0.03,
                exploration_noise=0.03,
                diversity_weight=0.25
            )
            samples, info = sampler.sample(
                shape=shape,
                num_inference_steps=cfg["steps"],
                eta=cfg.get("eta", 0.0),
                return_intermediates=False,
                verbose=True
            )

        eval_summary = evaluator.evaluate_sampler(
            sampler_name=label,
            generated_samples=samples,
            info=info,
            baseline_time=baseline_time
        )
        benchmark_results.append(eval_summary)
        generated_dict[label] = samples.cpu()

    # 4. Display Results in Beautiful ASCII Table
    print("\n" + "=" * 90)
    print(" " * 28 + "BASELINE BENCHMARK RESULTS SUMMARY")
    print("=" * 90)
    
    header = f"{'Sampler':<24} | {'NFE':<6} | {'Time (s)':<9} | {'Speedup':<8} | {'SWD (x10^-2)':<13} | {'MMD (x10^-3)':<13} | {'Mode Cov %':<10}"
    print(header)
    print("-" * 90)

    for res in benchmark_results:
        swd_scaled = res["sliced_wasserstein"] * 100.0
        mmd_scaled = res["mmd"] * 1000.0
        mode_cov = res.get("mode_coverage_pct", "N/A")
        print(f"{res['sampler']:<24} | {res['nfe']:<6} | {res['inference_time_sec']:<9.3f} | {res['speedup_vs_baseline']:<7.2f}x | {swd_scaled:<13.3f} | {mmd_scaled:<13.3f} | {mode_cov}%")

    print("=" * 90)

    # 5. Save Report to JSON
    report_path = os.path.join(args.results_dir, "baseline_benchmark.json")
    with open(report_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"\n[+] Saved quantitative benchmark report to: {report_path}")

    # 6. Generate and Save Visualizations
    comp_path = os.path.join(args.results_dir, "baseline_comparison.png")
    plot_2d_distribution_comparison(
        ground_truth=gt_data,
        generated_dict=generated_dict,
        save_path=comp_path,
        title=f"Baseline Generation Quality Comparison ({args.dataset})"
    )
    print(f"[+] Saved distribution comparison plot to: {comp_path}")

    if trajectory_for_viz is not None and len(trajectory_for_viz) > 0:
        traj_path = os.path.join(args.results_dir, "trajectory_evolution.png")
        plot_2d_trajectory(trajectory_for_viz, save_path=traj_path)
        print(f"[+] Saved reverse trajectory evolution plot to: {traj_path}")

    pareto_path = os.path.join(args.results_dir, "pareto_frontier.png")
    plot_speed_vs_quality(benchmark_results, save_path=pareto_path)
    print(f"[+] Saved speed vs quality Pareto frontier plot to: {pareto_path}")
    print("\n[+] Deliverable 1 baseline evaluation completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Sampling Baseline (Deliverable 1)")
    parser.add_argument("--dataset", type=str, default="8gaussians", help="Dataset name: 8gaussians, 25gaussians, swissroll")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion timesteps T for baseline DDPM")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate for benchmarking")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--auto_train_epochs", type=int, default=20, help="Epochs to auto-train if no checkpoint exists")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    run_baseline_evaluation(args)


if __name__ == "__main__":
    main()

