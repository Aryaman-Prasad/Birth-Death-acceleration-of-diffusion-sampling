"""Standalone Evaluation Suite for Particle Filtering Solution Space Explorer.

Compares:
1. Baseline DDIM (No Birth-Death)
2. Particle Filtering Solution Space Explorer (Branches N -> N * B candidate paths, scores drift potential, resamples elite N particles)

Uses pretrained google/ddpm-cifar10-32 by default.

Usage:
    python3 scripts/run_particle_filter_eval.py --steps 10 25 50 --fid-samples 512 --branch-factor 2 --device cuda
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

from diffusion.acceleration import ParticleFilteringSolutionSpaceExplorer
from diffusion.datasets import make_data_module
from diffusion.metrics import FIDEvaluator, measure_sampling, rbf_mmd_squared
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler


def _unique_fraction(samples: torch.Tensor, threshold: float = 0.05) -> float:
    """Calculates the fraction of perceptually unique images in a batch.

    Images are normalized to [0, 1]. Two images whose Mean Absolute Error (MAE)
    is below `threshold` (default 0.05, i.e. 5% average pixel difference) are
    detected as duplicate/repeating modes from the same parent particle.
    """
    N = samples.shape[0]
    if N <= 1:
        return 1.0

    s = samples.detach().float()
    s_min = (
        s.amin(dim=(1, 2, 3), keepdim=True)
        if s.ndim == 4
        else s.amin(dim=1, keepdim=True)
    )
    s_max = (
        s.amax(dim=(1, 2, 3), keepdim=True)
        if s.ndim == 4
        else s.amax(dim=1, keepdim=True)
    )
    denom = (s_max - s_min).clamp_min(1e-6)
    normed = (s - s_min) / denom
    flat = normed.reshape(N, -1)

    # Pairwise Mean Absolute Error (MAE) across all pixels
    mae_matrix = torch.cdist(flat, flat, p=1) / flat.shape[1]

    unique_mask = torch.ones(N, dtype=torch.bool, device=samples.device)
    for i in range(N):
        if not unique_mask[i]:
            continue
        for j in range(i + 1, N):
            if unique_mask[j] and mae_matrix[i, j] < threshold:
                unique_mask[j] = False

    unique_count = int(unique_mask.sum().item())
    return float(unique_count / N)


def run_pf_case(
    *,
    model,
    scheduler,
    steps: int,
    eta: float,
    sample_shape: tuple[int, ...],
    count: int,
    device: torch.device,
    seed: int,
    use_pf: bool = False,
    branch_factor: int = 2,
    every_k: int = 2,
    temperature: float = 1.0,
    diversity_weight: float = 0.1,
    min_timestep: int | None = 100,
    max_event_probability: float = 0.25,
    real_images: torch.Tensor | None = None,
    fid_samples: int = 0,
    fid_batch_size: int = 32,
):
    """Executes a single sampling case with latency, NFE, diversity, MMD, and FID."""
    sampler = make_sampler("ddim", eta=eta)
    sampler_steps = len(sampler.inference_timesteps(scheduler, steps, device))

    modifiers = []
    if use_pf:
        modifiers.append(
            ParticleFilteringSolutionSpaceExplorer(
                branch_factor=branch_factor,
                every_k_steps=every_k,
                temperature=temperature,
                diversity_weight=diversity_weight,
                min_timestep=min_timestep,
                max_event_probability=max_event_probability,
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

    # 2. Quality Evaluation: MMD against real images
    mmd_score = None
    if real_images is not None:
        try:
            n_refs = min(512, real_images.shape[0])
            ref_subset = real_images[:n_refs].to(device=device, dtype=torch.float32)
            eval_samples = preview_samples.to(device=device, dtype=torch.float32)
            mmd_metrics, _ = rbf_mmd_squared(eval_samples, ref_subset)
            mmd_score = float(mmd_metrics["empirical_target_mmd2_rbf_multiscale"])
        except Exception:
            mmd_score = None

    # 3. FID evaluation
    fid_score = None
    if real_images is not None and fid_samples > 0:
        try:
            n_fid = min(fid_samples, real_images.shape[0])
            fid_evaluator = FIDEvaluator(device)
            for start in range(0, n_fid, fid_batch_size):
                chunk = real_images[start : min(start + fid_batch_size, n_fid)].to(device)
                fid_evaluator.update_real(chunk)

            remaining = n_fid
            fid_gen = torch.Generator(device=device).manual_seed(seed + 999)
            while remaining > 0:
                b_count = min(fid_batch_size, remaining)
                b_shape = (b_count, *sample_shape)
                fake_batch = pipeline.sample(model, b_shape, device, steps, fid_gen)
                fid_evaluator.update_fake(fake_batch)
                remaining -= b_count
            fid_score = float(fid_evaluator.compute())
        except Exception as e:
            print(f"[!] Warning: FID evaluation encountered an error ({e})")
            fid_score = None

    record = {
        "sampler": "ddim",
        "steps": steps,
        "method": "particle_filter" if use_pf else "baseline",
        "branch_factor": branch_factor if use_pf else None,
        "every_k": every_k if use_pf else None,
        "diversity_weight": diversity_weight if use_pf else None,
        "nfe": sampling_metrics.nfe,
        "elapsed_seconds": round(sampling_metrics.elapsed_seconds, 4),
        "samples_per_second": round(sampling_metrics.samples_per_second, 2),
        "unique_fraction": round(unique_ratio, 4),
        "mmd": round(mmd_score, 6) if mmd_score is not None else None,
        "fid": round(fid_score, 3) if fid_score is not None else None,
    }
    return record, preview_samples


def main():
    parser = argparse.ArgumentParser(description="Standalone Particle Filtering Solution Explorer Evaluation")
    parser.add_argument(
        "--model-id",
        "--pretrained-model-id",
        dest="model_id",
        default="google/ddpm-cifar10-32",
        help="Hugging Face pretrained model ID (default: google/ddpm-cifar10-32)",
    )
    parser.add_argument("--dataset", choices=("cifar10", "imagefolder"), default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download dataset if missing (default: True)",
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument(
        "--fid-samples",
        type=int,
        default=0,
        help="Number of real & generated images for FID evaluation (default: 0 = disabled)",
    )
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 25, 50])
    parser.add_argument("--branch-factor", type=int, default=2)
    parser.add_argument("--filter-every-k", type=int, default=2)
    parser.add_argument("--filter-temp", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.1)
    parser.add_argument(
        "--bd-min-timestep",
        type=int,
        default=100,
        help="Minimum diffusion timestep for Birth-Death intervention (default: 100)",
    )
    parser.add_argument(
        "--bd-max-event-prob",
        type=float,
        default=0.25,
        help="Maximum fraction of population that can undergo BD per step (default: 0.25)",
    )
    parser.add_argument("--output-dir", default="particle_filter_benchmark_results")

    args = parser.parse_args()
    if getattr(args, "no_fid", False):
        args.fid_samples = 0

    device = torch.device(
        "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"[+] Initializing Particle Filtering Evaluation on {device}...")
    print(f"[+] Loading Pretrained Model: {args.model_id}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained Diffusers Model and Scheduler
    try:
        model = make_model("huggingface-unet", pretrained_model_id=args.model_id).to(device)
        scheduler = make_scheduler("huggingface", pretrained_model_id=args.model_id).to(device)
        sample_shape = tuple(model.sample_shape)
        print(f"[✓] Model loaded successfully: {args.model_id} (sample shape: {sample_shape})")
    except Exception as e:
        print(f"[!] Warning: Could not load Hugging Face model ({e}). Using linear scheduler fallback.")
        scheduler = make_scheduler("linear", 1000, 1e-4, 2e-2).to(device)
        sample_shape = (3, args.image_size, args.image_size)
        from diffusion.models.toy_mlp import ToyMLP
        model = ToyMLP(sample_shape, 128).to(device)

    model.eval()

    # 2. Real Images for FID / MMD Evaluation (only loaded if FID requested)
    real_images = None
    if args.fid_samples > 0:
        print(f"[+] Loading {args.fid_samples} real {args.dataset.upper()} images for FID evaluation...")
        download_dataset = True if args.dataset == "cifar10" else getattr(args, "download", True)
        data_module = make_data_module(
            args.dataset, args.data_dir, args.image_size, download_dataset, train=True
        )
        dataset = data_module.dataset()
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

    # 3. Run Benchmark Cases
    results = []
    for st in args.steps:
        # A) Baseline DDIM
        case_name_base = f"ddim_{st}_baseline"
        print(f"\n[Case: {case_name_base}] Running Baseline DDIM at {st} steps...")
        rec_base, prev_base = run_pf_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            use_pf=False,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_base["case_name"] = case_name_base
        results.append(rec_base)
        grid_base = (prev_base.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_base, samples_dir / f"{case_name_base}.png", nrow=4)
        fid_disp_b = f" | FID: {rec_base['fid']}" if rec_base.get("fid") is not None else ""
        mmd_disp_b = f" | MMD: {rec_base['mmd']}" if rec_base.get("mmd") is not None else ""
        print(f"   -> Samples/s: {rec_base['samples_per_second']} | Unique: {rec_base['unique_fraction']:.4f} | NFE: {rec_base['nfe']}{mmd_disp_b}{fid_disp_b}")

        # B) Particle Filtering Explorer
        case_name_pf = f"ddim_{st}_particle_filter"
        print(f"\n[Case: {case_name_pf}] Running Particle Filtering Explorer (B={args.branch_factor}) at {st} steps...")
        rec_pf, prev_pf = run_pf_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            use_pf=True,
            branch_factor=args.branch_factor,
            every_k=args.filter_every_k,
            temperature=args.filter_temp,
            diversity_weight=args.diversity_weight,
            min_timestep=args.bd_min_timestep,
            max_event_probability=args.bd_max_event_prob,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_pf["case_name"] = case_name_pf
        results.append(rec_pf)
        grid_pf = (prev_pf.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_pf, samples_dir / f"{case_name_pf}.png", nrow=4)
        fid_disp_p = f" | FID: {rec_pf['fid']}" if rec_pf.get("fid") is not None else ""
        mmd_disp_p = f" | MMD: {rec_pf['mmd']}" if rec_pf.get("mmd") is not None else ""
        print(f"   -> Samples/s: {rec_pf['samples_per_second']} | Unique: {rec_pf['unique_fraction']:.4f} | NFE: {rec_pf['nfe']}{mmd_disp_p}{fid_disp_p}")

    # Save CSV and JSON
    csv_file = out_dir / "particle_filter_benchmark_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_file = out_dir / "particle_filter_benchmark_metrics.json"
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
            ("particle_filter", f"Particle Filter (B={args.branch_factor})", "#e377c2", "p-"),
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
        plt.title(f"Particle Filtering Solution Quality ({args.dataset.upper()})", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig1_pf_steps_vs_fid.png")
        plt.close()

        # Plot 2: Diversity comparison
        plt.figure(figsize=(7, 4.5), dpi=150)
        st_ref = 25 if 25 in args.steps else args.steps[0]
        pts_sp = [r for r in results if r["steps"] == st_ref]
        plt.bar(
            [p["method"] for p in pts_sp],
            [p["unique_fraction"] for p in pts_sp],
            color=["#1f77b4", "#e377c2"],
            width=0.4,
        )
        plt.axhline(1.0, linestyle=":", color="gray", label="Ideal Diversity (1.0)")
        plt.ylabel("Unique Particle Fraction", fontsize=11)
        plt.title(f"Diversity at {st_ref} Steps", fontsize=12, fontweight="bold")
        plt.ylim(0, 1.15)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig2_pf_diversity.png")
        plt.close()
    except Exception as e:
        print(f"[!] Plot note: {e}")

    print("\n" + "=" * 80)
    print(f"[+] Particle Filtering Benchmark Complete! Results saved to:")
    print(f"    - CSV:   {csv_file}")
    print(f"    - JSON:  {json_file}")
    print(f"    - Grids: {samples_dir}/")
    print("=" * 80)

    try:
        import pandas as pd
        df = pd.DataFrame(results)
        cols = [
            c
            for c in [
                "case_name",
                "steps",
                "method",
                "unique_fraction",
                "samples_per_second",
                "elapsed_seconds",
                "nfe",
                "mmd",
                "fid",
            ]
            if c in df.columns and (c != "fid" or args.fid_samples > 0)
        ]
        print("\n[+] Benchmark Summary Table:")
        print(df[cols].to_string(index=False))
        print("=" * 80)
    except Exception:
        pass


if __name__ == "__main__":
    main()
