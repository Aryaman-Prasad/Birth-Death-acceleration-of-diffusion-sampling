"""Standalone Evaluation Suite for Exact Flow / Continuous Normalizing Flow (CNF) Birth-Death.

Compares:
1. Baseline DDIM (No Birth-Death)
2. Exact Flow Birth-Death (CNF divergence / exact marginal log-density vs population density)

Usage:
    python3 scripts/run_exact_flow_eval.py --dataset cifar10 --download --device cuda --steps 10 25 50
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

from diffusion.acceleration import FlowBirthDeathAcceleration
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


def run_flow_case(
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
    use_flow_bd: bool = False,
    every_k: int = 2,
    total_time: float = 0.01,
    bandwidth: float = 0.3,
    real_images: torch.Tensor | None = None,
    fid_samples: int = 256,
    fid_batch_size: int = 32,
):
    """Executes a single sampling case with latency, NFE, diversity, MMD, and FID."""
    sampler = make_sampler("ddim", eta=eta)
    sampler_steps = len(sampler.inference_timesteps(scheduler, steps, device))

    modifiers = []
    if use_flow_bd and target is not None:
        modifiers.append(
            FlowBirthDeathAcceleration(
                flow_target=target,
                every_k_steps=every_k,
                total_time=total_time,
                bandwidth=bandwidth,
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
        "method": "flow_exact" if use_flow_bd else "baseline",
        "every_k": every_k if use_flow_bd else None,
        "total_time": total_time if use_flow_bd else None,
        "bandwidth": bandwidth if use_flow_bd else None,
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
    parser = argparse.ArgumentParser(description="Standalone Exact Flow (CNF) Birth-Death Evaluation")
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
    parser.add_argument("--flow-every-k", type=int, default=2)
    parser.add_argument("--flow-total-time", type=float, default=0.01)
    parser.add_argument("--flow-bandwidth", type=float, default=0.3)
    parser.add_argument("--output-dir", default="exact_flow_benchmark_results")

    args = parser.parse_args()
    if getattr(args, "no_fid", False):
        args.fid_samples = 0

    device = torch.device(
        "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"[+] Initializing Exact Flow (CNF) Evaluation Suite on {device}...")

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
        rec_base, prev_base = run_flow_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            use_flow_bd=False,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_base["case_name"] = case_name_base
        results.append(rec_base)
        grid_base = (prev_base.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_base, samples_dir / f"{case_name_base}.png", nrow=4)
        print(f"   -> Samples/s: {rec_base['samples_per_second']} | Unique: {rec_base['unique_fraction']} | MMD: {rec_base.get('mmd')} | FID: {rec_base.get('fid')}")

        # B) Exact Flow Birth-Death
        case_name_flow = f"ddim_{st}_flow_exact"
        print(f"\n[Case: {case_name_flow}] Running Exact Flow Birth-Death at {st} steps...")
        rec_flow, prev_flow = run_flow_case(
            model=model,
            scheduler=scheduler,
            steps=st,
            eta=1.0,
            sample_shape=sample_shape,
            count=args.count,
            device=device,
            seed=args.seed,
            target=target,
            use_flow_bd=True,
            every_k=args.flow_every_k,
            total_time=args.flow_total_time,
            bandwidth=args.flow_bandwidth,
            real_images=real_images,
            fid_samples=args.fid_samples,
            fid_batch_size=args.fid_batch_size,
        )
        rec_flow["case_name"] = case_name_flow
        results.append(rec_flow)
        grid_flow = (prev_flow.clamp(-1.0, 1.0) + 1.0) * 0.5
        save_image(grid_flow, samples_dir / f"{case_name_flow}.png", nrow=4)
        print(f"   -> Samples/s: {rec_flow['samples_per_second']} | Unique: {rec_flow['unique_fraction']} | MMD: {rec_flow.get('mmd')} | FID: {rec_flow.get('fid')}")

    # Save CSV and JSON
    csv_file = out_dir / "exact_flow_benchmark_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_file = out_dir / "exact_flow_benchmark_metrics.json"
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
            ("flow_exact", "Exact Flow Birth-Death (CNF)", "#8c564b", "*--"),
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
        plt.title(f"Exact Flow (CNF) Quality ({args.dataset.upper()})", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plots_dir / "fig1_flow_steps_vs_fid.png")
        plt.close()

        # Plot 2: Diversity comparison
        plt.figure(figsize=(7, 4.5), dpi=150)
        st_ref = 25 if 25 in args.steps else args.steps[0]
        pts_sp = [r for r in results if r["steps"] == st_ref]
        plt.bar(
            [p["method"] for p in pts_sp],
            [p["unique_fraction"] for p in pts_sp],
            color=["#1f77b4", "#8c564b"],
            width=0.4,
        )
        plt.axhline(1.0, linestyle=":", color="gray", label="Ideal Diversity (1.0)")
        plt.ylabel("Unique Particle Fraction", fontsize=11)
        plt.title(f"Diversity at {st_ref} Steps", fontsize=12, fontweight="bold")
        plt.ylim(0, 1.15)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig2_flow_diversity.png")
        plt.close()
    except Exception as e:
        print(f"[!] Plot note: {e}")

    print("\n" + "=" * 80)
    print(f"[+] Exact Flow Benchmark Complete! Results saved to:")
    print(f"    - CSV:   {csv_file}")
    print(f"    - JSON:  {json_file}")
    print(f"    - Grids: {samples_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
