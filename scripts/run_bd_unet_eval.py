"""Standalone Evaluation Suite for BD-UNet (U-Net Transport + Empirical KDE Birth-Death).

Architecture:
- Reverse Diffusion Transport (moving x_t -> x_{t-1}): Pretrained U-Net (google/ddpm-cifar10-32)
- Birth-Death Resampling (adding/removing particles): Empirical Kernel Density Estimation (KDE) over reference images

Compares:
1. Baseline DDIM (Pure U-Net, No Birth-Death)
2. BD-UNet (U-Net Transport + Empirical KDE Birth-Death)

Usage on Kaggle/GPU:
    python3 scripts/run_bd_unet_eval.py --steps 10 25 50 --device cuda
    python3 scripts/run_bd_unet_eval.py --steps 10 25 --fid --fid-samples 512 --device cuda
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diffusion.acceleration import BirthDeathAcceleration
from diffusion.datasets import make_data_module
from diffusion.metrics import FIDEvaluator, measure_sampling, rbf_mmd_squared
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import EmpiricalNoisedDataTarget


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

    diff = (flat.unsqueeze(1) - flat.unsqueeze(0)).abs().mean(dim=-1)
    diff.fill_diagonal_(1.0)

    visited = set()
    unique_count = 0
    for i in range(N):
        if i in visited:
            continue
        unique_count += 1
        visited.add(i)
        duplicates = (diff[i] < threshold).nonzero(as_tuple=True)[0].tolist()
        for d in duplicates:
            visited.add(d)

    return float(unique_count / N)


def run_baseline_case(
    *,
    model,
    scheduler,
    device,
    shape,
    steps,
    sampler_name="ddim",
    eta=0.0,
    seed=42,
    real_images=None,
    fid_batch_size=64,
):
    """Runs standard baseline DDIM without Birth-Death."""
    sampler = make_sampler(sampler_name, eta=eta)
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=[])

    gen = torch.Generator(device=device).manual_seed(seed)
    preview_samples, sampling_metrics = measure_sampling(
        model,
        device,
        shape[0],
        steps,
        lambda: pipeline.sample(model, shape, device, steps, gen),
    )

    uniqueness = _unique_fraction(preview_samples)
    mmd_score = None
    fid_score = None

    if real_images is not None:
        flat_gen = preview_samples.reshape(preview_samples.shape[0], -1)
        flat_real = real_images[: preview_samples.shape[0]].reshape(
            preview_samples.shape[0], -1
        )
        mmd_score = float(rbf_mmd_squared(flat_gen, flat_real).item())

        n_fid = min(real_images.shape[0], 512)
        evaluator = FIDEvaluator(device)
        evaluator.update_real(real_images[:n_fid])

        batches = []
        needed = n_fid
        while needed > 0:
            b_size = min(fid_batch_size, needed)
            b_shape = (b_size, *shape[1:])
            g = torch.Generator(device=device)
            s_batch = pipeline.sample(model, b_shape, device, steps, g)
            batches.append(s_batch)
            needed -= b_size

        all_fake = torch.cat(batches, dim=0)[:n_fid]
        evaluator.update_fake(all_fake)
        try:
            fid_score = float(evaluator.compute())
        except Exception as e:
            print(f"[!] FID warning: {e}")
            fid_score = None

    record = {
        "case": f"baseline_{steps}",
        "steps": steps,
        "method": "Baseline DDIM (Pure U-Net)",
        "complexity": "O(N) Deep",
        "throughput_img_per_sec": round(sampling_metrics.samples_per_second, 2),
        "latency_sec_per_sample": round(sampling_metrics.seconds_per_sample, 4),
        "uniqueness": round(uniqueness, 4),
        "nfe": sampling_metrics.nfe,
        "mmd": round(mmd_score, 4) if mmd_score is not None else None,
        "fid": round(fid_score, 2) if fid_score is not None else None,
    }
    return record, preview_samples


def run_bd_unet_case(
    *,
    model,
    scheduler,
    target,
    device,
    shape,
    steps,
    sampler_name="ddim",
    eta=0.0,
    seed=42,
    kernel_bandwidth=0.3,
    every_k=1,
    total_time=1.0,
    max_delta_tau=0.005,
    min_timestep=100,
    max_timestep=None,
    max_event_prob=0.25,
    real_images=None,
    fid_batch_size=64,
):
    """Runs U-Net transport with Empirical KDE Birth-Death resampling."""
    sampler = make_sampler(sampler_name, eta=eta)
    bd_modifier = BirthDeathAcceleration(
        target,
        kernel_bandwidth=kernel_bandwidth,
        every_k_steps=every_k,
        total_time=total_time,
        min_timestep=min_timestep,
        max_timestep=max_timestep,
        max_delta_tau=max_delta_tau,
        max_event_probability=max_event_prob,
    )
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=[bd_modifier])

    gen = torch.Generator(device=device).manual_seed(seed)
    mod_gen = torch.Generator(device=device).manual_seed(seed + 100_003)

    preview_samples, sampling_metrics = measure_sampling(
        model,
        device,
        shape[0],
        steps,
        lambda: pipeline.sample(model, shape, device, steps, gen, mod_gen),
    )

    uniqueness = _unique_fraction(preview_samples)
    mmd_score = None
    fid_score = None

    if real_images is not None:
        flat_gen = preview_samples.reshape(preview_samples.shape[0], -1)
        flat_real = real_images[: preview_samples.shape[0]].reshape(
            preview_samples.shape[0], -1
        )
        mmd_score = float(rbf_mmd_squared(flat_gen, flat_real).item())

        n_fid = min(real_images.shape[0], 512)
        evaluator = FIDEvaluator(device)
        evaluator.update_real(real_images[:n_fid])

        batches = []
        needed = n_fid
        while needed > 0:
            b_size = min(fid_batch_size, needed)
            b_shape = (b_size, *shape[1:])
            g = torch.Generator(device=device)
            mg = torch.Generator(device=device)
            s_batch = pipeline.sample(model, b_shape, device, steps, g, mg)
            batches.append(s_batch)
            needed -= b_size

        all_fake = torch.cat(batches, dim=0)[:n_fid]
        evaluator.update_fake(all_fake)
        try:
            fid_score = float(evaluator.compute())
        except Exception as e:
            print(f"[!] FID warning: {e}")
            fid_score = None

    record = {
        "case": f"bd_unet_{steps}",
        "steps": steps,
        "method": "BD-UNet (U-Net Transport + Empirical KDE BD)",
        "complexity": "O(N) + O(NMD) KDE",
        "throughput_img_per_sec": round(sampling_metrics.samples_per_second, 2),
        "latency_sec_per_sample": round(sampling_metrics.seconds_per_sample, 4),
        "uniqueness": round(uniqueness, 4),
        "nfe": sampling_metrics.nfe,
        "mmd": round(mmd_score, 4) if mmd_score is not None else None,
        "fid": round(fid_score, 2) if fid_score is not None else None,
    }
    return record, preview_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate BD-UNet (U-Net Transport + Empirical KDE BD)")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[10, 25, 50],
        help="Diffusion step counts to benchmark",
    )
    parser.add_argument(
        "--pretrained-model-id",
        default="google/ddpm-cifar10-32",
        help="Hugging Face pretrained model ID",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=512,
        help="Number of reference images used for empirical KDE target",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Preview batch size")
    parser.add_argument(
        "--bd-min-timestep",
        type=int,
        default=100,
        help="Minimum timestep for BD (set to 0 for unrestricted BD)",
    )
    parser.add_argument(
        "--bd-max-timestep",
        type=int,
        default=None,
        help="Maximum timestep for BD interventions (default: None)",
    )
    parser.add_argument(
        "--bd-max-time",
        type=float,
        default=0.005,
        help="Maximum pseudo-time step delta_tau per BD intervention (default: 0.005)",
    )
    parser.add_argument(
        "--bd-max-event-prob",
        type=float,
        default=0.25,
        help="Max probability cap for single BD event (set to 1.0 for unrestricted)",
    )
    parser.add_argument("--bd-total-time", type=float, default=1.0)
    parser.add_argument("--bd-kernel-bandwidth", type=float, default=0.3)
    parser.add_argument("--bd-every-k", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="bd_unet_results")
    parser.add_argument("--fid", action="store_true", help="Compute FID")
    parser.add_argument("--fid-samples", type=int, default=512)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
    )
    print(f"[+] Initializing BD-UNet Evaluation on {device}...")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Pretrained U-Net Model
    print(f"[+] Loading Pretrained Model: {args.pretrained_model_id}...")
    model = make_model("huggingface-unet", pretrained_model_id=args.pretrained_model_id).to(device)
    scheduler = make_scheduler("huggingface", pretrained_model_id=args.pretrained_model_id).to(device)
    model.eval()
    sample_shape = tuple(model.sample_shape)
    print(f"[✓] Model loaded: {args.pretrained_model_id} (sample shape: {sample_shape})")

    # 2. Build Empirical KDE Target for Birth-Death
    print(f"[+] Loading CIFAR-10 reference images (count={args.reference_count})...")
    try:
        data_mod = make_data_module("cifar10", "data", 32, False, train=True)
        dataset = data_mod.dataset()
    except RuntimeError:
        data_mod = make_data_module("cifar10", "data", 32, True, train=True)
        dataset = data_mod.dataset()

    ref_count = min(args.reference_count, len(dataset))
    target = EmpiricalNoisedDataTarget.from_dataset(
        dataset, num_reference_samples=ref_count, seed=args.seed, reference_chunk_size=64
    )
    print(f"[✓] Empirical KDE Target constructed with {ref_count} references.")

    # 3. Setup real reference images for metrics
    real_images = None
    if args.fid or True:
        print("[+] Preparing real CIFAR-10 reference images for evaluation...")
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=args.fid_batch_size, shuffle=True)
        imgs = []
        acc = 0
        target_imgs = args.fid_samples if args.fid else args.batch_size
        for batch in loader:
            b_img = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(target_imgs - acc, b_img.shape[0])
            imgs.append(b_img[:take])
            acc += take
            if acc >= target_imgs:
                break
        real_images = torch.cat(imgs, dim=0).to(device)

    # 4. Run Benchmark Grid
    results = []
    shape = (args.batch_size, *sample_shape)

    for steps in args.steps:
        print(f"\n[Case: baseline_{steps}] Running Baseline DDIM (Pure U-Net) at {steps} steps...")
        rec_base, prev_base = run_baseline_case(
            model=model,
            scheduler=scheduler,
            device=device,
            shape=shape,
            steps=steps,
            eta=args.eta,
            seed=args.seed,
            real_images=real_images if args.fid else None,
            fid_batch_size=args.fid_batch_size,
        )
        results.append(rec_base)
        save_image(
            (prev_base + 1.0) / 2.0,
            samples_dir / f"baseline_{steps}steps.png",
            nrow=4,
        )
        print(
            f"   -> Samples/s: {rec_base['throughput_img_per_sec']} | "
            f"Unique: {rec_base['uniqueness']:.4f} | "
            f"FID: {rec_base['fid']}"
        )

        print(f"[Case: bd_unet_{steps}] Running BD-UNet (U-Net + Empirical KDE BD) at {steps} steps...")
        rec_bd, prev_bd = run_bd_unet_case(
            model=model,
            scheduler=scheduler,
            target=target,
            device=device,
            shape=shape,
            steps=steps,
            eta=args.eta,
            seed=args.seed,
            kernel_bandwidth=args.bd_kernel_bandwidth,
            every_k=args.bd_every_k,
            total_time=args.bd_total_time,
            max_delta_tau=args.bd_max_time,
            min_timestep=args.bd_min_timestep,
            max_timestep=args.bd_max_timestep,
            max_event_prob=args.bd_max_event_prob,
            real_images=real_images if args.fid else None,
            fid_batch_size=args.fid_batch_size,
        )
        results.append(rec_bd)
        save_image(
            (prev_bd + 1.0) / 2.0,
            samples_dir / f"bd_unet_{steps}steps.png",
            nrow=4,
        )
        print(
            f"   -> Samples/s: {rec_bd['throughput_img_per_sec']} | "
            f"Unique: {rec_bd['uniqueness']:.4f} | "
            f"FID: {rec_bd['fid']}"
        )

    # 5. Export Summary Artifacts
    json_path = out_dir / "bd_unet_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = out_dir / "bd_unet_results.csv"
    keys = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    # 6. Zip everything
    zip_path = Path("bd_unet_results.zip")
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in out_dir.rglob("*"):
            if file.is_file():
                zipf.write(file, arcname=file.relative_to(out_dir.parent))
    print(f"\n[✓] Results packaged into {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
