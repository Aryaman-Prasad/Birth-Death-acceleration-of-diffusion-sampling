"""Benchmark baseline diffusion samplers and collect quality and inference time data."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from diffusion.acceleration import (
    BirthDeathDDPM,
    add_birth_death_ddpm_args,
    create_birth_death_ddpm_from_args,
)
from diffusion.cli import DEFAULT_PRETRAINED_MODEL, _device
from diffusion.engine import load_checkpoint
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler



def _get_sample_shape(model, config: dict | None = None, default: tuple[int, ...] = (3, 32, 32)) -> tuple[int, ...]:
    if hasattr(model, "sample_shape") and model.sample_shape is not None:
        return tuple(model.sample_shape)
    if config:
        if "sample_shape" in config:
            return tuple(config["sample_shape"])
        if "channels" in config and "image_size" in config:
            return (config["channels"], config["image_size"], config["image_size"])
    unet = getattr(model, "unet", model)
    if hasattr(unet, "config"):
        channels = getattr(unet.config, "in_channels", 3)
        size = getattr(unet.config, "sample_size", 32)
        spatial = (size, size) if isinstance(size, int) else tuple(size)
        return (channels, *spatial)
    return default

def run_benchmark(
    checkpoint: str | None = None,
    pretrained_model_id: str | None = None,
    pretrained_revision: str | None = None,
    pretrained_unet_subfolder: str | None = None,
    pretrained_scheduler_subfolder: str | None = None,
    count: int = 16,
    seed: int = 42,
    device_name: str = "auto",
    output_dir: str = "baseline_data",
    steps_list: list[int] | None = None,
    eta: float = 0.0,
    bd_modifier: Any = None,
    fid: bool = False,
    fid_samples: int = 1000,
    fid_batch_size: int = 64,
    fid_dataset: str = "cifar10",
    fid_data_dir: str = "data",
    fid_download: bool = True,
    no_ddpm: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    device = _device(device_name)
    print(f"Using device: {device}")

    # Load model and scheduler
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}...")
        model, scheduler, config, _ = load_checkpoint(checkpoint, device)
        sample_shape = _get_sample_shape(model, config)
    else:
        model_id = pretrained_model_id or DEFAULT_PRETRAINED_MODEL
        print(f"Loading pretrained model: {model_id}...")
        model = make_model(
            "huggingface-unet",
            pretrained_model_id=model_id,
            revision=pretrained_revision,
            subfolder=pretrained_unet_subfolder,
        ).to(device)
        scheduler = make_scheduler(
            "huggingface",
            num_train_timesteps=1000,
            beta_start=1e-4,
            beta_end=2e-2,
            pretrained_model_id=model_id,
            revision=pretrained_revision,
            pretrained_subfolder=pretrained_scheduler_subfolder,
        ).to(device)
        sample_shape = _get_sample_shape(model)

    model.eval()
    shape = (count, *sample_shape)
    train_timesteps = scheduler.num_train_timesteps

    # Benchmark configurations:
    if steps_list is None:
        steps_list = [100, 50, 25, 10]

    configs = []
    if not no_ddpm:
        configs.append(("ddpm", train_timesteps, 0.0, None))
    for s in steps_list:
        if s < train_timesteps:
            configs.append(("ddim", s, eta, None))
            if bd_modifier is not None:
                configs.append(("ddim+bd", s, eta, [bd_modifier]))

    # Load real reference images for FID if enabled
    real_batches = []
    if fid:
        from diffusion.metrics import FIDEvaluator
        from diffusion.datasets import make_data_module
        from torch.utils.data import DataLoader

        print(f"\n[+] Preparing FID reference data from {fid_dataset} ({fid_samples} images)...")
        ref_module = make_data_module(fid_dataset, fid_data_dir, sample_shape[-1], fid_download, train=False)
        ref_dataset = ref_module.dataset()
        ref_loader = DataLoader(
            ref_dataset, batch_size=fid_batch_size, shuffle=True,
            num_workers=2 if device.type == "cuda" else 0,
            pin_memory=device.type == "cuda"
        )
        remaining = fid_samples
        for batch in ref_loader:
            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            take = min(remaining, imgs.shape[0])
            real_batches.append(imgs[:take])
            remaining -= take
            if remaining <= 0:
                break
        print(f"[+] Loaded {sum(b.shape[0] for b in real_batches)} reference images.")

    results = []
    csv_file = out_path / "benchmark_results.csv"

    print("\n" + "=" * (88 if fid else 75))
    if fid:
        print(f"{'Sampler':<10} | {'Steps':<8} | {'Total Time (s)':<15} | {'Sec/Sample':<12} | {'Steps/Sec':<10} | {'FID':<8}")
    else:
        print(f"{'Sampler':<10} | {'Steps':<8} | {'Total Time (s)':<15} | {'Sec/Sample':<12} | {'Steps/Sec':<10}")
    print("-" * (88 if fid else 75))

    for sampler_name, num_steps, sampler_eta, active_modifiers in configs:
        base_sampler = "ddim" if "ddim" in sampler_name else "ddpm"
        sampler = make_sampler(base_sampler, sampler_eta)
        if active_modifiers:
            for m in active_modifiers:
                if hasattr(m, "reset"):
                    m.reset()
        pipeline = SamplingPipeline(sampler, scheduler, modifiers=active_modifiers)
        generator = torch.Generator(device=device).manual_seed(seed)

        # Warmup if CUDA
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        fid_score = None
        if fid:
            evaluator = FIDEvaluator(device)
            for r_batch in real_batches:
                evaluator.update_real(r_batch)

            preview_batches = []
            remaining_preview = count
            remaining_fake = fid_samples

            start = time.perf_counter()
            while remaining_fake > 0:
                batch_count = min(fid_batch_size, remaining_fake)
                batch_shape = (batch_count, *sample_shape)
                generated = pipeline.sample(
                    model=model,
                    shape=batch_shape,
                    device=device,
                    num_inference_steps=num_steps if base_sampler != "ddpm" else None,
                    generator=generator,
                )
                if remaining_preview > 0:
                    take = min(remaining_preview, batch_count)
                    preview_batches.append(generated[:take].cpu())
                    remaining_preview -= take
                evaluator.update_fake(generated)
                remaining_fake -= batch_count

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            fid_score = evaluator.compute()
            samples = torch.cat(preview_batches, dim=0)
            sec_per_sample = elapsed / fid_samples
            steps_per_sec = (num_steps * fid_samples) / elapsed
        else:
            start = time.perf_counter()
            samples = pipeline.sample(
                model=model,
                shape=shape,
                device=device,
                num_inference_steps=num_steps if base_sampler != "ddpm" else None,
                generator=generator,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            sec_per_sample = elapsed / count
            steps_per_sec = num_steps / elapsed

        img_filename = f"{sampler_name}_{num_steps}steps.png" if samples.ndim == 4 else f"{sampler_name}_{num_steps}steps.pt"
        save_target = out_path / img_filename
        if samples.ndim == 4:
            save_image((samples + 1) / 2, save_target, nrow=int(count**0.5) or 4)
        else:
            torch.save(samples.cpu(), save_target)

        entry = {
            "sampler": sampler_name,
            "steps": num_steps,
            "count": count,
            "fid_samples": fid_samples if fid else 0,
            "total_time_sec": round(elapsed, 4),
            "sec_per_sample": round(sec_per_sample, 4),
            "steps_per_sec": round(steps_per_sec, 2),
            "fid": round(fid_score, 2) if fid_score is not None else "",
            "image_path": str(save_target),
        }
        results.append(entry)

        if fid:
            print(f"{sampler_name:<10} | {num_steps:<8} | {elapsed:<15.2f} | {sec_per_sample:<12.4f} | {steps_per_sec:<10.2f} | {fid_score:<8.2f}")
        else:
            print(f"{sampler_name:<10} | {num_steps:<8} | {elapsed:<15.2f} | {sec_per_sample:<12.4f} | {steps_per_sec:<10.2f}")

    print("=" * (88 if fid else 75))

    # Save CSV
    fieldnames = ["sampler", "steps", "count", "fid_samples", "total_time_sec", "sec_per_sample", "steps_per_sec", "fid", "image_path"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Benchmark data saved to: {csv_file}")
    print(f"[+] Sample image grids saved in: {out_path.resolve()}/\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Collect baseline quality and inference time data")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--checkpoint", default=None, help="path to local trained checkpoint")
    source.add_argument("--pretrained-model-id", default=DEFAULT_PRETRAINED_MODEL,
                        help=f"HuggingFace model id (default: {DEFAULT_PRETRAINED_MODEL})")
    parser.add_argument("--count", type=int, default=16, help="number of samples per configuration")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--output-dir", default="baseline_data", help="directory to store csv and image results")
    parser.add_argument("--steps", nargs="+", type=int, default=[100, 50, 25, 10],
                        help="inference steps for DDIM (default: 100 50 25 10)")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (default 0.0)")
    parser.add_argument("--no-ddpm", action="store_true", help="skip full 1000-step DDPM to save time")
    parser.add_argument("--fid", action="store_true", help="evaluate FID score against reference dataset")
    parser.add_argument("--fid-samples", type=int, default=1000, help="number of real and generated images for FID")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="batch size for FID generation")
    parser.add_argument("--fid-dataset", choices=("cifar10", "imagefolder"), default="cifar10")
    parser.add_argument("--fid-data-dir", default="data", help="dataset root directory")
    parser.add_argument("--fid-download", action="store_true", default=True, help="download reference data if needed")
    add_birth_death_ddpm_args(parser)
    args = parser.parse_args()

    bd_modifier = create_birth_death_ddpm_from_args(args)

    run_benchmark(
        checkpoint=args.checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        count=args.count,
        seed=args.seed,
        device_name=args.device,
        output_dir=args.output_dir,
        steps_list=args.steps,
        eta=args.eta,
        bd_modifier=bd_modifier,
        fid=args.fid,
        fid_samples=args.fid_samples,
        fid_batch_size=args.fid_batch_size,
        fid_dataset=args.fid_dataset,
        fid_data_dir=args.fid_data_dir,
        fid_download=args.fid_download,
        no_ddpm=args.no_ddpm,
    )


if __name__ == "__main__":
    main()

