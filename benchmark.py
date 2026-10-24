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
    # 1. Full ancestral DDPM (all training timesteps)
    # 2. DDIM at varying steps (e.g. 100, 50, 25, 10)
    if steps_list is None:
        steps_list = [100, 50, 25, 10]

    configs = [("ddpm", train_timesteps, 0.0, None)]
    for s in steps_list:
        if s < train_timesteps:
            configs.append(("ddim", s, eta, None))
            if bd_modifier is not None:
                configs.append(("ddim+bd", s, eta, [bd_modifier]))

    results = []
    csv_file = out_path / "benchmark_results.csv"

    print("\n" + "=" * 75)
    print(f"{'Sampler':<10} | {'Steps':<8} | {'Total Time (s)':<15} | {'Sec/Sample':<12} | {'Steps/Sec':<10}")
    print("-" * 75)

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

        results.append({
            "sampler": sampler_name,
            "steps": num_steps,
            "count": count,
            "total_time_sec": round(elapsed, 4),
            "sec_per_sample": round(sec_per_sample, 4),
            "steps_per_sec": round(steps_per_sec, 2),
            "image_path": str(save_target),
        })

        print(f"{sampler_name:<10} | {num_steps:<8} | {elapsed:<15.2f} | {sec_per_sample:<12.4f} | {steps_per_sec:<10.2f}")

    print("=" * 75)

    # Save CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sampler", "steps", "count", "total_time_sec", "sec_per_sample", "steps_per_sec", "image_path"])
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
    )


if __name__ == "__main__":
    main()

