"""Argument parsers backing the root-level train.py and sample.py scripts."""

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from diffusion.datasets import make_data_module
from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.engine import load_checkpoint, save_checkpoint, train_epoch
from diffusion.metrics import FIDEvaluator, SamplingMetrics, measure_sampling, write_metrics
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import ToyGaussianMixtureTarget


DEFAULT_PRETRAINED_MODEL = "google/ddpm-cifar10-32"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _model_configuration(name: str, sample_shape: tuple[int, ...], base_channels: int,
                         toy_hidden: int, pretrained_model_id: str, revision: str | None,
                         pretrained_unet_subfolder: str | None):
    if name == "auto":
        name = "toy-mlp" if len(sample_shape) == 1 else "simple-unet"
    if name == "toy-mlp":
        if len(sample_shape) != 1:
            raise ValueError("toy-mlp requires vector data, such as --dataset toy-mixture")
        model_args = {"input_dim": math.prod(sample_shape), "hidden_channels": toy_hidden}
    elif name == "simple-unet":
        if len(sample_shape) != 3:
            raise ValueError("simple-unet requires image data, such as --dataset toy-images")
        model_args = {"image_channels": sample_shape[0], "base_channels": base_channels}
    elif name == "huggingface-unet":
        if len(sample_shape) != 3:
            raise ValueError("huggingface-unet requires image data")
        model_args = {"pretrained_model_id": pretrained_model_id, "revision": revision,
                      "subfolder": pretrained_unet_subfolder}
    else:
        raise ValueError(f"Unknown model: {name}")
    return name, model_args


def _sample_with_fid(model, pipeline, sample_shape, device, args, sampler_steps, generator):
    """Stream an FID-sized generated set while retaining only the preview grid."""
    if len(sample_shape) != 3 or sample_shape[0] != 3:
        raise ValueError("FID is available only for RGB image models")
    if args.count < 1 or args.fid_samples < 2 or args.fid_batch_size < 1:
        raise ValueError("--count and --fid-batch-size must be positive; --fid-samples must be at least 2")
    if args.fid_samples < args.count:
        raise ValueError("--fid-samples must be at least --count so the saved preview comes from the FID run")

    evaluator = FIDEvaluator(device)
    reference_module = make_data_module(
        args.fid_dataset, args.fid_data_dir, sample_shape[-1], args.fid_download,
        train=False,
    )
    if tuple(reference_module.sample_shape) != sample_shape:
        raise ValueError(
            f"FID reference shape {reference_module.sample_shape} does not match "
            f"generated image shape {sample_shape}"
        )
    reference_dataset = reference_module.dataset()
    if len(reference_dataset) < args.fid_samples:
        raise ValueError(
            f"FID requested {args.fid_samples} real images, but the reference dataset "
            f"contains only {len(reference_dataset)}"
        )
    reference_generator = (torch.Generator().manual_seed(args.seed)
                           if args.seed is not None else None)
    reference_loader = DataLoader(
        reference_dataset,
        batch_size=args.fid_batch_size,
        shuffle=True,
        num_workers=args.fid_num_workers,
        pin_memory=device.type == "cuda",
        generator=reference_generator,
    )

    remaining = args.fid_samples
    for batch in reference_loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        take = min(remaining, images.shape[0])
        evaluator.update_real(images[:take])
        remaining -= take
        if remaining == 0:
            break

    preview_batches = []
    remaining_preview = args.count
    remaining_fake = args.fid_samples
    total_nfe = 0
    total_forward_calls = 0
    sampling_seconds = 0.0
    sampling_batches = 0
    while remaining_fake:
        batch_count = min(args.fid_batch_size, remaining_fake)
        shape = (batch_count, *sample_shape)
        generated, batch_metrics = measure_sampling(
            model, device, batch_count, sampler_steps,
            lambda shape=shape: pipeline.sample(
                model, shape, device, args.steps, generator
            ),
        )
        total_nfe += batch_metrics.nfe
        total_forward_calls += batch_metrics.model_forward_calls
        sampling_seconds += batch_metrics.elapsed_seconds
        sampling_batches += 1

        preview_count = min(remaining_preview, batch_count)
        if preview_count:
            preview_batches.append(generated[:preview_count].cpu())
            remaining_preview -= preview_count
        evaluator.update_fake(generated)
        remaining_fake -= batch_count

    fid_score = evaluator.compute()
    # NFE is reported per generated sampling batch/trajectory. The total number
    # of actual model invocations across all chunks is recorded separately.
    average_nfe = round(total_nfe / sampling_batches) if sampling_batches else 0
    metrics = SamplingMetrics(
        nfe=average_nfe,
        sampler_steps=sampler_steps,
        sample_count=args.fid_samples,
        elapsed_seconds=sampling_seconds,
        model_forward_calls=total_forward_calls,
        sampling_batches=sampling_batches,
    )
    samples = torch.cat(preview_batches, dim=0)
    return samples, metrics, fid_score, evaluator.elapsed_seconds


def train_main(argv=None):
    parser = argparse.ArgumentParser(description="Train a diffusion noise-prediction model")
    parser.add_argument("--dataset", default="cifar10",
                        help="cifar10, imagefolder, toy-images, or toy-mixture")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true", help="download CIFAR-10 when needed")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--toy-size", type=int, default=512,
                        help="number of examples to generate for either toy dataset")
    parser.add_argument("--toy-seed", type=int, default=0)
    parser.add_argument("--model", default="auto",
                        help="auto, simple-unet, toy-mlp, or huggingface-unet")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--toy-hidden", type=int, default=128)
    parser.add_argument("--pretrained-model-id", default=None,
                        help=f"Hugging Face Diffusers repository (default: {DEFAULT_PRETRAINED_MODEL})")
    parser.add_argument("--pretrained-revision", default=None)
    parser.add_argument("--pretrained-unet-subfolder", default=None)
    parser.add_argument("--pretrained-scheduler-subfolder", default=None)
    parser.add_argument("--scheduler", default="auto",
                        help="auto, linear, cosine, or huggingface")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output", default="checkpoints/ddpm.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    data = make_data_module(args.dataset, args.data_dir, args.image_size, args.download,
                            args.toy_size, args.toy_seed)
    sample_shape = tuple(data.sample_shape)
    model_name = args.model
    pretrained_model_id = args.pretrained_model_id
    if model_name == "huggingface-unet" or args.scheduler == "huggingface":
        pretrained_model_id = pretrained_model_id or DEFAULT_PRETRAINED_MODEL
    model_name, model_args = _model_configuration(
        model_name, sample_shape, args.base_channels, args.toy_hidden,
        pretrained_model_id or DEFAULT_PRETRAINED_MODEL, args.pretrained_revision,
        args.pretrained_unet_subfolder
    )
    scheduler_name = args.scheduler
    if scheduler_name == "auto":
        scheduler_name = "huggingface" if model_name == "huggingface-unet" else "linear"

    device = _device(args.device)
    loader = data.make_dataloader(args.batch_size, args.num_workers)
    model = make_model(model_name, **model_args).to(device)
    if getattr(model, "sample_shape", None) is not None and tuple(model.sample_shape) != sample_shape:
        raise ValueError(f"Dataset shape {sample_shape} does not match pretrained model shape {model.sample_shape}")
    scheduler = make_scheduler(scheduler_name, args.timesteps, args.beta_start, args.beta_end,
                               pretrained_model_id, args.pretrained_revision,
                               args.pretrained_scheduler_subfolder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    config = vars(args).copy()
    config.update({
        "model": model_name,
        "model_args": model_args,
        "scheduler": scheduler_name,
        "pretrained_model_id": pretrained_model_id if scheduler_name == "huggingface" else
        (model_args.get("pretrained_model_id") if model_name == "huggingface-unet" else None),
        "pretrained_scheduler_subfolder": args.pretrained_scheduler_subfolder,
        "sample_shape": sample_shape,
        "channels": sample_shape[0] if len(sample_shape) == 3 else None,
        "image_size": sample_shape[-1] if len(sample_shape) == 3 else None,
        "timesteps": scheduler.num_train_timesteps,
    })
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, scheduler, loader, optimizer, device, args.grad_clip)
        print(f"epoch {epoch}/{args.epochs} | noise_mse={loss:.6f}")
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output, model, optimizer, epoch, config)
            print(f"saved {args.output}")


def sample_main(argv=None):
    parser = argparse.ArgumentParser(description="Generate samples with a trained or pretrained diffusion model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default=None, help="local framework checkpoint")
    source.add_argument("--pretrained-model-id", default=None,
                        help=f"sample directly from a Diffusers repository, e.g. {DEFAULT_PRETRAINED_MODEL}")
    source.add_argument("--toy-mixture-oracle", action="store_true",
                        help="use the exact score model for the analytic 2D Gaussian-mixture target")
    parser.add_argument("--pretrained-revision", default=None)
    parser.add_argument("--pretrained-unet-subfolder", default=None)
    parser.add_argument("--pretrained-scheduler-subfolder", default=None)
    parser.add_argument("--scheduler", default="auto",
                        help="auto, linear, cosine, or huggingface for direct pretrained sampling")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--sampler", default="ddpm",
                        help="ddpm, ddim, annealed-langevin, or fixed-target langevin")
    parser.add_argument("--steps", type=int, default=None,
                        help="DDIM steps, Langevin iterations, or annealed Langevin noise levels")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity; zero is deterministic")
    parser.add_argument("--langevin-step-size", type=float, default=None,
                        help="Langevin base step size (defaults depend on sampler)")
    parser.add_argument("--langevin-steps-per-level", type=int, default=1,
                        help="ULA updates at each noise level for annealed-langevin")
    parser.add_argument("--langevin-timestep", type=int, default=0,
                        help="fixed DDPM noise level for the langevin sampler")
    parser.add_argument("--langevin-initial-scale", type=float, default=1.0,
                        help="initial Gaussian scale for fixed-target langevin")
    parser.add_argument("--birth-death", action="store_true",
                        help="enable fixed-population birth-death after each fixed-target ULA update")
    parser.add_argument("--bd-kernel-bandwidth", type=float, default=0.3,
                        help="Gaussian KDE bandwidth for birth-death particle rates")
    parser.add_argument("--toy-mixture-radius", type=float, default=2.0)
    parser.add_argument("--toy-mixture-standard-deviation", type=float, default=0.15)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="defaults to a PNG grid for images or a PT tensor for vectors")
    parser.add_argument("--metrics-output", default=None,
                        help="metrics JSON path (defaults to a sidecar next to --output)")
    parser.add_argument("--fid", action="store_true",
                        help="also compute FID against a real image reference dataset")
    parser.add_argument("--fid-dataset", choices=("cifar10", "imagefolder"), default="cifar10")
    parser.add_argument("--fid-data-dir", default="data",
                        help="dataset root used for FID reference images")
    parser.add_argument("--fid-download", action="store_true",
                        help="download the CIFAR-10 reference split if needed")
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="number of real and generated images used for FID")
    parser.add_argument("--fid-batch-size", type=int, default=64,
                        help="streaming batch size for FID reference and generated images")
    parser.add_argument("--fid-num-workers", type=int, default=0,
                        help="data-loader workers for FID reference images")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = _device(args.device)
    target = None
    if args.checkpoint:
        model, scheduler, config, _ = load_checkpoint(args.checkpoint, device)
        sample_shape = tuple(config.get("sample_shape", (config["channels"],
                              config["image_size"], config["image_size"])))
    elif args.toy_mixture_oracle:
        scheduler_name = "linear" if args.scheduler == "auto" else args.scheduler
        if scheduler_name == "huggingface":
            raise ValueError("the toy-mixture oracle requires a local linear or cosine scheduler")
        scheduler = make_scheduler(scheduler_name, args.timesteps, args.beta_start,
                                   args.beta_end).to(device)
        target = ToyGaussianMixtureTarget(
            radius=args.toy_mixture_radius,
            standard_deviation=args.toy_mixture_standard_deviation,
        )
        model = make_model(
            "toy-mixture-oracle", target=target, alpha_bars=scheduler.alpha_bars
        ).to(device)
        sample_shape = (2,)
    else:
        model_id = args.pretrained_model_id or DEFAULT_PRETRAINED_MODEL
        model = make_model("huggingface-unet", pretrained_model_id=model_id,
                           revision=args.pretrained_revision,
                           subfolder=args.pretrained_unet_subfolder).to(device)
        scheduler_name = "huggingface" if args.scheduler == "auto" else args.scheduler
        scheduler = make_scheduler(scheduler_name, args.timesteps, args.beta_start, args.beta_end,
                                   model_id, args.pretrained_revision,
                                   args.pretrained_scheduler_subfolder).to(device)
        sample_shape = tuple(model.sample_shape)
    model.eval()
    sampler = make_sampler(
        args.sampler,
        args.eta,
        langevin_step_size=args.langevin_step_size,
        langevin_steps_per_level=args.langevin_steps_per_level,
        langevin_timestep=args.langevin_timestep,
        langevin_initial_scale=args.langevin_initial_scale,
    )
    modifiers = []
    if args.birth_death:
        if target is None or not args.toy_mixture_oracle:
            raise ValueError("--birth-death currently requires --toy-mixture-oracle with its known target density")
        if args.sampler not in {"langevin", "ula"}:
            raise ValueError("--birth-death currently supports only the fixed-target --sampler langevin")
        if args.count < 2:
            raise ValueError("--birth-death requires --count of at least 2")
        modifiers.append(BirthDeathAcceleration(target, args.bd_kernel_bandwidth))
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    sampler_steps = len(sampler.inference_timesteps(scheduler, args.steps, device))
    fid_score = None
    fid_seconds = None
    if args.fid:
        samples, metrics, fid_score, fid_seconds = _sample_with_fid(
            model, pipeline, sample_shape, device, args, sampler_steps, generator
        )
    else:
        shape = (args.count, *sample_shape)
        samples, metrics = measure_sampling(
            model, device, args.count, sampler_steps,
            lambda: pipeline.sample(model, shape, device, args.steps, generator),
        )

    output = Path(args.output) if args.output else Path(
        "samples/sample.png" if samples.ndim == 4 else "samples/samples.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 4:
        save_image((samples + 1) / 2, output, nrow=args.nrow)
    else:
        torch.save(samples.cpu(), output)
    metrics_path = Path(args.metrics_output) if args.metrics_output else output.with_name(
        f"{output.stem}.metrics.json"
    )
    model_source = (
        f"checkpoint:{args.checkpoint}" if args.checkpoint else
        "toy-mixture-oracle" if args.toy_mixture_oracle else
        f"huggingface:{args.pretrained_model_id or DEFAULT_PRETRAINED_MODEL}"
    )
    metadata = {
        "model_source": model_source,
        "sampler": args.sampler,
        "device": str(device),
        "seed": args.seed,
        "sample_shape": list(sample_shape),
        "sample_output": str(output),
        "saved_preview_count": int(samples.shape[0]),
        "scheduler": type(scheduler).__name__,
        "birth_death_enabled": args.birth_death,
    }
    if args.sampler in {"langevin", "ula", "annealed-langevin", "ald"}:
        metadata.update({
            "langevin_step_size": sampler.step_size,
            "langevin_steps_per_level": getattr(sampler, "steps_per_level", None),
            "langevin_fixed_timestep": getattr(sampler, "timestep", None),
        })
    if args.birth_death:
        metadata.update({
            "birth_death_kernel_bandwidth": args.bd_kernel_bandwidth,
            "toy_mixture_radius": args.toy_mixture_radius,
            "toy_mixture_standard_deviation": args.toy_mixture_standard_deviation,
        })
    additional_metrics = None
    if args.fid:
        metadata.update({
            "fid_reference_dataset": args.fid_dataset,
            "fid_reference_data_dir": args.fid_data_dir,
            "fid_reference_split": "test" if args.fid_dataset == "cifar10" else "all",
            "fid_samples": args.fid_samples,
            "fid_batch_size": args.fid_batch_size,
            "fid_num_workers": args.fid_num_workers,
            "fid_feature_dimension": 2048,
        })
        additional_metrics = {
            "fid": fid_score,
            "fid_evaluation_seconds": fid_seconds,
        }
    write_metrics(metrics_path, metrics, metadata, additional_metrics)
    print(f"saved {samples.shape[0]} samples to {output}")
    print(f"sampler={args.sampler} | steps={metrics.sampler_steps} | NFE={metrics.nfe} | "
          f"model_forward_calls={metrics.model_forward_calls} | "
          f"time={metrics.elapsed_seconds:.2f}s | "
          f"throughput={metrics.samples_per_second:.2f} samples/s")
    if args.fid:
        print(f"FID-2048={fid_score:.4f} | samples={args.fid_samples} | "
              f"reference={args.fid_dataset} | evaluation_time={fid_seconds:.2f}s")
    print(f"saved sampling metrics to {metrics_path}")
