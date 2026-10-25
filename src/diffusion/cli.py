"""Argument parsers backing the root-level train.py and sample.py scripts."""

import argparse
import math
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from diffusion.datasets import make_data_module
from diffusion.acceleration.birth_death import BirthDeathAcceleration
from diffusion.engine import load_checkpoint, save_checkpoint, train_epoch
from diffusion.metrics import (
    FIDEvaluator,
    SamplingMetrics,
    measure_sampling,
    rbf_mmd_squared,
    write_metrics,
)
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler
from diffusion.targets import EmpiricalNoisedDataTarget, ToyGaussianMixtureTarget


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


def _sample_with_fid(model, pipeline, sample_shape, device, args, sampler_steps,
                     generator, modifier_generator):
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
                model, shape, device, args.steps, generator, modifier_generator
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


@torch.no_grad()
def _empirical_target_metrics(samples, target, sampler, scheduler, args):
    """Score samples against the same finite data subset used as the target.

    The sampler describes its final state through SamplingContext. That keeps
    the metric coordinate conversion aligned with the sampler interface rather
    than relying on sampler names.
    """
    references = target.reference_samples.to(device=samples.device, dtype=samples.dtype)
    trajectory = sampler.inference_timesteps(scheduler, args.steps, samples.device)
    last_timestep = int(trajectory[-1].item())
    final_context = sampler.sampling_context(last_timestep, -1, scheduler)
    if final_context.is_clean_sample or final_context.state_coordinates == "clean_x0":
        density_timestep = 0
        alpha_bar = scheduler.alpha_bars[density_timestep].to(samples)
        density_samples = samples / alpha_bar.sqrt()
        state_description = "clean x0 treated as the timestep-0 empirical marginal"
        # DDPM/DDIM return clean image estimates; comparing those against the
        # clean reference examples gives an intuitive data-space MMD.
        mmd_timestep = None
        mmd_state_description = "clean reference examples"
        mmd_samples = samples
    else:
        if final_context.state_timestep is None:
            raise ValueError("the final sampler state does not specify a target timestep")
        density_timestep = int(final_context.state_timestep)
        if final_context.state_coordinates == "normalized_x_t":
            density_samples = samples
        elif final_context.state_coordinates == "x_t":
            alpha_bar = scheduler.alpha_bars[density_timestep].to(samples)
            density_samples = samples / alpha_bar.sqrt()
        else:
            raise ValueError(
                "empirical-target metrics support final states in x_t, normalized_x_t, "
                "or clean_x0 coordinates"
            )
        state_description = (
            f"{final_context.state_coordinates} sampler state at timestep {density_timestep}"
        )
        mmd_timestep = density_timestep
        mmd_samples = density_samples
        mmd_state_description = (
            f"sampled Gaussian-smoothed target at timestep {density_timestep}"
        )

    if mmd_timestep is None:
        mmd_references = references
    else:
        # ULA states represent a noised empirical marginal rather than clean
        # images. Draw an independent reproducible reference set from that
        # same finite Gaussian mixture before comparing them with MMD.
        metric_generator = torch.Generator(device=samples.device)
        if args.seed is None:
            metric_generator.seed()
        else:
            metric_generator.manual_seed(args.seed + 200_003)
        indices = torch.randint(
            references.shape[0], (references.shape[0],),
            device=samples.device, generator=metric_generator,
        )
        sigma = target.noise_std(
            mmd_timestep, scheduler, device=samples.device, dtype=samples.dtype
        )
        reference_noise = torch.randn(
            references.shape, device=samples.device, dtype=samples.dtype,
            generator=metric_generator,
        )
        mmd_references = references.index_select(0, indices) + sigma * reference_noise

    quality_metrics, metadata = rbf_mmd_squared(mmd_samples, mmd_references)

    log_prob = target.log_prob(density_samples, density_timestep, scheduler)
    mean_nll = float((-log_prob.mean()).item())
    dimensions = density_samples[0].numel()
    quality_metrics.update({
        "empirical_target_nll": mean_nll,
        "empirical_target_nll_per_dimension": mean_nll / dimensions,
    })
    metadata.update({
        "empirical_target_nll_scope": (
            "negative log density under the Gaussian-smoothed finite reference subset; "
            "not the unknown full source-dataset density"
        ),
        "empirical_target_nll_evaluation_timestep": int(density_timestep),
        "empirical_target_nll_sample_state": state_description,
        "empirical_target_nll_sample_count": int(samples.shape[0]),
        "empirical_target_nll_reference_count": int(references.shape[0]),
        "empirical_target_nll_dimensions": int(dimensions),
        "empirical_target_mmd_reference_state": mmd_state_description,
        "empirical_target_mmd_evaluation_timestep": mmd_timestep,
    })
    return quality_metrics, metadata


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
    source.add_argument("--empirical-data-target", action="store_true",
                        help="use the exact score/log density of a Gaussian-smoothed subset of a dataset")
    parser.add_argument("--target-dataset", default="toy-mixture",
                        help="dataset used by --empirical-data-target: cifar10, imagefolder, toy-images, or toy-mixture")
    parser.add_argument("--target-data-dir", default="data",
                        help="local dataset root for --empirical-data-target")
    parser.add_argument("--target-download", action="store_true",
                        help="download CIFAR-10 for --empirical-data-target when it is not present")
    parser.add_argument("--target-image-size", type=int, default=32,
                        help="image resolution used by image datasets for --empirical-data-target")
    parser.add_argument("--target-toy-size", type=int, default=512,
                        help="number of examples generated for a toy empirical target")
    parser.add_argument("--target-reference-count", type=int, default=256,
                        help="number of dataset examples used in the empirical Gaussian mixture")
    parser.add_argument("--target-reference-seed", type=int, default=0,
                        help="seed for selecting the empirical target's reference subset")
    parser.add_argument("--target-reference-chunk-size", type=int, default=64,
                        help="reference examples processed per log-density/score chunk")
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
                        help="enable fixed-population birth-death between compatible sampler updates")
    parser.add_argument("--bd-kernel-bandwidth", type=float, default=0.3,
                        help="Gaussian KDE bandwidth for birth-death particle rates")
    parser.add_argument("--bd-every-k", type=int, default=1,
                        help="apply birth-death after every k sampler updates (default: 1)")
    parser.add_argument("--bd-total-time", type=float, default=1.0,
                        help="total birth-death pseudo-time budget divided over scheduled interventions")
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

    if args.birth_death and args.sampler == "ddim" and args.eta == 0.0:
        print(
            "warning: DDIM with eta=0 is deterministic. Birth-death can copy particles, "
            "and identical copies then remain identical through later DDIM updates. "
            "Consider a positive --eta and tune --bd-total-time and --bd-kernel-bandwidth.",
            file=sys.stderr,
        )

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
    elif args.empirical_data_target:
        scheduler_name = "linear" if args.scheduler == "auto" else args.scheduler
        if scheduler_name == "huggingface":
            raise ValueError("--empirical-data-target currently requires a local linear or cosine scheduler")
        scheduler = make_scheduler(scheduler_name, args.timesteps, args.beta_start,
                                   args.beta_end).to(device)
        # This assumes the selected dataset is present locally (unless the user
        # opts into CIFAR-10 download) and uses its data-module preprocessing.
        data_module = make_data_module(
            args.target_dataset, args.target_data_dir, args.target_image_size,
            args.target_download, args.target_toy_size, args.target_reference_seed,
        )
        reference_dataset = data_module.dataset()
        target = EmpiricalNoisedDataTarget.from_dataset(
            reference_dataset,
            num_reference_samples=args.target_reference_count,
            seed=args.target_reference_seed,
            reference_chunk_size=args.target_reference_chunk_size,
        )
        model = make_model("target-score", target=target, scheduler=scheduler).to(device)
        sample_shape = tuple(data_module.sample_shape)
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
        if target is None or not (args.toy_mixture_oracle or args.empirical_data_target):
            raise ValueError(
                "--birth-death requires --toy-mixture-oracle or --empirical-data-target with a known target density"
            )
        if args.sampler not in {"ddpm", "ddim", "langevin", "ula"}:
            raise ValueError(
                "--birth-death supports ddpm, ddim, or fixed-target langevin; "
                "annealed-langevin is not supported yet"
            )
        if args.count < 2 and not args.fid:
            raise ValueError("--birth-death requires --count of at least 2")
        if args.fid and (args.fid_batch_size < 2 or
                         args.fid_samples % args.fid_batch_size == 1):
            raise ValueError(
                "--birth-death with --fid requires every generated batch to contain "
                "at least 2 samples; adjust --fid-batch-size or --fid-samples"
            )
        modifiers.append(BirthDeathAcceleration(
            target,
            args.bd_kernel_bandwidth,
            every_k_steps=args.bd_every_k,
            total_time=args.bd_total_time,
        ))
    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    modifier_generator = torch.Generator(device=device)
    if args.seed is None:
        modifier_generator.seed()
    else:
        modifier_generator.manual_seed(args.seed + 100_003)
    sampler_steps = len(sampler.inference_timesteps(scheduler, args.steps, device))
    fid_score = None
    fid_seconds = None
    if args.fid:
        samples, metrics, fid_score, fid_seconds = _sample_with_fid(
            model, pipeline, sample_shape, device, args, sampler_steps,
            generator, modifier_generator,
        )
    else:
        shape = (args.count, *sample_shape)
        samples, metrics = measure_sampling(
            model, device, args.count, sampler_steps,
            lambda: pipeline.sample(
                model, shape, device, args.steps, generator, modifier_generator
            ),
        )

    output = Path(args.output) if args.output else Path(
        "samples/sample.png" if samples.ndim == 4 else "samples/samples.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 4:
        save_image((samples + 1) / 2, output, nrow=args.nrow)
        # Count the distinct image tiles that survive the same 8-bit rounding
        # used by save_image, so exact visual duplicates are visible in metrics.
        saved_samples = ((samples + 1) * 127.5 + 0.5).clamp(0, 255).to(torch.uint8)
        unique_sample_label = "distinct 8-bit image tiles"
    else:
        torch.save(samples.cpu(), output)
        saved_samples = samples
        unique_sample_label = "distinct saved samples"
    saved_preview_unique_count = int(
        torch.unique(saved_samples.reshape(saved_samples.shape[0], -1), dim=0).shape[0]
    )
    metrics_path = Path(args.metrics_output) if args.metrics_output else output.with_name(
        f"{output.stem}.metrics.json"
    )
    model_source = (
        f"checkpoint:{args.checkpoint}" if args.checkpoint else
        "toy-mixture-oracle" if args.toy_mixture_oracle else
        f"empirical-data-target:{args.target_dataset}" if args.empirical_data_target else
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
        "saved_preview_unique_count": saved_preview_unique_count,
        "scheduler": type(scheduler).__name__,
        "birth_death_enabled": args.birth_death,
    }
    if args.empirical_data_target:
        metadata.update({
            "target_dataset": args.target_dataset,
            "target_data_dir": args.target_data_dir,
            "target_reference_count": args.target_reference_count,
            "target_reference_seed": args.target_reference_seed,
            "target_reference_chunk_size": args.target_reference_chunk_size,
        })
    if args.sampler in {"langevin", "ula", "annealed-langevin", "ald"}:
        metadata.update({
            "langevin_step_size": sampler.step_size,
            "langevin_steps_per_level": getattr(sampler, "steps_per_level", None),
            "langevin_fixed_timestep": getattr(sampler, "timestep", None),
        })
    if args.birth_death:
        metadata.update({
            "birth_death_kernel_bandwidth": args.bd_kernel_bandwidth,
            "birth_death_every_k_steps": args.bd_every_k,
            "birth_death_total_time": args.bd_total_time,
        })
        if args.toy_mixture_oracle:
            metadata.update({
                "toy_mixture_radius": args.toy_mixture_radius,
                "toy_mixture_standard_deviation": args.toy_mixture_standard_deviation,
            })
    additional_metrics = {}
    if args.empirical_data_target:
        target_metrics, target_metric_metadata = _empirical_target_metrics(
            samples, target, sampler, scheduler, args
        )
        additional_metrics.update(target_metrics)
        metadata.update(target_metric_metadata)
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
        additional_metrics.update({
            "fid": fid_score,
            "fid_evaluation_seconds": fid_seconds,
        })
    write_metrics(metrics_path, metrics, metadata, additional_metrics or None)
    print(f"saved {samples.shape[0]} samples to {output}")
    print(f"{unique_sample_label}={saved_preview_unique_count}/{samples.shape[0]}")
    print(f"sampler={args.sampler} | steps={metrics.sampler_steps} | NFE={metrics.nfe} | "
          f"model_forward_calls={metrics.model_forward_calls} | "
          f"time={metrics.elapsed_seconds:.2f}s | "
          f"throughput={metrics.samples_per_second:.2f} samples/s")
    if args.fid:
        print(f"FID-2048={fid_score:.4f} | samples={args.fid_samples} | "
              f"reference={args.fid_dataset} | evaluation_time={fid_seconds:.2f}s")
    if args.empirical_data_target:
        print(
            "empirical-target metrics | "
            f"NLL={additional_metrics['empirical_target_nll']:.4f} nats/sample "
            f"({additional_metrics['empirical_target_nll_per_dimension']:.6f} nats/dimension, "
            f"t={metadata['empirical_target_nll_evaluation_timestep']}) | "
            f"MMD^2={additional_metrics['empirical_target_mmd2_rbf_multiscale']:.6f} "
            "(pixel/sample-coordinate RBF, multi-scale)"
        )
    print(f"saved sampling metrics to {metrics_path}")
