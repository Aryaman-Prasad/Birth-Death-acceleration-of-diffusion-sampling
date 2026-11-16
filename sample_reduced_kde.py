"""Sample from a diffusion model with Reduced-KDE Birth-Death acceleration.

Usage:
    python3 sample_reduced_kde.py --empirical-data-target --target-dataset cifar10 \
        --sampler ddim --steps 25 --eta 1.0 --birth-death --kde_reduce 25

This script adds ``--kde_reduce`` to compress high-dimensional image states
(e.g. 3072 dims for CIFAR-10, 196608 dims for CelebA-HQ 256) into a compact semantic
feature space (e.g. 25 dims) using a CNN backbone before evaluating population KDE.
"""

import argparse
import math
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from diffusion.acceleration.reduced_kde_birth_death import (
    ReducedKDEBirthDeathAcceleration,
)
from diffusion.datasets import make_data_module
from diffusion.engine import load_checkpoint
from diffusion.metrics import (
    FIDEvaluator,
    SamplingMetrics,
    measure_sampling,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sampling with Pretrained-CNN Reduced-KDE Birth-Death Acceleration"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default=None, help="local checkpoint")
    source.add_argument(
        "--pretrained-model-id",
        default=None,
        help=f"Diffusers model ID, e.g. {DEFAULT_PRETRAINED_MODEL}",
    )
    source.add_argument(
        "--toy-mixture-oracle",
        action="store_true",
        help="exact score model for 2D Gaussian mixture",
    )
    source.add_argument(
        "--empirical-data-target",
        action="store_true",
        help="exact score/log-density of Gaussian-smoothed dataset subset",
    )

    parser.add_argument(
        "--target-dataset",
        default="toy-mixture",
        help="dataset for --empirical-data-target: cifar10, imagefolder, toy-images, toy-mixture",
    )
    parser.add_argument(
        "--target-data-dir", default="data", help="dataset root directory"
    )
    parser.add_argument(
        "--target-download",
        action="store_true",
        help="download CIFAR-10 when needed",
    )
    parser.add_argument(
        "--target-image-size",
        type=int,
        default=32,
        help="image resolution for image datasets",
    )
    parser.add_argument(
        "--target-toy-size",
        type=int,
        default=512,
        help="number of examples for toy target",
    )
    parser.add_argument(
        "--target-reference-count",
        type=int,
        default=256,
        help="reference examples for empirical target",
    )
    parser.add_argument(
        "--target-reference-seed",
        type=int,
        default=0,
        help="seed for empirical target subset",
    )
    parser.add_argument(
        "--target-reference-chunk-size",
        type=int,
        default=64,
        help="chunk size for density evaluation",
    )

    parser.add_argument("--pretrained-revision", default=None)
    parser.add_argument("--pretrained-unet-subfolder", default=None)
    parser.add_argument("--pretrained-scheduler-subfolder", default=None)

    parser.add_argument(
        "--scheduler",
        default="auto",
        help="auto, linear, cosine, or huggingface",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument(
        "--sampler",
        default="ddim",
        help="ddpm, ddim, annealed-langevin, or fixed-target langevin",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="sampling steps count (e.g. 25, 50)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="DDIM stochasticity (eta=0 is deterministic; eta=1 is DDPM stochasticity)",
    )

    # Birth-Death Acceleration options
    parser.add_argument(
        "--birth-death",
        action="store_true",
        help="enable Birth-Death particle updates",
    )
    parser.add_argument(
        "--bd-kernel-bandwidth",
        type=float,
        default=0.3,
        help="Gaussian KDE bandwidth in reduced feature space",
    )
    parser.add_argument(
        "--bd-every-k",
        type=int,
        default=1,
        help="apply birth-death after every k sampler updates",
    )
    parser.add_argument(
        "--bd-total-time",
        type=float,
        default=1.0,
        help="total birth-death pseudo-time budget",
    )

    # NEW: Reduced-KDE Dimensionality Reduction arguments
    parser.add_argument(
        "--kde_reduce",
        "--kde-reduce",
        type=int,
        dest="kde_reduce",
        default=25,
        help="target compressed feature dimension for KDE (default: 25, much smaller than 3072)",
    )
    parser.add_argument(
        "--cnn-backbone",
        choices=("auto", "vgg16", "resnet18", "convnet"),
        default="auto",
        help="pretrained CNN backbone for feature extraction (default: auto)",
    )
    parser.add_argument(
        "--projection-method",
        choices=("pca", "orthogonal", "linear", "mlp"),
        default="pca",
        help="compression method to reach --kde_reduce dimensions: pca (SVD on refs), orthogonal, linear (nn.Linear), mlp (nn.Sequential)",
    )
    parser.add_argument(
        "--noise-references",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to noise reference images at timestep t to match marginal noise level",
    )

    # General sampling options
    parser.add_argument("--toy-mixture-radius", type=float, default=2.0)
    parser.add_argument(
        "--toy-mixture-standard-deviation", type=float, default=0.15
    )
    parser.add_argument(
        "--count", type=int, default=16, help="number of samples to generate"
    )
    parser.add_argument(
        "--nrow", type=int, default=4, help="grid rows for saved image preview"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument(
        "--fid", action="store_true", help="evaluate FID against real dataset"
    )
    parser.add_argument(
        "--fid-dataset", choices=("cifar10", "imagefolder"), default="cifar10"
    )
    parser.add_argument("--fid-data-dir", default="data")
    parser.add_argument("--fid-download", action="store_true")
    parser.add_argument("--fid-samples", type=int, default=10000)
    parser.add_argument("--fid-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = _device(args.device)

    print(
        f"[+] Reduced-KDE Sampler: --kde_reduce={args.kde_reduce} "
        f"backbone={args.cnn_backbone} proj={args.projection_method}"
    )

    target = None
    if args.checkpoint:
        model, scheduler, config, _ = load_checkpoint(args.checkpoint, device)
        sample_shape = tuple(
            config.get(
                "sample_shape",
                (config["channels"], config["image_size"], config["image_size"]),
            )
        )
    elif args.toy_mixture_oracle:
        scheduler_name = "linear" if args.scheduler == "auto" else args.scheduler
        scheduler = make_scheduler(
            scheduler_name, args.timesteps, args.beta_start, args.beta_end
        ).to(device)
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
        scheduler = make_scheduler(
            scheduler_name, args.timesteps, args.beta_start, args.beta_end
        ).to(device)
        data_module = make_data_module(
            args.target_dataset,
            args.target_data_dir,
            args.target_image_size,
            args.target_download,
            args.target_toy_size,
            args.target_reference_seed,
        )
        reference_dataset = data_module.dataset()
        target = EmpiricalNoisedDataTarget.from_dataset(
            reference_dataset,
            num_reference_samples=args.target_reference_count,
            seed=args.target_reference_seed,
            reference_chunk_size=args.target_reference_chunk_size,
        )
        model = make_model("target-score", target=target, scheduler=scheduler).to(
            device
        )
        sample_shape = tuple(data_module.sample_shape)
    else:
        model_id = args.pretrained_model_id or DEFAULT_PRETRAINED_MODEL
        model = make_model(
            "huggingface-unet",
            pretrained_model_id=model_id,
            revision=args.pretrained_revision,
            subfolder=args.pretrained_unet_subfolder,
        ).to(device)
        scheduler_name = "huggingface" if args.scheduler == "auto" else args.scheduler
        scheduler = make_scheduler(
            scheduler_name,
            args.timesteps,
            args.beta_start,
            args.beta_end,
            model_id,
            args.pretrained_revision,
            args.pretrained_scheduler_subfolder,
        ).to(device)
        sample_shape = tuple(model.sample_shape)

    model.eval()
    sampler = make_sampler(args.sampler, args.eta)

    modifiers = []
    if args.birth_death:
        if target is None:
            raise ValueError(
                "--birth-death requires a known target (e.g. --empirical-data-target or --toy-mixture-oracle)"
            )
        modifiers.append(
            ReducedKDEBirthDeathAcceleration(
                target,
                args.bd_kernel_bandwidth,
                kde_reduce=args.kde_reduce,
                every_k_steps=args.bd_every_k,
                total_time=args.bd_total_time,
                backbone=args.cnn_backbone,
                projection_method=args.projection_method,
                noise_references=args.noise_references,
                seed=args.seed or 42,
            )
        )

    pipeline = SamplingPipeline(sampler, scheduler, modifiers=modifiers)

    generator = (
        torch.Generator(device=device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    modifier_generator = torch.Generator(device=device)
    if args.seed is not None:
        modifier_generator.manual_seed(args.seed + 100_003)

    sampler_steps = len(sampler.inference_timesteps(scheduler, args.steps, device))
    shape = (args.count, *sample_shape)

    print(
        f"[+] Sampling {args.count} particles with {args.sampler} ({sampler_steps} steps, eta={args.eta})..."
    )
    samples, metrics = measure_sampling(
        model,
        device,
        args.count,
        sampler_steps,
        lambda: pipeline.sample(
            model, shape, device, args.steps, generator, modifier_generator
        ),
    )

    flat = samples.reshape(samples.shape[0], -1)
    unique_ratio = float(torch.unique(flat, dim=0).shape[0] / samples.shape[0])
    print(
        f"[+] Completed: elapsed={metrics.elapsed_seconds:.3f}s | "
        f"sec/sample={metrics.seconds_per_sample:.4f} | "
        f"unique_fraction={unique_ratio:.4f}"
    )

    # Save output preview if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(sample_shape) == 3:
            grid = (samples.clamp(-1.0, 1.0) + 1.0) * 0.5
            save_image(grid, out_path, nrow=args.nrow)
            print(f"[+] Saved preview image grid to {out_path}")
        else:
            torch.save(samples.cpu(), out_path)
            print(f"[+] Saved vector tensor to {out_path}")

    # Save metrics JSON if requested
    if args.metrics_output:
        metrics_dict = {
            "nfe": metrics.nfe,
            "sampler_steps": metrics.sampler_steps,
            "sample_count": metrics.sample_count,
            "elapsed_seconds": metrics.elapsed_seconds,
            "seconds_per_sample": metrics.seconds_per_sample,
            "unique_fraction": unique_ratio,
            "kde_reduce": args.kde_reduce,
            "cnn_backbone": args.cnn_backbone,
            "projection_method": args.projection_method,
            "birth_death": args.birth_death,
        }
        write_metrics(args.metrics_output, metrics_dict, {})
        print(f"[+] Saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
