"""Argument parsers backing the root-level train.py and sample.py scripts."""

import argparse
import math
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.datasets import make_data_module
from diffusion.engine import load_checkpoint, save_checkpoint, train_epoch
from diffusion.models import make_model
from diffusion.pipeline import SamplingPipeline
from diffusion.samplers import make_sampler
from diffusion.schedulers import make_scheduler


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
    parser.add_argument("--pretrained-revision", default=None)
    parser.add_argument("--pretrained-unet-subfolder", default=None)
    parser.add_argument("--pretrained-scheduler-subfolder", default=None)
    parser.add_argument("--scheduler", default="auto",
                        help="auto (repository schedule), linear, or cosine for direct pretrained sampling")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--sampler", default="ddpm")
    parser.add_argument("--steps", type=int, default=None,
                        help="inference steps; DDIM can use fewer than training timesteps")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity; zero is deterministic")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="defaults to a PNG grid for images or a PT tensor for vectors")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    device = _device(args.device)
    if args.checkpoint:
        model, scheduler, config, _ = load_checkpoint(args.checkpoint, device)
        sample_shape = tuple(config.get("sample_shape", (config["channels"],
                              config["image_size"], config["image_size"])))
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
    sampler = make_sampler(args.sampler, args.eta)
    pipeline = SamplingPipeline(sampler, scheduler)
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    shape = (args.count, *sample_shape)
    start = time.perf_counter()
    samples = pipeline.sample(model, shape, device, args.steps, generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    output = Path(args.output) if args.output else Path(
        "samples/sample.png" if samples.ndim == 4 else "samples/samples.pt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if samples.ndim == 4:
        save_image((samples + 1) / 2, output, nrow=args.nrow)
    else:
        torch.save(samples.cpu(), output)
    print(f"saved {samples.shape[0]} samples to {output}; sampler={args.sampler}; "
          f"steps={args.steps or scheduler.num_train_timesteps}; seconds={elapsed:.2f}")
