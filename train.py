"""
Training script for Diffusion Baseline Models.
Supports both 2D multimodal toy benchmarks and image datasets.
Includes Exponential Moving Average (EMA) and automatic checkpointing.
"""

import argparse
import copy
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.models.toy_mlp import ToyScoreMLP
from src.models.unet import UNet
from src.diffusion.ddpm import GaussianDiffusion
from src.utils.datasets import get_2d_dataloader, get_image_dataloader, Synthetic2DDataset
from src.utils.visualization import plot_2d_distribution_comparison, save_image_grid
from src.samplers.ddpm_sampler import DDPMSampler
from src.samplers.ddim_sampler import DDIMSampler


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, beta: float = 0.9999):
        self.beta = beta
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.data.mul_(self.beta).add_(param.data, alpha=1.0 - self.beta)


def train_2d(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Training 2D Diffusion Model on '{args.dataset}' using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # 1. Dataset
    dataloader = get_2d_dataloader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed
    )

    # 2. Model & Diffusion
    score_model = ToyScoreMLP(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        time_emb_dim=64
    ).to(device)

    diffusion = GaussianDiffusion(
        model=score_model,
        timesteps=args.timesteps,
        beta_schedule=args.schedule,
        loss_type="mse"
    ).to(device)

    ema = EMA(score_model)
    optimizer = AdamW(score_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"[*] Total trainable parameters: {sum(p.numel() for p in score_model.parameters()):,}")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        score_model.train()
        total_loss = 0.0
        batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss = diffusion(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), 1.0)
            optimizer.step()
            ema.update(score_model)

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / batches

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Avg Loss: {avg_loss:.5f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Quick validation sample save
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            ema.shadow.eval()
            eval_diffusion = GaussianDiffusion(
                model=ema.shadow,
                timesteps=args.timesteps,
                beta_schedule=args.schedule
            ).to(device)

            ddim = DDIMSampler(eval_diffusion)
            samples, _ = ddim.sample(shape=(2000, 2), num_inference_steps=50, eta=0.0)

            gt_data = Synthetic2DDataset(name=args.dataset, num_samples=2000).data
            sample_path = os.path.join(args.results_dir, f"sample_epoch_{epoch:03d}.png")
            plot_2d_distribution_comparison(
                ground_truth=gt_data,
                generated_dict={"DDIM-50": samples},
                save_path=sample_path,
                title=f"Sample Progress (Epoch {epoch})"
            )

    train_time = time.time() - start_time
    print(f"[+] Training completed in {train_time:.2f}s")

    # Save model checkpoints
    ckpt_path = os.path.join(args.checkpoint_dir, f"ddpm_{args.dataset}.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state": score_model.state_dict(),
        "ema_state": ema.shadow.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": vars(args)
    }, ckpt_path)
    print(f"[+] Saved model checkpoint to: {ckpt_path}")


def train_image(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Training Image Diffusion Model on '{args.dataset}' using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    in_channels = 3 if args.dataset == "cifar10" else 1
    image_size = 32 if args.dataset == "cifar10" else 28

    dataloader = get_image_dataloader(
        dataset_name=args.dataset,
        data_dir="./data",
        batch_size=args.batch_size,
        image_size=image_size
    )

    model = UNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
        channel_multipliers=[1, 2, 2],
        attention_resolutions=[1],
        num_res_blocks=2
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=args.timesteps,
        beta_schedule=args.schedule
    ).to(device)

    ema = EMA(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"[*] Total trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in pbar:
            images = images.to(device)
            optimizer.zero_grad()
            loss = diffusion(images)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            total_loss += loss.item()
            batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / batches
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Avg Loss: {avg_loss:.5f}")

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            ema.shadow.eval()
            eval_diffusion = GaussianDiffusion(model=ema.shadow, timesteps=args.timesteps).to(device)
            ddim = DDIMSampler(eval_diffusion)
            samples, _ = ddim.sample(shape=(16, in_channels, image_size, image_size), num_inference_steps=20)
            sample_path = os.path.join(args.results_dir, f"sample_image_epoch_{epoch:03d}.png")
            save_image_grid(samples, sample_path, nrow=4)

    ckpt_path = os.path.join(args.checkpoint_dir, f"ddpm_image_{args.dataset}.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state": model.state_dict(),
        "ema_state": ema.shadow.state_dict(),
        "config": vars(args)
    }, ckpt_path)
    print(f"[+] Saved image model checkpoint to: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Model Baseline")
    parser.add_argument("--mode", type=str, choices=["2d", "image"], default="2d", help="Training mode")
    parser.add_argument("--dataset", type=str, default="8gaussians", help="Dataset name: 8gaussians, 25gaussians, swissroll, mnist")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=1000, help="Total diffusion timesteps T")
    parser.add_argument("--schedule", type=str, default="cosine", choices=["linear", "cosine", "sigmoid"])
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for 2D MLP")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of residual blocks for 2D MLP")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel count for U-Net")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of dataset samples")
    parser.add_argument("--device", type=str, default="cuda", help="Target device: cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=5, help="Logging frequency in epochs")
    parser.add_argument("--eval_interval", type=int, default=10, help="Evaluation sample saving frequency")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save generated visual results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.mode == "2d":
        train_2d(args)
    else:
        train_image(args)


if __name__ == "__main__":
    main()

