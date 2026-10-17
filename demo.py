"""
Quick Interactive Demo / Verification Script for Deliverable 1.
Run this script to immediately test the baseline diffusion model and samplers:
    python demo.py
"""

import sys
import os
import torch

from evaluate_baseline import run_baseline_evaluation
import argparse

def main():
    print("""
    ===================================================================
     IE 643 Course Project: Birth-Death Process Accelerated Diffusion
     Deliverable 1: Diffusion Baseline Sampling Pipeline Demo
    ===================================================================
    """)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Detected computation device: {device.upper()}")

    # Set parameters for a fast, responsive demo
    args = argparse.Namespace(
        dataset="8gaussians",
        timesteps=500,           # Baseline T=500 for rapid demo verification
        num_samples=1500,
        schedule="cosine",
        hidden_dim=128,
        num_layers=4,
        device=device,
        auto_train_epochs=15,   # Fast 15-epoch convergence
        checkpoint_dir="./checkpoints",
        results_dir="./results"
    )

    run_baseline_evaluation(args)
    print("\n[+] Verification successful! All baseline components are fully functional.")
    print("[+] Check './results/' for trajectory and quality plots.\n")

if __name__ == "__main__":
    main()

