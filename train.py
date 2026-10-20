"""Train the baseline with ``python train.py [options]``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from diffusion.cli import train_main


if __name__ == "__main__":
    train_main()
