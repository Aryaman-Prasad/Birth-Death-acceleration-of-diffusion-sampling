"""Sample from a checkpoint with ``python sample.py [options]``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from diffusion.cli import sample_main


if __name__ == "__main__":
    sample_main()
