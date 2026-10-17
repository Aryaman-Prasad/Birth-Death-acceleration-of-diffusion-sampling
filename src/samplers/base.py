"""
Base abstract class for all diffusion samplers.
Provides a unified interface, timing instruments, NFE counters,
and trajectory recording for baseline and accelerated sampling frameworks.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from src.diffusion.ddpm import GaussianDiffusion


class BaseSampler(ABC):
    """
    Abstract base sampler for diffusion models.
    Every sampler implements `sample()` and tracks inference statistics:
    - total wall-clock latency
    - latency per step
    - number of function evaluations (NFE)
    - trajectory path length
    """
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        self.model = diffusion.model
        self.device = diffusion.device
        self.nfe = 0

    def reset_metrics(self):
        self.nfe = 0

    def evaluate_model(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Wraps model evaluation to track Number of Function Evaluations (NFE).
        """
        self.nfe += 1
        return self.model(x, t)

    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        return_intermediates: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate samples from pure noise x_T ~ N(0, I).
        
        Args:
            shape: Shape of the generated tensor, e.g. (batch_size, channels, H, W) or (batch_size, 2)
            return_intermediates: If True, record states along the reverse trajectory
            verbose: If True, display progress bar
            
        Returns:
            samples: Final generated tensor x_0 of specified shape
            info: Dictionary containing timing, NFEs, steps, and intermediate states
        """
        pass

