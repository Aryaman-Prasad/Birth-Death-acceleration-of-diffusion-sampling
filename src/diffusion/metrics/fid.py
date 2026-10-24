"""Optional streaming Fréchet Inception Distance evaluator."""

from __future__ import annotations

import time

import torch


class FIDEvaluator:
    """Accumulate real and generated image statistics without storing images.

    Inputs are expected as RGB tensors normalized to ``[-1, 1]``, matching the
    framework's image dataset and sampler conventions. TorchMetrics is imported
    only when this evaluator is constructed, so it remains an optional extra.
    """

    def __init__(self, device: torch.device, feature: int = 2048):
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            metric = FrechetInceptionDistance(feature=feature, normalize=True)
        except ImportError as exc:
            raise RuntimeError(
                "FID requires optional dependencies. Install them with "
                "`python -m pip install -e '.[fid]'`."
            ) from exc

        self.device = device
        self.metric = metric.to(device)
        self.metric.set_dtype(torch.float64)
        self.real_count = 0
        self.fake_count = 0
        self.elapsed_seconds = 0.0

    def _update(self, images: torch.Tensor, real: bool) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("FID requires batches of RGB images with shape (N, 3, H, W)")
        if images.shape[0] == 0:
            return

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        images = ((images.detach().clamp(-1, 1) + 1) / 2).to(device=self.device, dtype=torch.float32, non_blocking=True)
        self.metric.update(images, real=real)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_seconds += time.perf_counter() - start
        if real:
            self.real_count += images.shape[0]
        else:
            self.fake_count += images.shape[0]

    def update_real(self, images: torch.Tensor) -> None:
        """Add a batch from the reference image distribution."""
        self._update(images, real=True)

    def update_fake(self, images: torch.Tensor) -> None:
        """Add a generated image batch."""
        self._update(images, real=False)

    def compute(self) -> float:
        """Return FID after equal-sized real and generated sets were added."""
        if self.real_count < 2 or self.fake_count < 2:
            raise ValueError("FID needs at least two real and two generated images")
        if self.real_count != self.fake_count:
            raise ValueError(
                f"FID needs equal-sized image sets; got {self.real_count} real and "
                f"{self.fake_count} generated images"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        score = float(self.metric.compute().item())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.elapsed_seconds += time.perf_counter() - start
        return score
