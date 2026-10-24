"""Low-overhead metrics collected during a sampling run."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class SamplingMetrics:
    """Measurements for one call to ``SamplingPipeline.sample``.

    ``nfe`` counts top-level denoiser calls per sampling batch/trajectory. For
    ordinary DDPM and DDIM sampling this equals the number of sampler steps.
    ``model_forward_calls`` records the total calls observed across all batches.
    Calls made by sampling-time modifiers are included in both counts.
    """

    nfe: int
    sampler_steps: int
    sample_count: int
    elapsed_seconds: float
    model_forward_calls: int
    sampling_batches: int = 1

    @property
    def seconds_per_sample(self) -> float:
        return self.elapsed_seconds / self.sample_count if self.sample_count else 0.0

    @property
    def samples_per_second(self) -> float:
        return self.sample_count / self.elapsed_seconds if self.elapsed_seconds else 0.0

    @property
    def model_calls_per_second(self) -> float:
        return self.model_forward_calls / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def to_dict(self) -> dict[str, int | float]:
        values = asdict(self)
        values.update({
            "seconds_per_sample": self.seconds_per_sample,
            "samples_per_second": self.samples_per_second,
            "model_calls_per_second": self.model_calls_per_second,
        })
        return values


def _synchronize(device: torch.device) -> None:
    # CUDA work is asynchronous, so synchronize to include its time in the
    # wall-clock measurement. Other devices do not need this CUDA-specific call.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_sampling(model: nn.Module, device: torch.device, sample_count: int,
                     sampler_steps: int, run: Callable[[], torch.Tensor]
                     ) -> tuple[torch.Tensor, SamplingMetrics]:
    """Run sampling while counting model calls and elapsed wall-clock time."""
    call_count = 0

    def count_model_call(_module, _inputs, _output):
        nonlocal call_count
        call_count += 1

    hook = model.register_forward_hook(count_model_call)
    try:
        _synchronize(device)
        start = time.perf_counter()
        samples = run()
        _synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        hook.remove()

    return samples, SamplingMetrics(
        nfe=call_count,
        sampler_steps=sampler_steps,
        sample_count=sample_count,
        elapsed_seconds=elapsed,
        model_forward_calls=call_count,
    )


def write_metrics(path: str | Path, metrics: SamplingMetrics,
                  metadata: dict | None = None,
                  additional_metrics: dict | None = None) -> None:
    """Write measurements and run metadata as a JSON sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = metrics.to_dict()
    if additional_metrics:
        record.update(additional_metrics)
    if metadata:
        record["run"] = metadata
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
