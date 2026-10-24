"""Gaussian-smoothed empirical data distribution under a DDPM forward process."""

import math

import torch
from torch import nn
from torch.utils.data import Dataset

from diffusion.targets.base import TargetDistribution


class EmpiricalNoisedDataTarget(nn.Module, TargetDistribution):
    """Exact density and score of a finite dataset after forward noising.

    The reference tensor has shape ``(references, *sample_shape)`` and contains
    clean examples in the same coordinates used by training (for example,
    images normalized to [-1, 1]). Target methods take normalized noisy states
    ``z = x_t / sqrt(alpha_bar_t)``. In those coordinates the empirical noisy
    marginal is a Gaussian mixture centered on the clean references, with
    standard deviation ``sqrt((1-alpha_bar_t)/alpha_bar_t)``.

    This is the exact density of the finite reference set after DDPM noising;
    its agreement with the underlying data distribution depends on the subset.
    Reference examples are assumed to be available locally and are stored in
    memory once. The chunk size bounds temporary pairwise-distance memory.
    """

    def __init__(self, reference_samples: torch.Tensor, reference_chunk_size: int = 256):
        super().__init__()
        if reference_samples.ndim < 2 or reference_samples.shape[0] < 1:
            raise ValueError("reference_samples must have shape (M, *sample_shape) with M >= 1")
        if reference_chunk_size < 1:
            raise ValueError("reference_chunk_size must be positive")
        if not reference_samples.is_floating_point():
            reference_samples = reference_samples.float()
        if not torch.isfinite(reference_samples).all():
            raise ValueError("reference_samples must contain only finite values")

        self.sample_shape = tuple(reference_samples.shape[1:])
        self.reference_chunk_size = int(reference_chunk_size)
        self.register_buffer("reference_samples", reference_samples.detach().clone())

    @classmethod
    def from_dataset(cls, dataset: Dataset, num_reference_samples: int,
                     seed: int = 0, reference_chunk_size: int = 256):
        """Materialize a reproducible subset from a tensor or ``(tensor, label)`` dataset.

        The dataset must already be accessible to the caller. This method does
        not download, preprocess, or split data; use the project's data-module
        adapters to construct the dataset with the desired preprocessing first.
        """
        if num_reference_samples < 1:
            raise ValueError("num_reference_samples must be positive")
        if len(dataset) < num_reference_samples:
            raise ValueError(
                f"requested {num_reference_samples} references from a dataset of size {len(dataset)}"
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:num_reference_samples].tolist()
        examples = []
        expected_shape = None
        for index in indices:
            item = dataset[index]
            sample = item[0] if isinstance(item, (tuple, list)) else item
            sample = torch.as_tensor(sample)
            if not sample.is_floating_point():
                sample = sample.float()
            # Data must already exist and be preprocessed by its data module;
            # the target deliberately does not download or rescale examples.
            if expected_shape is None:
                expected_shape = tuple(sample.shape)
            elif tuple(sample.shape) != expected_shape:
                raise ValueError("all reference examples must have the same sample shape")
            examples.append(sample)
        return cls(torch.stack(examples), reference_chunk_size=reference_chunk_size)

    def _flatten_inputs(self, samples: torch.Tensor):
        if samples.ndim < 2 or samples.shape[0] < 1:
            raise ValueError("samples must have shape (batch, *sample_shape) with batch >= 1")
        if not samples.is_floating_point():
            raise ValueError("samples must be floating point tensors")
        if tuple(samples.shape[1:]) != self.sample_shape:
            raise ValueError(
                f"expected samples shaped (batch, {self.sample_shape}), "
                f"got (batch, {tuple(samples.shape[1:])})"
            )
        # Float32 accumulation keeps log-sum-exp stable for half-precision
        # image models; float64 inputs retain their precision.
        compute_dtype = torch.float64 if samples.dtype == torch.float64 else torch.float32
        flat = samples.reshape(samples.shape[0], -1).to(dtype=compute_dtype)
        references = self.reference_samples.reshape(self.reference_samples.shape[0], -1).to(
            device=samples.device, dtype=compute_dtype
        )
        return flat, references

    def noise_std(self, timestep: int, scheduler, *, device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
        if not 0 <= timestep < scheduler.num_train_timesteps:
            raise ValueError("timestep is outside the scheduler's training range")
        alpha_bar = scheduler.alpha_bars[timestep].to(device=device, dtype=dtype)
        if not 0 < alpha_bar < 1:
            raise ValueError("scheduler alpha_bar must be strictly between zero and one")
        return ((1 - alpha_bar) / alpha_bar).clamp_min(0).sqrt()

    def _log_kernel_sum(self, flat: torch.Tensor, references: torch.Tensor,
                        variance: torch.Tensor) -> torch.Tensor:
        running_logsumexp = None
        for start in range(0, references.shape[0], self.reference_chunk_size):
            reference_chunk = references[start:start + self.reference_chunk_size]
            # Direct pairwise distances avoid cancellation from ||x||^2 +
            # ||y||^2 - 2<x,y>, which matters when low-noise image samples
            # are close to a reference in a high-dimensional space.
            distances_squared = torch.cdist(
                flat, reference_chunk, p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).square()
            logits = -0.5 * distances_squared / variance
            chunk_logsumexp = torch.logsumexp(logits, dim=1)
            running_logsumexp = (chunk_logsumexp if running_logsumexp is None else
                                 torch.logaddexp(running_logsumexp, chunk_logsumexp))
        return running_logsumexp

    def log_prob(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        flat, references = self._flatten_inputs(samples)
        sigma = self.noise_std(timestep, scheduler, device=flat.device, dtype=flat.dtype)
        variance = sigma.square()
        log_kernel_sum = self._log_kernel_sum(flat, references, variance)
        dimensions = flat.shape[1]
        log_normalizer = 0.5 * dimensions * torch.log(2 * math.pi * variance)
        log_reference_weight = math.log(references.shape[0])
        return log_kernel_sum - log_normalizer - log_reference_weight

    def score(self, samples: torch.Tensor, timestep: int, scheduler) -> torch.Tensor:
        flat, references = self._flatten_inputs(samples)
        sigma = self.noise_std(timestep, scheduler, device=flat.device, dtype=flat.dtype)
        variance = sigma.square()
        log_kernel_sum = self._log_kernel_sum(flat, references, variance)

        weighted_reference_sum = torch.zeros_like(flat)
        for start in range(0, references.shape[0], self.reference_chunk_size):
            reference_chunk = references[start:start + self.reference_chunk_size]
            distances_squared = torch.cdist(
                flat, reference_chunk, p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            ).square()
            logits = -0.5 * distances_squared / variance
            responsibilities = torch.exp(logits - log_kernel_sum.unsqueeze(1))
            weighted_reference_sum += responsibilities @ reference_chunk

        score = (weighted_reference_sum - flat) / variance
        return score.reshape_as(samples).to(dtype=samples.dtype)
