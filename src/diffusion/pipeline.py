"""Sampling orchestration: sampler updates composed with optional modifiers."""

import torch
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.samplers.base import Sampler
from diffusion.schedulers.base import NoiseScheduler


class SamplingPipeline:
    def __init__(self, sampler: Sampler, scheduler: NoiseScheduler,
                 modifiers: list[SamplingModifier] | None = None):
        self.sampler = sampler
        self.scheduler = scheduler
        self.modifiers = list(modifiers or [])

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple[int, ...], device: torch.device,
               num_inference_steps: int | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
        model.eval()
        self.scheduler.to(device)
        timesteps = self.sampler.inference_timesteps(self.scheduler, num_inference_steps, device)
        samples = torch.randn(shape, device=device, generator=generator)
        for index, timestep_tensor in enumerate(timesteps):
            timestep = int(timestep_tensor.item())
            previous = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            samples = self.sampler.step(model, samples, timestep, previous, self.scheduler, generator)
            state = SamplingState(samples, timestep, previous)
            for modifier in self.modifiers:
                state = modifier.apply(state, model=model, scheduler=self.scheduler, generator=generator)
            samples = state.samples
        return samples.clamp(-1, 1)
