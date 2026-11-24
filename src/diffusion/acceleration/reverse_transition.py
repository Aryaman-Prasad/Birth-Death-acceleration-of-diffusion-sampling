"""Shared Gaussian reverse-transition calculations for sampler-aware modifiers."""

import torch
from torch import nn


def reverse_transition_parameters(
    source_samples: torch.Tensor,
    timestep: int,
    previous_timestep: int,
    model: nn.Module,
    scheduler,
    metadata: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reverse transition mean and scalar variance per coordinate.

    The calculations mirror the framework's DDIM and DDPM sampler updates.
    ``source_samples`` is x_t; the returned mean/variance describe the law of
    x_s for s=``previous_timestep``. Stochastic DDIM (eta > 0) and ancestral
    DDPM are supported. Deterministic DDIM has no ordinary density and is
    rejected for likelihood-based weighting/branching.
    """
    sampler_name = metadata.get("sampler_name")
    if sampler_name not in {"ddim", "ddpm"}:
        raise ValueError(
            "reverse-transition extensions currently require a DDIM or DDPM sampler"
        )
    if scheduler.prediction_type != "epsilon":
        raise ValueError("reverse-transition extensions require epsilon prediction")
    if timestep < 0 or timestep >= scheduler.num_train_timesteps:
        raise ValueError("source timestep is outside the scheduler range")
    if previous_timestep >= timestep:
        raise ValueError("reverse transition must move to a lower timestep")

    ts = torch.full(
        (source_samples.shape[0],), timestep,
        device=source_samples.device, dtype=torch.long,
    )
    with torch.no_grad():
        predicted_noise = model(source_samples, ts)

    alpha_bar = scheduler.alpha_bars[timestep].to(source_samples)
    alpha_bar_previous = (
        scheduler.alpha_bars[previous_timestep].to(source_samples)
        if previous_timestep >= 0
        else alpha_bar.new_tensor(1.0)
    )
    predicted_clean = (
        source_samples - (1.0 - alpha_bar).sqrt() * predicted_noise
    ) / alpha_bar.sqrt()
    if getattr(scheduler, "clip_sample", False):
        limit = getattr(scheduler, "clip_sample_range", 1.0)
        predicted_clean = predicted_clean.clamp(-limit, limit)

    if sampler_name == "ddim":
        eta = float(metadata.get("eta", 0.0))
        if eta <= 0:
            raise ValueError(
                "transition-likelihood and branching methods require stochastic DDIM (eta > 0)"
            )
        variance = eta**2 * (
            (1.0 - alpha_bar_previous)
            / (1.0 - alpha_bar).clamp_min(1e-20)
            * (1.0 - alpha_bar / alpha_bar_previous)
        ).clamp_min(0)
        direction = (1.0 - alpha_bar_previous - variance).clamp_min(0).sqrt()
        mean = alpha_bar_previous.sqrt() * predicted_clean + direction * predicted_noise
        return mean, variance.clamp_min(0)

    # DDPMSampler uses the adjacent posterior q(x_{t-1}|x_t,x_0) and does
    # not implement skipped inference timesteps.
    if previous_timestep != timestep - 1:
        raise ValueError("DDPM transition weighting requires adjacent timesteps")
    beta = scheduler.betas[timestep].to(source_samples)
    alpha = scheduler.alphas[timestep].to(source_samples)
    alpha_bar_previous = scheduler.alpha_bars_prev[timestep].to(source_samples)
    mean = (
        beta * alpha_bar_previous.sqrt() / (1.0 - alpha_bar) * predicted_clean
        + (1.0 - alpha_bar_previous) * alpha.sqrt() / (1.0 - alpha_bar)
        * source_samples
    )
    if scheduler.variance_type in {"fixed_large", "fixed_large_log"}:
        variance = beta
    elif scheduler.variance_type in {"fixed_small", "fixed_small_log"}:
        variance = beta * (1.0 - alpha_bar_previous) / (1.0 - alpha_bar)
    else:
        raise ValueError(f"unsupported DDPM variance type: {scheduler.variance_type}")
    return mean, variance.clamp_min(1e-20)
