"""Fixed-population birth-death particle updates for known target densities."""

import torch

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


class BirthDeathAcceleration(SamplingModifier):
    """Apply a KDE-based birth-death step to a fixed-size particle population.

    The implementation follows the population correction in Lu, Lu, and Nolen:
    ``beta_i = log KDE_h(x_i) - log pi(x_i)``, centered over the population.
    Positive rates kill particle ``i`` and duplicate a uniformly chosen other
    particle; negative rates duplicate ``i`` and kill a uniformly chosen other
    particle. A frozen rate is converted to an event probability over a share
    of the configured total birth-death pseudo-time.

    For DDPM/DDIM, this is an exploratory discrete splitting: evaluate the
    target marginal at the destination noise level after a reverse update,
    then resample before the next reverse update. It requires a target density
    and is not implied by the stationary-target convergence result in the
    original Langevin paper.
    """

    def __init__(self, target: TargetDistribution, kernel_bandwidth: float = 0.3,
                 *, every_k_steps: int = 1, total_time: float = 1.0):
        if kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if total_time <= 0:
            raise ValueError("total_time must be positive")
        self.target = target
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)

    def _target_coordinates(self, samples: torch.Tensor, context: SamplingContext,
                             scheduler) -> torch.Tensor:
        target_coordinates = getattr(self.target, "state_coordinates", None)
        if target_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError(
                "birth-death target.state_coordinates must be 'x_t' or 'normalized_x_t'"
            )
        if context.state_coordinates == target_coordinates:
            return samples
        if context.state_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError(
                f"cannot evaluate birth-death at state coordinates {context.state_coordinates!r}"
            )
        timestep = context.state_timestep
        if timestep is None:
            raise ValueError("noisy-coordinate conversion requires a state timestep")
        alpha_bar = scheduler.alpha_bars[timestep].to(samples)
        scale = alpha_bar.sqrt()
        if context.state_coordinates == "x_t":
            # x_t = sqrt(alpha_bar_t) * z. The current targets use z.
            return samples / scale
        return samples * scale

    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        samples = state.samples
        context = state.context
        if context is None:
            raise ValueError("birth-death requires the sampler's post-update SamplingContext")
        # The final DDPM/DDIM update returns clean x_0. Current target
        # distributions describe noisy marginals, so there is no BD call there.
        if (context.is_clean_sample or context.state_timestep is None or
                context.state_coordinates == "clean_x0"):
            return state

        if not 0 <= context.step_index < context.num_steps:
            raise ValueError("SamplingContext has an invalid trajectory position")
        # Apply after every k completed sampler updates. An incomplete tail is
        # intentionally left alone; the total pseudo-time is divided only
        # among the interventions that will actually run.
        if (context.step_index + 1) % self.every_k_steps != 0:
            return state

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = eligible_updates // self.every_k_steps
        if intervention_count < 1:
            return state
        delta_tau = self.total_time / intervention_count

        population = samples.shape[0]
        if population < 2:
            raise ValueError("birth-death updates require at least two particles")
        timestep = context.state_timestep
        target_samples = self._target_coordinates(samples, context, scheduler)

        flat = target_samples.reshape(population, -1)
        distances_squared = torch.cdist(flat, flat).square()
        log_kernel = -distances_squared / (2 * self.kernel_bandwidth ** 2)
        # The Gaussian kernel's normalizing constant cancels when beta is
        # centered, so only the log-sum-exp and population normalization matter.
        log_kde = torch.logsumexp(log_kernel, dim=1) - torch.log(
            target_samples.new_tensor(float(population))
        )
        log_target = self.target.log_prob(target_samples, int(timestep), scheduler)
        if log_target.shape != (population,):
            raise ValueError("target.log_prob must return one log density per particle")
        beta = log_kde - log_target
        beta = beta - beta.mean()

        event_probability = -torch.expm1(-beta.abs() * delta_tau)
        uniforms = torch.rand(population, device=samples.device, generator=generator)
        selected = uniforms < event_probability
        partners = torch.randint(population - 1, (population,), device=samples.device,
                                 generator=generator)
        # Bring the compact event decisions to the host once, then follow the
        # paper's sequential particle loop without per-particle device syncs.
        event_data = torch.stack((
            selected.to(torch.float32),
            beta.gt(0).to(torch.float32),
            partners.to(torch.float32),
        ), dim=1).cpu().tolist()
        source_indices = list(range(population))
        births = 0
        deaths = 0
        for index, (is_selected, is_overrepresented, raw_partner) in enumerate(event_data):
            if not is_selected:
                continue
            # Draw uniformly from all other particle positions.
            partner = int(raw_partner)
            if partner >= index:
                partner += 1
            if is_overrepresented:
                # An overrepresented particle is killed; the partner branches.
                source_indices[index] = source_indices[partner]
                deaths += 1
            else:
                # An underrepresented particle branches; the partner is killed.
                source_indices[partner] = source_indices[index]
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )
        metadata = dict(state.metadata)
        metadata.update({
            "birth_death_events": births + deaths,
            "birth_death_births": births,
            "birth_death_deaths": deaths,
            "birth_death_population": population,
        })
        return SamplingState(
            updated, state.timestep, state.previous_timestep, metadata, context=state.context
        )
