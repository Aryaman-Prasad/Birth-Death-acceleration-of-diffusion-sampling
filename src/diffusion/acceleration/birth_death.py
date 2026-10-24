"""Fixed-population birth-death particle updates for known target densities."""

import torch

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.targets.base import TargetDistribution


class BirthDeathAcceleration(SamplingModifier):
    """Apply a KDE-based birth-death step to a fixed-size particle population.

    The implementation follows the population correction in Lu, Lu, and Nolen:
    ``beta_i = log KDE_h(x_i) - log pi(x_i)``, centered over the population.
    Positive rates kill particle ``i`` and duplicate a uniformly chosen other
    particle; negative rates duplicate ``i`` and kill a uniformly chosen other
    particle. A frozen rate is converted to an event probability over the
    supplied Langevin step ``dt``.

    This requires a stationary target with evaluable log density. It is not
    enabled for time-varying DDPM/DDIM trajectories or learned image models.
    """

    def __init__(self, target: TargetDistribution, kernel_bandwidth: float = 0.3):
        if kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")
        self.target = target
        self.kernel_bandwidth = float(kernel_bandwidth)

    def apply(self, state: SamplingState, *, model, scheduler, generator) -> SamplingState:
        samples = state.samples
        population = samples.shape[0]
        if population < 2:
            raise ValueError("birth-death updates require at least two particles")
        if "birth_death_delta_t" not in state.metadata:
            raise ValueError(
                "birth-death requires a fixed-target Langevin sampler that supplies a step size"
            )
        if state.metadata.get("state_coordinates") != "normalized_x_t":
            raise ValueError("birth-death target and particle coordinates are incompatible")
        timestep = state.metadata.get("target_timestep")
        if timestep is None:
            raise ValueError("birth-death requires the target timestep in sampling metadata")
        delta_t = float(state.metadata["birth_death_delta_t"])
        if delta_t <= 0:
            raise ValueError("birth-death delta_t must be positive")

        flat = samples.reshape(population, -1)
        distances_squared = torch.cdist(flat, flat).square()
        log_kernel = -distances_squared / (2 * self.kernel_bandwidth ** 2)
        # The Gaussian kernel's normalizing constant cancels when beta is
        # centered, so only the log-sum-exp and population normalization matter.
        log_kde = torch.logsumexp(log_kernel, dim=1) - torch.log(
            samples.new_tensor(float(population))
        )
        log_target = self.target.log_prob(samples, int(timestep), scheduler)
        if log_target.shape != (population,):
            raise ValueError("target.log_prob must return one log density per particle")
        beta = log_kde - log_target
        beta = beta - beta.mean()

        event_probability = -torch.expm1(-beta.abs() * delta_t)
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
        return SamplingState(updated, state.timestep, state.previous_timestep, metadata)
