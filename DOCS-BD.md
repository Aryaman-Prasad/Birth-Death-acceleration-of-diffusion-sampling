# Birth-death sampling implementation notes

This document records the birth-death (BD) implementation inspired by Lu, Lu, and Nolen, [Accelerating Langevin Sampling with Birth-death](https://arxiv.org/abs/1905.09863). BD runs only during generation. It is a particle-population modifier that can be composed with sampler updates; it is never part of model training.

## Current implementation

The implementation is in `src/diffusion/acceleration/birth_death.py`. It keeps the particle count fixed and applies a KDE-based population correction after selected sampler updates. The command line supports these pairings:

| Score/model and target | Samplers |
| --- | --- |
| Analytic Gaussian-mixture oracle and its exact noisy density | DDPM, DDIM, fixed-target Langevin (ULA) |
| Score and exact noisy density of a finite empirical data subset | DDPM, DDIM, fixed-target Langevin (ULA) |

The empirical target can be built from the toy datasets, CIFAR-10, or an image folder (for example, a locally available CelebA directory) through the existing dataset interface. BD is not enabled with a standalone pretrained or checkpoint model: those models do not provide a matching target log density. Pairing their learned score with an unrelated empirical density would make the rate compare two different distributions. Annealed Langevin is also not enabled in the CLI yet; its level transitions need a separately defined intervention convention.

The population size remains constant at every intervention. An overrepresented particle is killed and replaced with a uniformly selected other particle; an underrepresented particle is duplicated in place of a uniformly selected other particle. This changes particle ancestry and multiplicity while keeping a fixed-size batch that the next sampler update can consume.

## Sampling context and destination state

Each sampler describes the state it returns through `SamplingContext`, which is carried in `SamplingState.context`. The context records the state noise level, coordinate system, whether the state is clean, and the update's position in the full trajectory.

| Sampler | State after an update | Coordinates supplied by the sampler | BD behavior |
| --- | --- | --- | --- |
| DDPM / DDIM | Destination level `s = previous_timestep`; clean `x_0` after the last update | `x_t` for noisy states; `clean_x0` at the endpoint | Convert to target coordinates, then evaluate the target at level `s`; skip clean endpoint |
| Fixed-target Langevin | Same fixed noise level as its score | `normalized_x_t` | Evaluate at its fixed level |
| Annealed Langevin | Current update's noise level | `normalized_x_t` | CLI currently rejects BD for this sampler |

For DDPM and DDIM, the sampler input level is not the density level for BD. The modifier runs after the update, so it uses the destination level represented by the resulting particles. DDIM's skipped transitions use the actual next inference timestep as that destination. The final DDPM/DDIM update returns a clean sample, for which these target implementations do not define a noisy marginal; BD is skipped there.

The pipeline precomputes each update's context and adds `step_index`, `num_steps`, and `has_clean_endpoint`. This lets modifiers share one scheduling convention without embedding birth-death logic in individual samplers.

## Coordinates and target marginals

`TargetDistribution.state_coordinates` declares the coordinates accepted by its `log_prob` and `score` methods. The targets currently implemented use normalized noisy coordinates

\[
z_t = x_t / \sqrt{\bar\alpha_t}.
\]

When a DDPM or DDIM state is supplied as `x_t`, BD converts it to `z_t` using the scheduler's `alpha_bars[t]`. The particle KDE and target density are both evaluated in the target's declared coordinates, at the same destination timestep. Thus the discrepancy compares densities in one coordinate system rather than mixing the sampler's raw state with a normalized-space target.

The analytic toy target is a mixture of eight isotropic Gaussian modes arranged around a ring. Under the forward process, the mixture in normalized coordinates has standard deviation

\[
\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}.
\]

The empirical target uses preprocessed reference examples `x_0^(1), ..., x_0^(M)` from the selected dataset. Its exact finite-subset marginal in normalized coordinates is

\[
\widehat p_t(z)=\frac{1}{M}\sum_{j=1}^M
\mathcal N\!\left(z; x_0^{(j)}, \sigma_t^2 I\right),\qquad
\sigma_t^2=\frac{1-\bar\alpha_t}{\bar\alpha_t}.
\]

The implementation evaluates this density with `logsumexp`; the same target also supplies its analytic score to `TargetScoreModel`. Therefore the empirical-target DDPM/DDIM experiments use a score and a BD density for the same finite-data distribution. Reference selection is uniform and reproducible by seed, and examples use the dataset adapter's preprocessing (including image normalization). Data must be locally available unless CIFAR-10 download is explicitly enabled.

This is exact for the selected finite subset, not necessarily for the full data distribution. Small subsets can omit modes, and estimating densities in high-dimensional image spaces is statistically difficult. The default reference count and chunk size are for smoke runs, not validated CIFAR-10 or CelebA density estimates.

## Rate, intervention cadence, and pseudo-time

For particles `x_1, ..., x_N`, the modifier forms a Gaussian KDE over the current population:

\[
\widehat q_h(x_i)=\frac{1}{N}\sum_{j=1}^N K_h(x_i-x_j).
\]

It then computes the centered discrepancy

\[
\beta_i=\log\widehat q_h(x_i)-\log\pi_t(x_i),\qquad
\bar\beta_i=\beta_i-\frac{1}{N}\sum_{j=1}^N\beta_j.
\]

Positive `\bar\beta_i` marks a particle as relatively overrepresented and negative `\bar\beta_i` as underrepresented. The Gaussian kernel's normalizing constant is omitted because it is common to every KDE log density and cancels when the rates are centered. Rates are evaluated once at the beginning of an intervention, before that intervention's resampling loop.

Use `--bd-every-k K` to intervene after every `K` completed sampler updates. The final clean DDPM/DDIM update is excluded. If a trajectory has `S` updates and ends clean, the number of eligible updates is `S - 1`; otherwise it is `S`. The implementation schedules at update numbers `K, 2K, 3K, ...` among eligible updates. Any incomplete tail is left without an intervention.

`--bd-total-time T` sets the total BD pseudo-time budget for a trajectory. It is divided evenly over the interventions that are scheduled:

\[
\Delta\tau = T / N_{\mathrm{interventions}},\qquad
p_i=1-\exp(-|\bar\beta_i|\Delta\tau).
\]

The default is `T = 1.0`. This pseudo-time is independent of the diffusion scheduler's time and the Langevin sampler's step size. Keeping `T` fixed while changing `--bd-every-k` keeps the aggregate BD time budget fixed, while changing when the corrections occur. If `K` exceeds the number of eligible updates, there are no interventions.

Event decisions and partner draws are sampled independently before applying the selected events sequentially in particle-index order. A partner is uniformly selected from all other particle positions. The implementation uses a fixed-size population and a dense pairwise particle KDE.

## Running experiments

These vector experiments run without external data downloads. DDPM uses all scheduler timesteps; choose a small `--timesteps` for a short toy run. DDIM can use a reduced number of sampler updates through `--steps`.

`tests/toy_bd_experiment.py` is a repeatable end-to-end comparison runner for the analytic toy target. It can run DDPM, DDIM, or both with the same seed, each with and without BD:

```bash
python3 tests/toy_bd_experiment.py --sampler both --count 128 --steps 25 --timesteps 1000 --device cpu
```

It prints NFE, elapsed time, seconds per sample, clean-target negative log likelihood, multi-scale RBF MMD-squared against independent samples from the exact target, mode coverage, and mode-balance total variation. The clean-target NLL uses the analytic mixture at zero added noise; mode coverage counts how many of the eight known modes received at least one generated sample, and mode-balance TV compares nearest-mode frequencies with the uniform target weights. MMD-squared and mode metrics compare sample distributions; NLL is a complementary pointwise diagnostic. It saves generated tensors and a `metrics.json` report in `samples/toy-bd-experiment/` by default. These target-specific values are useful toy diagnostics, not general-purpose image metrics.

```bash
# Analytic toy target, standard DDPM
python3 sample.py --toy-mixture-oracle --timesteps 100 --sampler ddpm --birth-death --count 128 --seed 7 --output samples/mixture-bd-ddpm.pt

# Same target with a shorter DDIM trajectory and an intervention every two updates
python3 sample.py --toy-mixture-oracle --timesteps 100 --sampler ddim --steps 25 --birth-death --bd-every-k 2 --bd-total-time 1.0 --count 128 --seed 7 --output samples/mixture-bd-ddim.pt

# Fixed-target ULA baseline with BD
python3 sample.py --toy-mixture-oracle --sampler langevin --steps 500 --birth-death --count 128 --seed 7 --output samples/mixture-bd-ula.pt
```

For a DDIM run using the finite empirical mixture of a toy dataset:

```bash
python3 sample.py --empirical-data-target --target-dataset toy-mixture --target-toy-size 512 --target-reference-count 256 --sampler ddim --steps 25 --birth-death --bd-every-k 2 --bd-total-time 1.0 --count 128 --seed 7 --output samples/empirical-bd-ddim.pt
```

For local CIFAR-10 data, replace `toy-mixture` with `cifar10` and set `--target-data-dir`. Add `--target-download` only if you want the dataset fetched when absent. For example, a small experimental run could use:

```bash
python3 sample.py --empirical-data-target --target-dataset cifar10 --target-data-dir data --target-image-size 32 --target-reference-count 32 --sampler ddim --steps 25 --birth-death --bd-every-k 2 --count 16 --seed 7 --output samples/cifar-empirical-bd-ddim.png
```

That small subset is only a wiring/runtime smoke configuration, not a reliable CIFAR-10 density estimate. Tune `--target-reference-count`, `--target-reference-chunk-size`, `--bd-kernel-bandwidth`, and `--bd-total-time` for experiments, and report them with results. The KDE bandwidth is measured in the target's coordinates.

Sampling metrics still report sampler steps and denoiser NFE; the birth-death operation makes no denoiser calls. Its work does contribute to elapsed time: the population KDE is `O(N^2 D)` per intervention for population `N` and flattened dimension `D`, and empirical target evaluation is approximately `O(N M D)` for `M` reference samples (with chunking to bound temporary memory).

When `--fid` streams generated images in batches, each generated batch is a separate BD population; keep `--fid-batch-size` at least 2 and avoid a final one-sample remainder (`--fid-samples % --fid-batch-size == 1`). Without streaming FID, `--count` is the population size for each trajectory.

## Relationship to the paper and limitations

The KDE-to-target-density discrepancy and birth/death resampling rule follow the paper's population correction. For fixed-target ULA, the modifier composes that correction with a Langevin update. Applying it after a DDPM/DDIM update instead evaluates a time-dependent noisy target marginal and resamples before the next reverse transition. This is a reasonable experiment for the project's diffusion-sampling objective, but it is a discrete operator-splitting extension: the paper's stationary Langevin convergence analysis does not establish correctness or acceleration for this time-varying reverse process.

BD does not itself remove DDPM/DDIM updates or reduce NFE. It adds population selection between updates. To test acceleration, compare sample quality at matched NFE or measure the number of reverse updates needed to reach a chosen quality, while also reporting wall-clock time because KDE and density evaluation add cost. A changed final distribution is possible, so compare both with and without BD across seeds and settings.

The empirical target is not paired with a standalone pretrained score model in this CLI. A pretrained denoiser's score and an empirical subset density generally describe different distributions. A justified log density for the learned model, or a consistency argument connecting the two, would be needed before combining them.

## Not implemented yet

- BD with annealed Langevin, with a clearly specified convention for when to evaluate its changing noise-level targets.
- Variable-size populations, optimized GPU KDE kernels, and dedicated BD event diagnostics/benchmarks.
- A paper-exact numerical reproduction, including alignment of integrators, event scheduling, model parameters, and benchmark protocol.
- BD with standalone pretrained/checkpoint scores unless a matching target density is available.
