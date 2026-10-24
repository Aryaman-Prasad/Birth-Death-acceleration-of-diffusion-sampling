# Birth-death sampling implementation notes

This document describes the project's current implementation of the birth-death (BD) correction inspired by Lu, Lu, and Nolen, [“Accelerating Langevin Sampling with Birth-death”](https://arxiv.org/abs/1905.09863). It is a sampling-time particle-population update and is never part of training.

## What is implemented

The method lives in `src/diffusion/acceleration/birth_death.py` as a `SamplingModifier`. It runs after each sampler update and does not modify DDPM, DDIM, or Langevin sampler code. It is wired through `sample.py` for fixed-noise, unadjusted Langevin updates (`--sampler langevin`) with either the analytic toy mixture (`ToyGaussianMixtureTarget` and `ToyMixtureOracleModel`) or an empirical noised-data mixture (`EmpiricalNoisedDataTarget` and `TargetScoreModel`). Each pairing has a matching target density and score.

The fixed population size is preserved at every modifier call. For each particle, one potential event is sampled using its centered birth/death rate. A death event replaces that particle with a uniformly selected other particle; a birth event replaces a uniformly selected other particle with a copy of the current particle. Consequently events alter particle ancestry and multiplicity while the batch still contains the same number of particles.

## Rate and event rule

For particle locations \(x_1,\dots,x_N\), estimate the current population density at each location with a Gaussian kernel estimate

\[
\widehat q_h(x_i) = \frac{1}{N}\sum_{j=1}^N K_h(x_i-x_j).
\]

For a target density \(\pi\), calculate

\[
\beta_i = \log \widehat q_h(x_i) - \log \pi(x_i), \qquad
\bar\beta_i = \beta_i - \frac{1}{N}\sum_{j=1}^N\beta_j.
\]

The implementation omits the Gaussian kernel's normalizing constant because it is common to every \(\log \widehat q_h(x_i)\) and cancels when the rates are centered. If \(\bar\beta_i>0\), the empirical population is relatively overrepresented at \(x_i\), so particle \(i\) is selected for death. If \(\bar\beta_i<0\), particle \(i\) is selected for birth. The rate magnitude is \(|\bar\beta_i|\). Over a Langevin step of size \(\Delta t\), the code samples an event with probability

\[
p_i = 1-\exp(-|\bar\beta_i|\Delta t).
\]

This is the probability of at least one event for a frozen rate over a finite interval; for small \(\Delta t\), it agrees with the rate-times-step approximation. Selected events are applied sequentially in particle-index order. Event decisions and partner draws are sampled independently before the loop, and partners are uniformly drawn from the other particle positions. The KDE and rates are computed once at the start of each modifier call, before those within-call replacements.

## Why the toy target is a useful validation case

The target is a mixture of eight isotropic Gaussians arranged around a ring. Its density and score are analytically available. Under the DDPM forward process, the normalized variable \(z=x_t/\sqrt{\bar\alpha_t}\) is the clean mixture convolved with Gaussian noise of standard deviation

\[
\sigma_t = \sqrt{(1-\bar\alpha_t)/\bar\alpha_t}.
\]

The target module evaluates the exact noisy-mixture density and score in these normalized coordinates. The oracle denoiser converts that score to the corresponding epsilon prediction, and fixed-target ULA uses the inverse relation to recover the score. This makes both the target density in the BD rate and the Langevin score known exactly, avoiding a guessed density ratio for image models.

Run a baseline ULA population and the same setup with BD enabled:

```bash
python3 sample.py --toy-mixture-oracle --sampler langevin --steps 1000 --count 256 --seed 7 --output samples/mixture-ula.pt
python3 sample.py --toy-mixture-oracle --sampler langevin --steps 1000 --count 256 --birth-death --bd-kernel-bandwidth 0.3 --seed 7 --output samples/mixture-bd-ula.pt
```

The vectors are saved as PyTorch tensors. The sampler's fixed noise level defaults to timestep zero; `--langevin-timestep` can choose another scheduler level. The KDE bandwidth is measured in normalized target coordinates. The population and step count should be kept the same when comparing runs. Use different seeds and settings to assess variability and bandwidth/step-size sensitivity.

## Empirical target from a dataset subset

`EmpiricalNoisedDataTarget` gives a precise meaning to a subset-based density estimate. Given preprocessed reference examples `x_0^(1), ..., x_0^(M)`, the forward process makes the normalized state `z = x_t / sqrt(alpha_bar_t)` a Gaussian mixture:

\[
\widehat p_t(z)=\frac{1}{M}\sum_{j=1}^M \mathcal N\!\left(z; x_0^{(j)}, \sigma_t^2 I\right),\qquad
\sigma_t^2=\frac{1-\bar\alpha_t}{\bar\alpha_t}.
\]

The implementation evaluates the normalized log density with `logsumexp` and also evaluates its analytic score. `TargetScoreModel` converts that score to the epsilon prediction expected by the existing DDPM, DDIM, and Langevin samplers. Therefore, when this target is selected, the sampler score and the density used by birth-death refer to the same finite-data distribution. Subset selection is uniform and reproducible by seed; examples come through the dataset adapter's preprocessing (including image normalization). The target assumes the data is already available locally; pass `--target-download` only when you want CIFAR-10 fetched if it is missing.

The mixture's smoothing scale `sigma_t` is fixed by the diffusion scheduler and timestep. It is separate from `--bd-kernel-bandwidth`, which controls the KDE over the current particle population in the birth-death rate.

The empirical mixture is the exact smoothed distribution of the selected finite subset, but it is only a proxy for the unknown full data distribution. A small subset can omit modes, and kernel density estimation becomes statistically difficult in high-dimensional image spaces. The default reference count and chunk size are intended for smoke runs, not as validated CIFAR-10/CelebA density estimates. Evaluation costs O(B M D) arithmetic for batch size B, subset size M, and flattened data dimension D; reference chunking bounds temporary memory, not total work. Increase the subset only when runtime and memory permit, and report the subset, preprocessing, timestep, and KDE bandwidth in experiments.

This mode does not combine a pretrained DDPM score with the subset log density. Those would generally describe different distributions, making the birth-death discrepancy inconsistent. A pretrained score can be useful for sampling, but a compatible log density would still need to be derived or estimated for the same learned distribution.

Run matched ULA baselines with and without birth-death on the built-in vector toy dataset:

```bash
python3 sample.py --empirical-data-target --target-dataset toy-mixture --target-toy-size 512 --target-reference-count 256 --sampler langevin --steps 500 --count 128 --seed 7 --output samples/empirical-ula.pt
python3 sample.py --empirical-data-target --target-dataset toy-mixture --target-toy-size 512 --target-reference-count 256 --sampler langevin --steps 500 --count 128 --birth-death --bd-kernel-bandwidth 0.3 --seed 7 --output samples/empirical-bd-ula.pt
```

For locally available image data, replace `toy-mixture` with `cifar10` or `imagefolder`, and set `--target-data-dir`, `--target-image-size`, and (for CIFAR-10 if required) `--target-download`. Use `--target-reference-count` to choose the subset size, `--target-reference-seed` to choose it reproducibly, and `--target-reference-chunk-size` to bound pairwise density-evaluation memory. The empirical target can also supply scores to DDPM, DDIM, and annealed Langevin without birth-death; the CLI only permits birth-death with fixed-target Langevin because that is the stationary-target setting currently implemented.

## Relationship to the paper and limits

The implementation preserves the key BD construction from the paper: a nonlocal KDE-to-target-density discrepancy determines birth/death rates, and population resampling is composed with Langevin motion. It is therefore an executable finite-population experiment using a known target, rather than a placeholder or a density proxy inferred from the denoiser.

It is not a reproduction of every theoretical or numerical detail from the paper. In particular, this framework uses a discrete operator splitting (one ULA update, then one finite-interval BD update), a fixed population, a Gaussian KDE including the self-kernel term, and a frozen-rate sequential particle loop. The paper's convergence analysis is for its stated continuous-time/mean-field setting; that result does not automatically certify this finite-step implementation. The bandwidth and step size are experimental parameters and should be checked for sensitivity.

BD can improve mixing or convergence of a particle approximation; this implementation does not literally skip reverse diffusion timesteps or reduce the number of Langevin updates. Compare distributional error at matched NFE, or compare how many ULA updates are needed to reach a chosen error. NFE currently counts score-model evaluations; BD itself requires no denoiser calls, although its pairwise KDE is \(O(N^2)\) in population size and contributes to elapsed time.

## What is not implemented yet

- **BD with DDPM, DDIM, or annealed Langevin:** these methods visit a sequence of changing noisy distributions. Applying a stationary-target BD rate would be unjustified without specifying the target density, the state coordinates, and how the BD time increment maps to each diffusion update.
- **BD paired with a learned CIFAR-10/CelebA score model:** an epsilon model estimates a noisy score but does not provide the corresponding target log density needed by the rate. The empirical-data mode is separate: it uses both the subset mixture's score and density, not a pretrained model's score. Combining a learned score and subset density would need a justified consistency argument.
- **Variable-size particle populations, GPU-optimized KDE, and dedicated BD diagnostics/benchmarks:** the current implementation intentionally uses fixed population and a transparent dense pairwise KDE suitable for small 2D experiments.
- **A paper-exact reproduction:** integrator conventions, event scheduling, model parameters, and benchmark protocol still need to be aligned if the goal becomes reproducing the paper's quantitative experiments.

The command-line guard permits this experimental BD implementation with either the analytic toy oracle or the empirical noised-data target, and only with the fixed-target `langevin` sampler. The `SamplingModifier` interface remains available for future methods, but a new target/sampler pairing should document and justify its state-space density and event-time interpretation first.
