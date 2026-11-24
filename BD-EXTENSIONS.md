# Birth–Death extensions: equations, code paths, and experiments

This document describes the acceleration methods currently implemented in `src/diffusion/acceleration/`, the equations they use, and the experiments available in the repository. The emphasis is on distinguishing genuine birth–death (BD) copy/replace updates from other resampling ideas that are grouped under acceleration.

The implementations are experimental. Equations below describe the code as it runs, and call out where the code is only a heuristic or approximation. In particular, using a BD update inside a reverse diffusion trajectory is a discrete splitting choice; the convergence results for stationary-target Langevin sampling in Lu, Lu, and Nolen do not by themselves prove acceleration or convergence for DDPM/DDIM.

## 1. Pipeline and shared notation

`SamplingPipeline` runs one reverse-sampler update and then calls each configured `SamplingModifier` on the resulting population. The modifier returns a population for the next reverse step. These modifiers run during sampling, not training. A population usually remains at its original size: a “birth” is implemented by copying one particle over another, and a “death” by replacing it with a peer, so the net count stays fixed.

Notation used below:

| Symbol | Meaning |
|---|---|
| `N` | Number of particles/images in the sampling population |
| `M` | Number of reference examples used to estimate an empirical target |
| `x_t[i]` | Particle `i` at diffusion noise level `t` |
| `s < t` | Previous/less-noisy level after a reverse transition from `t` to `s` |
| `bar_alpha_t` | Cumulative forward signal coefficient at level `t`, from the scheduler |
| `d` | Number of scalar coordinates in a flattened sample or feature vector |
| `h` | Gaussian KDE bandwidth |
| `k` | Apply an intervention after every `k` completed sampler updates |
| `tau`, `Delta_tau` | Artificial BD time and the increment assigned to one intervention |
| `[a]_+` | `max(a, 0)` |
| `B` | Number of proposed branches per parent in the particle explorer |
| `T` | Temperature used to soften transition/branch scores |
| `lambda` | Optional dispersion/diversity bonus coefficient |
| `p_max` | Maximum allowed event probability per particle |
| `T_train` | Number of scheduler training timesteps |
| `s_0` | Standard deviation of the CNF's Gaussian base distribution |
| `L` | Number of random vectors in the Hutchinson trace estimate |

Indices `i`, `j`, and `r` refer to particles; `m` refers to a data reference; `b` refers to a branch. `I` is an identity matrix. The scheduler's `bar_alpha_t` is the product of its forward signal coefficients through step `t`.

For the DDPM forward process, the code’s empirical target uses normalized noisy coordinates `z_t = x_t / sqrt(bar_alpha_t)`. In these coordinates, a clean reference `x_0^(m)` is blurred by Gaussian noise with

$
\sigma_t^2=\frac{1-\bar\alpha_t}{\bar\alpha_t},\qquad
\pi_{M,t}(z)=\frac1M\sum_{m=1}^{M}
\mathcal N\!\left(z;x_0^{(m)},\sigma_t^2 I\right).
$

`EmpiricalNoisedDataTarget.log_prob` evaluates this finite Gaussian-mixture density, including its Gaussian normalization. It is exact for the chosen `M` references, not for the full CIFAR-10/CelebA distribution. BD classes that use target coordinates convert `x_t` to/from `z_t` using $\sqrt{\bar\alpha_t}$.

## 2. The common KDE birth–death update

The original method and the reduced-feature and flow variants share the same population update. First estimate the current population density at each particle. In the original method the code uses the Gaussian KDE

$
\widehat q_h(x_i)=\frac1N\sum_{j=1}^{N}
\exp\!\left(-\frac{\|x_i-x_j\|^2}{2h^2}\right).
$

The usual Gaussian KDE also has a factor $(2\pi h^2)^{-d/2}$. `BirthDeathAcceleration` omits that constant; it is the same for every particle and cancels when the rate is centered. The self term `j=i` is included.

Given a target density $\pi(x_i)$, define

$
\delta_i=\log\widehat q_h(x_i)-\log\pi(x_i),\qquad
\beta_i=\delta_i-\frac1N\sum_{r=1}^{N}\delta_r.
$

Here $\beta_i>0$ means particle (i) is overrepresented relative to the target compared with the population average; it is selected for death. If $\beta_i<0$, it is underrepresented and can give birth. For an intervention increment $\Delta\tau$, the code draws an event with probability

$
p_i=\min\!\left(p_{\max},1-\exp(-|\beta_i|\Delta\tau)\right).
$

The implementation evaluates the exponential stably as `-expm1(-abs(beta) * delta_tau)`. Equivalently, it draws `u_i ~ Uniform(0, 1)` and performs the event if `u_i < p_i`. It also draws a peer `j_i` uniformly from all particle positions except (i). When an event occurs:

- If $\beta_i>0$, particle (i) is replaced by its selected peer (death of (i)).
- If $\beta_i<0$, the peer is replaced by a copy of particle (i) (birth of (i)).

The event decisions and peers are drawn from the pre-update rate vector, then applied in particle order. This copy/replace rule preserves `N`, but can create exact duplicates.

In the current implementations, interventions are gated by cadence and timestep. The default minimum timestep is 100; optional maximum timestep can further restrict the interval. The scheduled raw pseudo-time is `total_time / floor(eligible_updates / k)` (where a clean terminal update is excluded), then the code applies $\Delta\tau=\min(\Delta\tau_{raw},\Delta\tau_{max})$. The default cap is 0.005 and the default event-probability cap is 0.25. Because the pseudo-time is capped, its sum across interventions can be less than `total_time`; timestep filtering can reduce the number of actual interventions further.

## 3. Methods in `src/diffusion/acceleration/`

### 3.1 `BirthDeathAcceleration`: target-density KDE BD

This is the direct form of the common rule above. At each eligible post-sampler state it evaluates the population KDE and a caller-provided `target.log_prob`, centers their log-density difference, and copy/replaces particles according to $p_i$. The target can be analytic (for example, the toy Gaussian mixture) or an empirical finite-data target such as $\pi_{M,t}$ above.

**What makes this BD:** the sign of the centered difference between population density and target density determines which particle is copied or killed. The target is evaluated at the destination noise level between reverse-sampler updates. This is an exploratory discrete extension to diffusion sampling, not a result established by the original stationary Langevin theorem.

**Cost and limits:** the pairwise population KDE is approximately $O(N^2d)$ per intervention. The finite empirical target adds approximately $O(NMd)$ distance work. In image dimensions, the KDE can be dominated by the self-kernel and cease to provide useful neighborhood information.

### 3.2 `ReducedKDEBirthDeathAcceleration`: feature-space KDE BD

This keeps the same BD discrepancy and event rule, but first maps particles and empirical references through a feature map $\phi_t$:

$
z_i=\phi_t(x_i),\qquad r_m=\phi_t(x_0^{(m)}+\sigma_t\epsilon_m),\qquad
\epsilon_m\sim\mathcal N(0,I),
$

when reference re-noising is enabled. The code then estimates two densities in feature space:

$
\widehat q_h^{feat}(z_i)=\frac1N\sum_j
\exp\!\left(-\frac{\|z_i-z_j\|^2}{2h^2}\right),\qquad
\widehat\pi_{M,h}^{feat}(z_i)=\frac1M\sum_m
\exp\!\left(-\frac{\|z_i-r_m\|^2}{2h^2}\right),
$

and uses

$
\beta_i=\left[\log\widehat q_h^{feat}(z_i)-
\log\widehat\pi_{M,h}^{feat}(z_i)\right]
-\frac1N\sum_r\left[\log\widehat q_h^{feat}(z_r)-
\log\widehat\pi_{M,h}^{feat}(z_r)\right].
$

The feature KDEs omit the common Gaussian kernel normalization, which cancels in the centered rate because both use the same feature dimension and bandwidth. The same positive-rate death / negative-rate birth rule follows.

By default, $\phi_t$ is the time-conditioned DDPM-style encoder/bottleneck trunk, globally pooled to 64 or 128 values and then PCA-reduced (default 25). PCA centers by its fitted mean and divides projected coordinates by their fitted standard deviations. Other options include orthogonal projection, frozen random linear/MLP projection, direct pixel PCA, and passing small vectors through unchanged. The bandwidth `h` is therefore measured in the projected feature coordinates.

This is **not** the original image density under a known invertible change of variables: the feature map is generally non-invertible and no Jacobian correction is applied. Thus $\widehat\pi_{M,h}^{feat}$ is a target proxy used to drive BD. For an analytic target without stored image references, the class instead calls `target.log_prob` on the target-coordinate particle samples.

**Weight caveat:** a compatible local `SimpleUNet` can have its encoder weights copied into the trunk, and a compatible checkpoint can be provided. A Hugging Face U-Net is not automatically adapted. In common reduced-KDE benchmark paths the feature trunk starts randomly initialized, with PCA fitted on those random features. Therefore a reduced-KDE run is not automatically a test in pretrained semantic diffusion latents. With `--no-noise-references`, cached features are extracted from clean/timestep-zero references and compared with current-time particle features; this saves work but mismatches their noise levels.

### 3.3 `SMCTransitionLikelihoodAcceleration`: transition-score BD-style resampling

This method avoids both population KDE and a data target density. At each scheduled intervention it uses `source_samples` (the pre-step parent $x_t^{(i)}$) and `samples` (the realized destination $x_s^{(i)}$) supplied by `SamplingPipeline`. It scores that actual transition using the model's Gaussian reverse-transition formula. Let $\epsilon_\theta(x_t,t)$ be the predicted noise and

$
\widehat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,
\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}},
$

with the optional scheduler clipping applied to $\widehat x_0$. For previous level `s`, the code computes

$
\sigma_{t,s}^2=\eta^2\frac{1-\bar\alpha_s}{1-\bar\alpha_t}
\left(1-\frac{\bar\alpha_t}{\bar\alpha_s}\right),\qquad
\mu_{t,s}=\sqrt{\bar\alpha_s}\,\widehat x_0+
\sqrt{[1-\bar\alpha_s-\sigma_{t,s}^2]_+}\,
\epsilon_\theta(x_t,t).
$

For DDIM, the variance is multiplied by the configured $\eta^2$ and the mean uses the matching DDIM direction term. For DDPM, the adjacent posterior mean and selected scheduler variance are used; skipped DDPM steps are rejected. Deterministic DDIM ($\eta=0$) is rejected because it has no non-degenerate Gaussian transition density.

The relative transition log score for each particle `i` is

$
\ell_i=\frac1T\left[-\frac{\|x_s^{(i)}-\mu_{t,s}^{(i)}\|^2}
{2\,\sigma_{t,s}^2}
-\frac d2\log(2\pi\sigma_{t,s}^2)
+\lambda\|x_s^{(i)}-\bar x_s\|^2\right],\qquad
\bar x_s=\frac1N\sum_i x_s^{(i)}.
$

Here `T` is `temperature`; $\lambda$ is the optional `divergence_penalty` (a batch-dispersion bonus, not a divergence calculation). The implementation includes the Gaussian normalizer; it cancels when rates are centered. The code defines

$
\beta_i=-\left(\ell_i-\frac1N\sum_r\ell_r\right).
$

Thus an unusually low transition likelihood gives $\beta_i>0$ and makes the particle eligible for death; a high likelihood makes it eligible for duplication. The common BD event formula, timestep/cadence gates, and caps then apply. This is a **BD-style transition-likelihood heuristic**, not complete SMC: it does not derive target-over-proposal incremental weights or include a correction for the desired endpoint distribution. It performs an extra denoiser call at interventions. The source/destination alignment is fixed, but the score is not a theoretically justified importance weight.

### 3.4 `ParticleFilteringSolutionSpaceExplorer`: per-parent branching, then BD exchange

This method has two distinct selection stages. First, for each parent (`state.source_samples`, $x_t^{(i)}$), it predicts $\widehat x_0^{(i)}$, $\mu_{t,s}^{(i)}$, and $\sigma_{t,s}$ using the sampler-aware formulas above. It generates `B` candidates using the configured DDIM eta or the DDPM posterior variance:

$
y_{i,b}=\mu_{t,s}^{(i)}+\sigma_{t,s}\epsilon_{i,b},
\qquad \epsilon_{i,b}\sim\mathcal N(0,I),\quad b=1,\ldots,B.
$

The branch score is

$
V_{i,b}=-\frac{\|y_{i,b}-\mu_{t,s}^{(i)}\|^2}
{2\,\max(\sigma_{t,s}^2,10^{-12})}
+\lambda\|y_{i,b}-\bar y\|^2,\qquad
\bar y=\frac1{NB}\sum_{i,b}y_{i,b}.
$

The optional diversity coefficient $\lambda$ rewards distance from the mean of the entire `N × B`-candidate pool; a large value can favor outliers. For each parent independently, the code draws one branch from

$
P(b\mid i)=\frac{\exp(V_{i,b}/T)}
{\sum_{c=1}^{B}\exp(V_{i,c}/T)},
$

where `T` is `temperature`. This leaves `N` selected candidates, one per original parent. This is different from a global softmax over all `N × B` candidates.

Second, if `max_event_probability` is positive, the method applies a separate capped cross-particle BD exchange to the selected candidates. Let $V_i^*$ be the chosen branch score and $\bar V^*=N^{-1}\sum_iV_i^*$. The code sets

$
\beta_i=-(V_i^*-\bar V^*),\qquad
p_i=\min\!\left(p_{\max},1-\exp(-|\beta_i|\Delta\tau)\right).
$

Here $\Delta\tau$ is derived from the configured `total_time`, cadence, and `max_delta_tau`. Positive $\beta_i$ kills the low-score selected particle in favor of a peer; negative $\beta_i$ duplicates the higher-score particle over a peer. If the event cap is `None` or non-positive, this second BD stage is skipped, but branch generation and per-parent selection still run.

This is a combination of proposal branching, within-parent selection, and a BD-like population exchange—not the original target-KDE BD rule. It uses neither a target log density nor a population KDE. Candidate count and temporary memory scale with `N × B`, and the model is evaluated to make candidates. The pipeline supplies the pre-step parent state, so branches are generated once from $x_t$ to $x_s$ instead of applying the transition to the already-updated destination. The standard sampler's destination is replaced by the selected branch population. The candidate likelihood is used as a selection score even though candidates were drawn from that same transition; this is an exploratory heuristic, not an importance-corrected particle filter.

### 3.5 `FlowBirthDeathAcceleration` and `ContinuousNormalizingFlowTarget`

`FlowBirthDeathAcceleration` uses the common BD rule with a population Gaussian KDE and a supplied target log density. Its code includes the full KDE normalization:

$
\log\widehat q_h(x_i)=\operatorname{LSE}_j\left(-\frac{\|x_i-x_j\|^2}{2h^2}\right)
-\log N-\frac d2\log(2\pi h^2),
\qquad
\beta_i=\left(\log\widehat q_h(x_i)-\log p_t(x_i)\right)-\text{population mean}.
$

It then uses the common capped event probability and copy/replace rule. In principle, the supplied density could come from a continuous normalizing flow (CNF), which would make this a population-versus-flow-target BD update.

For a CNF velocity field $v_\theta$ with ODE $dx_t/dt=v_\theta(x_t,t)$, the instantaneous change-of-variables identity is

$
\frac{d}{dt}\log p_t(x_t)=-\nabla\!\cdot v_\theta(x_t,t),
\qquad
\nabla\!\cdot v=\operatorname{tr}\!\left(\frac{\partial v}{\partial x}\right).
$

The density integrates this divergence along the ODE path from the queried time to the known Gaussian base at time 1. With the displayed forward convention, it computes

$
\log p_t(x_t)=\log p_1(x_1)+\int_t^1\nabla\!\cdot v_\theta(x_s,s)\,ds,
\qquad \frac{dx_s}{ds}=v_\theta(x_s,s).
$

The path and divergence integral are evaluated with fixed-step RK4. Exact divergence computes the Jacobian trace directly and is intended for low-dimensional states; it is prohibitively expensive for high-dimensional images. The Hutchinson option estimates it using

$
\operatorname{tr}(J_v)\approx\frac1L\sum_{l=1}^{L}
e_l^\top J_v e_l,\qquad e_l\sim\mathcal N(0,I),
$

which reduces trace cost but remains a stochastic estimate. `log_prob` enables the local autograd needed for divergence even inside the pipeline's `torch.no_grad()` context. `score` differentiates the integrated density with respect to the input.

**CLI caveat:** `sample_reduced_kde.py --acceleration-method flow_exact` and `scripts/run_exact_flow_eval.py` still use their selected analytic or empirical target directly; they do not construct a CNF. The new `tests/toy_acceleration_extensions.py` separately tests Flow-BD with an exact toy mixture density and tests the CNF density on an identity-flow Gaussian target. Flow BD converts between raw and normalized noisy-state coordinates when the target declares them.

## 4. At-a-glance differences

| Method | Density/score used for selection | Is it BD? | Main extra work or limitation |
|---|---|---|---|
| `BirthDeathAcceleration` | Full-coordinate population KDE versus supplied target $\log\pi_t$ | Yes; direct centered-rate copy/replace | `O(N^2 d)` KDE; high-dimensional self-kernel problem |
| `ReducedKDEBirthDeathAcceleration` | Feature KDE versus feature-reference KDE (or analytic toy target) | Yes; same BD rule in projected space | Quality depends on feature map; current common trunk can be random |
| `SMCTransitionLikelihoodAcceleration` | Actual Gaussian reverse-transition likelihood | BD-style copy/replace, but not complete SMC or target-corrected BD | Additional denoiser pass; supports stochastic DDIM eta and adjacent DDPM |
| `ParticleFilteringSolutionSpaceExplorer` | Branch-transition likelihood plus optional batch-dispersion bonus | Branch/filter stage plus a second BD-like exchange | `N × B` candidates; score is not an importance-corrected weight |
| `FlowBirthDeathAcceleration` | Full-coordinate population KDE versus provided `log_prob` | Yes, if given a valid target density | CNF exact trace is costly; Hutchinson mode is stochastic |

None of these modifiers currently removes reverse-sampler steps by itself. They intervene between the configured steps and can add compute; a demonstrated speedup would require comparing wall-clock at matched quality/diversity or adding a justified adaptive stopping/step-skipping policy.

## 5. How the methods map to the professor's comments

### KDE on image latents and why particles are retained or replaced

The two KDE BD methods expose the needed mathematical decision rule: positive centered $\log\widehat q-\log\pi$ means overrepresentation and death; negative means underrepresentation and birth. `ReducedKDEBirthDeathAcceleration` computes this in a compressed feature vector, while `BirthDeathAcceleration` computes it in the target coordinates. `scripts/run_bd_unet_eval.py` is a newer non-toy route using a pretrained CIFAR-10 DDPM U-Net for transport with an empirical CIFAR-10 target, but it uses **pixel-space** BD, not pretrained image latents.

Still missing for the latent-specific request: a reduced-KDE run with reliably loaded pretrained denoiser features, an image-feature KDE diagnostic, and a per-particle trace recording both density terms, $\beta_i$, event probability, random event decision, chosen peer, and resulting lineage.

### Diversity before and after each update

Several non-toy benchmark scripts compute a final perceptual-uniqueness fraction; toy diagnostics report quality/diversity before and after a BD update. Modifier metadata includes event counts and selected scores. However, the pipeline does not preserve a full per-intervention history, and image scripts do not record per-step lineage or before/after diversity. These metrics cannot locate exactly when image repetition begins.

### Matched quality/diversity and wall-clock including intervention cost

The benchmark runners compare baseline and modified sampling at selected step counts and report NFE, elapsed sampling time, diversity, and optionally MMD/FID. The timed call includes modifiers executed inside the sampling pipeline (including KDE and extra model calls). They do **not** automatically search for a lower-step BD run matching a higher-step baseline in both quality and diversity, or produce a complete quality/diversity versus wall-time frontier. Target/reference construction, PCA fitting before sampling, and separate FID generation/evaluation are outside the preview-sampling timer. FID references are generally drawn from the training view, not a held-out test split. Use multiple seeds and independent sampler/modifier RNGs for defensible comparisons.

## 6. Tests and experiment scripts

### Unit and toy diagnostics

| File | What it checks | Data regime / limitation |
|---|---|---|
| `tests/test_novel_acceleration_methods.py` | Transition score, flow target score/log-prob, one flow-BD update, particle-filter output size | Synthetic tensors; no full image-pipeline validation |
| `tests/test_reduced_kde_birth_death.py` | Feature sizes, projection/PCA, vector pass-through, one reduced-KDE update, CLI parsing | Synthetic vectors and random image tensors |
| `tests/bd_kde_dim_diagnostic.py` | KDE self-mass, effective neighbors, KDE/target rank, duplicates, MMD/NLL around one BD update | Known Gaussian vector targets across dimensions |
| `tests/bd_paper_toy_diagnostic.py` | Langevin/BD trajectories, bandwidth/seed sweeps, component TV/MSE/NLL/time | Known 2D mixture |
| `tests/bd_paper_equal_time.py` | ULA versus BD at equal step counts and longer ULA runs | Known 2D mixture |
| `tests/bd_diffusion_sweep.py` | DDPM/DDIM sweeps over steps, eta, bandwidth, BD pseudo-time and seeds; NLL/MMD/coverage/TV/uniqueness/time | Analytic 2D mixture |
| `tests/toy_bd_experiment.py`, `scripts/run_and_plot_toy.py` | Toy DDPM/DDIM/Langevin comparisons and sample plots | Analytic 2D mixture |
| `tests/toy_acceleration_extensions.py` | Runs direct KDE BD, reduced KDE, transition-likelihood resampling, branch/filter, Flow-BD, and an identity-CNF Gaussian comparison; reports NFE, time, NLL, MMD, mode coverage, and mode balance | Analytic 2D mixture plus an exact Gaussian CNF/ULA case; 2D reduced KDE passes vectors through and does not test image feature reduction |

### Non-toy benchmark scripts

| File | Purpose and caveat |
|---|---|
| `scripts/run_bd_unet_eval.py` | Baseline DDIM versus pretrained CIFAR DDPM transport plus pixel-space empirical KDE BD; CIFAR references, NFE/time, uniqueness, optional MMD/FID. Data path is hard-coded to `data/`; not the reduced-latent test. |
| `scripts/run_reduced_kde_eval.py` | Step-grid benchmark for baseline, pixel KDE, reduced KDE, pixel PCA, and selectable SMC/PF/flow paths; saves metrics/plots. Its common model is `target-score`, not a trained image denoiser, so use it as a controlled plumbing/ablation runner. |
| `scripts/run_deliverable4_eval.py` | Earlier pixel-space BD step/budget/cadence sweeps on CIFAR-10 or ImageFolder. |
| `scripts/run_smc_eval.py` | Baseline DDIM versus transition-score resampling with a pretrained model by default; default step list 10/25/50 and FID disabled. Current cases use eta 1. |
| `scripts/run_particle_filter_eval.py` | Baseline DDIM versus branching/filtering with a pretrained model by default; reports NFE/time, uniqueness and optional MMD/FID. Results remain exploratory due to state alignment. |
| `scripts/run_exact_flow_eval.py` | Baseline versus FlowBD modifier with a pretrained transport model; target is Gaussian when FID is disabled and empirical-noised data when FID images are loaded. Despite the script label, it does not train or instantiate a CNF. |
| `sample_reduced_kde.py` | One selectable method and sampler; writes NFE/time/uniqueness and optional JSON. Its `flow_exact` choice has the target caveat above. |

These tests mainly check shapes/execution and known toy targets. They do not prove that the transition heuristic is statistically correct, that branch/filter is a canonical particle filter, or that learned image features have semantic meaning. The CNF RK4 integration is checked against a linear-Gaussian analytic case; accuracy for a learned flow still depends on integration resolution, divergence estimator, and velocity model.

## 7. Running experiments

Install the package and optional FID dependencies if needed:

```bash
python3 -m pip install -e .
python3 -m pip install -e "[fid]"
```

Run unit/smoke checks:

```bash
python3 -m pytest tests/test_novel_acceleration_methods.py tests/test_reduced_kde_birth_death.py
```

Run the unified toy comparison for every extension:

```bash
python3 tests/toy_acceleration_extensions.py --count 128 --steps 25 --timesteps 1000 --eta 1 --device cpu --output-dir samples/toy-acceleration-extensions
```

The DDIM section compares all extensions against the same analytic Gaussian-mixture sampler. The separate fixed-target ULA section exercises CNF log-density inside Flow-BD using an identity CNF for a standard Gaussian. The report and sample tensors are saved under the output directory. Use `--eta 0` only for methods that do not need a stochastic transition density; the runner requires positive eta because it includes transition scoring and branching.

Toy diagnostics:

```bash
python3 tests/bd_kde_dim_diagnostic.py --dims 2,8,32,128,768 --counts 256,1024 --seeds 7,17,29 --total-times 0.1,1 --output samples/bd-kde-dim/metrics.json
python3 tests/bd_paper_toy_diagnostic.py --count 128 --steps 5000 --bandwidths 0.3,0.7,1.2 --output samples/bd-paper-toy-diagnostic/metrics.json
python3 tests/bd_diffusion_sweep.py --counts 64 --steps 5,10 --etas 1 --bandwidths 0.7 --total-times 0.1 --seeds 7 --output samples/bd-diffusion-sweep/metrics.json
```

For a pretrained CIFAR denoiser with empirical pixel-space BD:

```bash
python3 scripts/run_bd_unet_eval.py --steps 10 25 50 --batch-size 16 --reference-count 512 --bd-min-timestep 100 --bd-max-time 0.005 --bd-max-event-prob 0.25 --device cuda --output-dir samples/bd-unet-cifar
```

Add `--fid` to compute FID (the script uses up to `--fid-samples` real and generated examples). This script downloads/loads CIFAR-10 from its hard-coded `data/` location and uses the pretrained model selected by `--pretrained-model-id`.

Dedicated pretrained-model comparisons for the other extensions:

```bash
python3 scripts/run_smc_eval.py --steps 10 25 50 --count 16 --fid-samples 0 --device cuda --output-dir samples/smc-cifar
python3 scripts/run_particle_filter_eval.py --steps 10 25 50 --count 16 --fid-samples 0 --branch-factor 2 --device cuda --output-dir samples/pf-cifar
```

Set `--fid-samples` above zero to request FID/MMD reference loading. Transition-likelihood and branch/filter methods reject deterministic DDIM (`eta=0`) and use the actual stochastic transition parameters.

Reduced-KDE and method grid on CIFAR-10:

```bash
python3 scripts/run_reduced_kde_eval.py --dataset cifar10 --data-dir data --download --image-size 32 --reference-count 256 --fid-samples 0 --count 16 --seed 42 --device cuda --output-dir samples/cifar-reduced-kde
```

Add `--compare-all-paradigms` for its SMC and particle-filter cases; add `--pca-only` for pixel PCA comparisons; use `--methods baseline pixel_kde reduced_kde` to choose explicit methods. For a folder dataset, use `--dataset imagefolder --data-dir data/celeba --image-size 256` and ensure the directory has `ImageFolder` class subdirectories. The reduced-KDE benchmark constructs a `target-score` model from an empirical subset rather than loading a pretrained image denoiser.

The separate flow-BD benchmark is runnable as:

```bash
python3 scripts/run_exact_flow_eval.py --steps 10 25 --count 16 --fid-samples 0 --device cuda --output-dir samples/flow-bd-cifar
```

With FID disabled, its target is a standard Gaussian. With FID enabled, it constructs an empirical noisy-data target from the images loaded for evaluation and also performs FID work. Neither path is a CNF experiment; see Section 3.5.

The original pixel-BD sweep remains available:

```bash
python3 scripts/run_deliverable4_eval.py --dataset cifar10 --data-dir data --download --reference-count 256 --fid-samples 0 --count 16 --seed 42 --device cuda --quick --output-dir samples/deliverable4-cifar
```

## 8. Important next steps

1. Provide a reliable pretrained-denoiser feature extractor/checkpoint for reduced KDE, and diagnose KDE overlap on those actual image features.
2. Add a trajectory logger for per-update lineages, before/after diversity, KDE/target or transition scores, centered rates, event probabilities, peers, births/deaths, timestep, and wall time.
3. Derive transition resampling weights from the actual target/proposal relationship and ensure they match sampler type, DDIM eta, and skipped timesteps.
4. Fix particle-filter parent-state/timestep alignment; measure its `N × B` candidate overhead and diversity at each filtering event.
5. Implement a trained CNF target with numerical ODE integration of divergence, a validated time convention, gradient-enabled evaluation inside sampling, and an end-to-end known-density test.
6. Use held-out reference data, independent RNG streams, multiple seeds, and quality/diversity-versus-wall-clock curves. Include reference/PCA setup and FID work when reporting end-to-end costs.

For the original BD equations and assumptions, see [DOCS-BD.md](DOCS-BD.md). For earlier project findings, see [FINDINGS.md](FINDINGS.md).
