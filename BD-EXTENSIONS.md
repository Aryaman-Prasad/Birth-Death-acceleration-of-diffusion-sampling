# Birth–Death extensions: code map, experiments, and current gaps

This note describes the acceleration code currently in the repository and how to inspect or run it. The project presentation's slides 15 and 18 and the report's Sections 7–8 motivate the recent additions: high-dimensional KDE diagnostics, image-feature KDE, reverse-transition weighting, particle branching, and flow-based density evaluation.

The new methods are experimental. Their class names describe the intended idea; a few of the current integrations are prototypes and should not yet be treated as faithful implementations of a complete algorithm or as evidence of image-sampling acceleration.

## 1. What is in `src/diffusion/acceleration/`

| Implementation | What it uses to decide particle selection | Main difference | Current status |
|---|---|---|---|
| `BirthDeathAcceleration` | Difference between KDE of the current population and a supplied target log density | Original fixed-population BD rule, using the sampler's state coordinates directly | Paper-style rate/resampling rule; image-space KDE is unreliable in high dimensions |
| `ReducedKDEBirthDeathAcceleration` | The same population-versus-target KDE discrepancy after mapping images into a smaller feature vector | Replaces pixel distances with distances in a compressed feature space | Runnable experimental path; feature trunk is usually randomly initialized in the current image benchmark |
| `SMCTransitionLikelihoodAcceleration` | Likelihood of the observed reverse transition under a Gaussian DDPM-style reverse kernel | Replaces pairwise KDE and target log density with a transition-likelihood score | Heuristic transition-based resampling; not a complete SMC weighting derivation and not matched to skipped-step DDIM |
| `ParticleFilteringSolutionSpaceExplorer` | Scores a pool of sampled candidate reverse transitions, then retains N candidates | Branches each particle and resamples from an N×B candidate pool | Exploratory branch/filter method; current post-update state indexing needs correction before results are trusted |
| `FlowBirthDeathAcceleration` and `ContinuousNormalizingFlowTarget` | Population KDE compared with a flow-based log-density | Supplies a tractable-density target through a velocity field | Prototype; the current flow density is not integrated exactly, is not wired as a flow target in the CLI, and needs gradients inside the sampling pipeline |

All are `SamplingModifier`s called by `SamplingPipeline` after a sampler update. BD and the resampling variants keep the population size fixed; the particle-filter method temporarily expands candidates and filters back to the original count. They run during generation, never during model training.

### 1.1 Original pixel-space KDE BD

`BirthDeathAcceleration` computes, for each particle, a Gaussian KDE over the other current particles and compares it with `target.log_prob`. It centers the resulting rate over the population. A positive centered rate marks a relatively overrepresented particle, which is replaced by a uniformly selected peer; a negative rate marks an underrepresented particle, which is copied over a peer. The event probability is `1 - exp(-abs(rate) * delta_tau)`.

For toy targets, `log_prob` can be analytic. For image experiments, `EmpiricalNoisedDataTarget` supplies the exact density of a finite Gaussian mixture centered on a selected data subset. This is exact for that finite mixture, not for the full CIFAR-10 or CelebA distribution. The pixel-space particle KDE costs roughly O(N²D) per intervention; empirical target evaluation adds roughly O(NMD), where N is the particle count, M the reference count, and D the flattened image dimension.

This is the original rule documented in [DOCS-BD.md](DOCS-BD.md). Applying it between DDPM/DDIM updates is an exploratory time-dependent extension; the paper's fixed-target Langevin convergence result does not establish convergence or acceleration for that reverse process.

### 1.2 Reduced feature-space KDE

`ReducedKDEBirthDeathAcceleration` passes both particles and target references through `FeatureReducer`, then computes the particle KDE and empirical-reference KDE in the reduced coordinates. The default feature path is a small DDPM-style encoder/bottleneck trunk (64 or 128 pooled features), followed by PCA to 25 dimensions. Orthogonal, frozen random linear, and frozen random MLP projections are also available. The reference images can be re-noised to the current timestep before feature extraction.

The intended benefit is to make neighborhoods overlap in a compact representation, avoiding pixel-space self-kernel domination. The price is that the result depends on the representation and its scaling. For image targets, the target term is a KDE over transformed references; it is a feature-space density proxy, not the original pixel-space target density under a known invertible change of variables.

There is an important implementation qualification: the trunk is not automatically loaded from the selected Hugging Face pretrained U-Net. It copies weights only from a compatible local `SimpleUNet` exposing matching encoder layers. The non-toy benchmark script builds a `target-score` model, which also does not provide those encoder layers. In those common paths the trunk starts with random weights; PCA is then fitted to features from that random trunk. Thus the code tests a compressed feature-space KDE, but it does not currently demonstrate KDE in pretrained semantic diffusion latents as the option names and script descriptions suggest.

`--no-noise-references` caches features extracted at timestep 0 and compares them with current-timestep particle features. That avoids repeated reference feature extraction, but the feature distributions are then at different noise levels. Keep the default `--noise-references` for a state-matched experiment.

### 1.3 Reverse-transition likelihood resampling

`SMCTransitionLikelihoodAcceleration` caches the previous particle state. After a later sampler update, it evaluates the new state under a Gaussian reverse transition predicted from the cached parent. It centers these per-particle log likelihoods, converts them to event probabilities, and applies a fixed-population birth/death-style copy/replace rule. It uses no population pairwise KDE and no target log density. It performs an extra model forward call at interventions; the ordinary sampling hook counts these calls in NFE and sampling time.

This is related to the presentation's proposed analytical reverse-kernel/SMC extension. The current weight is a transition-likelihood heuristic, not a full sequential importance weight with a target/proposal correction. Its variance formula is DDPM-style and does not use DDIM's `eta` or skipped-step transition variance. Use adjacent-step DDPM to inspect the current formula; treat DDIM runs as exploratory until the likelihood matches the actual DDIM proposal.

### 1.4 Candidate branching and filtering

`ParticleFilteringSolutionSpaceExplorer` draws B candidate transitions per particle, scores each using transition residual energy plus an optional distance-from-batch-mean bonus, forms a softmax, and selects N candidates without replacement. Its controls are `branch_factor`, `temperature`, `diversity_weight`, and intervention cadence. It uses neither the BD KDE discrepancy nor an explicit target log density. Candidate generation also costs memory and compute proportional to N×B, plus an extra model evaluation at an intervention.

There is a state-indexing issue to fix before using its output as a sampler comparison. The pipeline invokes modifiers **after** the ordinary sampler step, so `state.samples` is already the destination state. The explorer currently treats that tensor as `x_t` and computes another reverse transition using the source timestep. It therefore branches from a post-update state with source-step labels, rather than branching the original parent state under the transition that just ran. The unit test checks pool size and output shape, not this trajectory alignment.

### 1.5 Flow-density prototype

`ContinuousNormalizingFlowTarget` accepts a velocity network and provides a Gaussian-prior log density with a divergence correction. `FlowBirthDeathAcceleration` then compares a population KDE with the supplied target density and applies the same general copy/replace mechanism.

The implementation is not yet an exact CNF density calculation. A CNF log density needs the divergence integrated along the ODE trajectory. The current code evaluates divergence at the endpoint and multiplies it by the remaining time interval, a one-point approximation. Exact divergence loops over every output dimension and is impractical for image dimensions; Hutchinson estimation is available in the class but does not make the path exact. Also, `SamplingPipeline.sample` runs under `torch.no_grad()`, while CNF divergence and score evaluation need autograd; the flow target needs an explicit gradient-enabled section before it can run through the pipeline.

The `sample_reduced_kde.py --acceleration-method flow_exact` branch does not construct `ContinuousNormalizingFlowTarget`. It passes the selected toy or empirical target to `FlowBirthDeathAcceleration`. That option currently runs a full-coordinate KDE against the selected target's log density; it is not a CNF experiment. The modifier also does not convert sampler states to the target's declared coordinates, so DDIM's raw `x_t` can be compared directly with a target defined in normalized `x_t/sqrt(alpha_bar_t)` coordinates. There is no flow training/configuration command or end-to-end flow sampling test in the repository yet.

## 2. How the recent code maps to the professor's comments

### Test KDE on image latents and explain retention/replacement

**Partly addressed.** `ReducedKDEBirthDeathAcceleration` performs particle and reference KDEs in a compact feature space, and its update metadata contains births and deaths for an intervention. `tests/test_reduced_kde_birth_death.py` checks feature dimensions, projection/PCA behavior, and that one update returns metadata. `tests/bd_kde_dim_diagnostic.py` measures self-kernel mass, effective neighbors, KDE/target rank correlation, duplicates, MMD, and NLL across vector dimensions.

**Still missing:** a trustworthy pretrained diffusion latent extractor in the common non-toy route, density/rate traces showing why particular particles were selected, and a high-dimensional KDE diagnostic directly on the image features used by the benchmark. The current high-dimensional diagnostic uses Gaussian vectors, not image latents.

### Track diversity before and after each update

**Partly addressed.** The non-toy benchmark scripts report a final unique-sample fraction. The Gaussian dimension diagnostic measures diversity before and after one BD update. BD modifiers put event counts in the returned `SamplingState.metadata`.

**Still missing:** the pipeline discards each intermediate state's metadata and no image run logs unique lineages or image tiles after every intervention. The available final unique fraction cannot identify which update first caused repetition. `DOCS-BD.md` describes a manual toy trajectory observation, but it is not a reusable image-trajectory logger.

### Compare DDPM/DDIM quality, diversity, and wall time including KDE

**Partly addressed.** `scripts/run_deliverable4_eval.py` and `scripts/run_reduced_kde_eval.py` compare baseline and modified sampling at selected equal step counts, report FID and final unique fraction, and time the preview `SamplingPipeline.sample` call. That timer includes interventions executed inside the pipeline, including KDE/feature work and extra model calls.

**Still missing:** these scripts do not automatically find a matched-quality or matched-diversity point, nor do they produce a quality/diversity-versus-wall-clock frontier. Their recorded time excludes target construction, PCA fitting performed when a modifier is created, and the separate FID-generation/evaluation pass. The FID reference examples are sampled from the same training dataset used to construct the empirical target, rather than a held-out split. The current comparison runners also pass the sampler generator as the modifier generator, so BD random draws consume the same random stream as sampler noise; the standalone `sample_reduced_kde.py` separates those streams.

## 3. Existing testing and experiment tools

| File | What it checks | Data regime |
|---|---|---|
| `tests/test_novel_acceleration_methods.py` | Smoke/unit checks for transition likelihood, flow log-probability/score, one flow-BD update, and branch/filter population size | Small synthetic tensors; it does not run a full diffusion pipeline or validate statistical correctness |
| `tests/test_reduced_kde_birth_death.py` | Encoder output shape/time-conditioning, projections/PCA, vector pass-through, one reduced-KDE update, CLI parsing | Synthetic vectors and random image tensors |
| `tests/bd_kde_dim_diagnostic.py` | KDE self-mass, neighbor count, rank relationship, duplicates, MMD/NLL before/after one update | Known Gaussian vector targets across dimensions |
| `tests/bd_paper_toy_diagnostic.py` | Fixed-target paper-inspired ULA/BD trajectories; bandwidth/seed sweeps and mode-responsibility TV, MSE, NLL, time | Known 2D four-component mixture |
| `tests/bd_paper_equal_time.py` | Fixed-target ULA versus BD at equal step counts and longer ULA runs, reporting quality and wall time | Known 2D mixture |
| `tests/bd_diffusion_sweep.py` | Matched-step DDIM/DDPM runs over step count, `eta`, bandwidth, pseudo-time, and seeds; NLL, MMD, mode coverage/TV, unique fraction, time | Analytic 2D toy mixture |
| `tests/toy_bd_experiment.py`, `scripts/run_and_plot_toy.py` | Toy DDPM/DDIM/Langevin comparisons with and without original BD; metrics and sample plots | Analytic 2D toy mixture |
| `scripts/run_deliverable4_eval.py` | Original pixel-space BD versus baseline over a step/budget/cadence/eta sweep; FID and final unique fraction | CIFAR-10 or ImageFolder |
| `scripts/run_reduced_kde_eval.py` | Baseline, pixel KDE, reduced KDE; optional SMC-transition and particle-filter cases; FID, final unique fraction, NFE/time, CSV/JSON/plots | CIFAR-10 or ImageFolder; builds an empirical target-score model |
| `sample_reduced_kde.py` | One run for one selected acceleration method; prints NFE/time/final unique fraction, saves sample output and optional JSON | Can select a pretrained/checkpoint model for SMC/PF, or an empirical target for reduced/flow branches |

The tests are useful for shape and execution regressions. They do not establish that SMC weights are statistically correct, that the CNF density is exact, that particle-filter state indices match sampler transitions, or that a feature space is semantically meaningful. The test suite was not executed as part of this review.

The main experiment scripts can be run directly for toy diagnostics:

```bash
# Pixel-KDE self-domination as state dimension increases
python3 tests/bd_kde_dim_diagnostic.py --dims 2,8,32,128,768 --counts 256,1024 --seeds 7,17,29 --total-times 0.1,1 --output samples/bd-kde-dim/metrics.json

# Paper-inspired fixed-target Langevin mixing and bandwidth sweep
python3 tests/bd_paper_toy_diagnostic.py --count 128 --steps 5000 --bandwidths 0.3,0.7,1.2 --output samples/bd-paper-toy-diagnostic/metrics.json

# Diffusion sampler sweep on the analytic 2D target
python3 tests/bd_diffusion_sweep.py --counts 64 --steps 5,10 --etas 1 --bandwidths 0.7 --total-times 0.1 --seeds 7 --output samples/bd-diffusion-sweep/metrics.json
```

The first isolates the KDE estimator; the second tests the paper's stationary ULA setting; the third applies the project pipeline to DDIM/DDPM. They answer different questions and their scores should not be combined as though they were one benchmark.

## 4. Running the non-toy experiments

Install the project and, for FID, its optional dependencies:

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[fid]"
```

Run the unit/smoke tests with:

```bash
python3 -m pytest tests/test_novel_acceleration_methods.py tests/test_reduced_kde_birth_death.py
```

### Compare original pixel BD and reduced KDE on CIFAR-10

This runs the six default DDIM cases (25 and 50 steps; baseline, pixel KDE, reduced KDE), saves grids, and writes CSV/JSON/plots:

```bash
python3 scripts/run_reduced_kde_eval.py --dataset cifar10 --data-dir data --download --image-size 32 --reference-count 256 --fid-samples 0 --count 16 --seed 42 --device cuda --output-dir samples/cifar-reduced-kde
```

`--fid-samples 0` skips FID. To include the optional SMC-transition and particle-filter cases, add `--compare-all-paradigms`. To include 10-step cases, add `--include-steps-10`; `--include-ddpm` adds a 1000-step DDPM baseline. For a folder dataset, use `--dataset imagefolder --data-dir data/celeba --image-size 256`; the directory must have `ImageFolder` class subdirectories.

This benchmark currently uses `target-score` built from the selected empirical subset, not a pretrained CIFAR/CelebA diffusion model. Its reduced features therefore use the randomly initialized encoder caveat above. Its FID reference images also come from that dataset's training view. Treat it as a plumbing/ablation runner, not a held-out learned-model benchmark. Set `--fid-samples` to a positive count to compute FID; this requires the optional FID dependencies and can dominate runtime.

### Run one method with the standalone sampler

For reduced KDE against a non-toy empirical target:

```bash
python3 sample_reduced_kde.py --empirical-data-target --target-dataset cifar10 --target-data-dir data --target-download --target-image-size 32 --target-reference-count 256 --sampler ddim --steps 25 --eta 1 --birth-death --acceleration-method reduced_kde --kde-reduce 25 --bd-every-k 2 --bd-total-time 0.01 --count 16 --seed 7 --device cuda --output samples/cifar-reduced-kde-one.png --metrics-output samples/cifar-reduced-kde-one.json
```

This also uses the empirical target-score model rather than a separately loaded learned model. `sample_reduced_kde.py` currently accepts `--fid`/FID options but does not evaluate FID; use `scripts/run_reduced_kde_eval.py` for FID. The standalone script only writes JSON when `--metrics-output` is provided.

For transition-likelihood resampling with a pretrained CIFAR model, the current likelihood is DDPM-style, so use adjacent-step DDPM for a formula-aligned smoke run:

```bash
python3 sample_reduced_kde.py --pretrained-model-id google/ddpm-cifar10-32 --sampler ddpm --birth-death --acceleration-method smc_transition --bd-every-k 2 --bd-total-time 0.01 --count 16 --seed 7 --device cuda --output samples/cifar-smc-transition.png --metrics-output samples/cifar-smc-transition.json
```

This uses all scheduler timesteps and can be slow. A reduced-step DDIM run is possible from the CLI, but its transition likelihood must first be changed to match the chosen DDIM `eta` and skipped-step proposal before its quality result is interpreted as an SMC comparison.

The `particle_filter` option is executable through `sample_reduced_kde.py` with `--acceleration-method particle_filter --filter-branch-factor 2`. Keep it to a smoke run until the parent-state/timestep alignment described above is corrected. `flow_exact` is also selectable, but it does not instantiate a CNF target; do not treat that flag as a CNF experiment.

### Run the original Deliverable 4 sweep

For the older pixel-space BD baseline, `scripts/run_deliverable4_eval.py` accepts the same CIFAR/ImageFolder data options and runs a larger fixed grid. `--quick` reduces the grid. Set `--fid-samples 0` to skip FID; otherwise its default is 500 images and requires the FID extra. Outputs go to `deliverable4_results/` by default. This runner compares fixed step counts and records final diversity, not diversity after every intervention.

```bash
python3 scripts/run_deliverable4_eval.py --dataset cifar10 --data-dir data --download --reference-count 256 --fid-samples 0 --count 16 --seed 42 --device cuda --quick --output-dir samples/deliverable4-cifar
```

## 5. What remains to make these results defensible

1. Load a real trained/pretrained denoiser's encoder for reduced-KDE, or provide and train a clearly specified image encoder. Allow the model and the empirical reference dataset to be selected independently so the learned sampler and target references can be evaluated together.
2. Add a per-intervention trace: particle IDs/lineages, unique float and saved-image counts before/after, KDE self-mass or neighbor statistics, `log KDE`, target/transition score, centered rate, event probability, births/deaths, timestep, and wall time. Preserve these records outside `SamplingState.metadata` across the whole pipeline.
3. Make SMC use the actual sampler proposal, including DDIM `eta` and skipped timesteps, and derive/validate target-over-proposal incremental weights. Clarify whether the desired method is SMC resampling or the paper's centered BD process.
4. Correct the particle-filter parent-state alignment and report the N×B candidate cost, added model evaluations, and diversity at each filtering event.
5. Implement a trained CNF/flow target with numerical ODE integration of divergence, gradient-enabled evaluation within the sampler, and an end-to-end sampling CLI/test against a target with known density.
6. Use a held-out reference split, independent random streams for sampler and modifier, multiple seeds, and quality/diversity-versus-wall-clock curves. Report both preview and FID sample counts. The current scripts compare selected step budgets but do not solve the matched-quality/diversity operating point automatically.

For the broad project findings and previous results, see [FINDINGS.md](FINDINGS.md). For the original BD equations and assumptions, see [DOCS-BD.md](DOCS-BD.md). The presentation's proposed SMC and flow directions are on slide 18; the report's high-dimensional diagnostics and image results are in Sections 7–8 of `IE643_Project_Report.pdf`.
