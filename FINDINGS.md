# Birth–death acceleration for diffusion sampling: findings

**Question.** Is the project's birth–death (BD) acceleration idea sound, and why have
the experiments so far shown no positive effect?

**Short answer.** The BD implementation is faithful to Lu, Lu & Nolen (2019) and
does what it claims in the paper's own regime. But that regime is *fixed-target
Langevin mixing with a known density*. It does not transfer to accelerating a
diffusion sampler:

1. **BD cannot reduce NFE.** A reverse diffusion step transports particles to a
   new noise level; BD only copies/kills existing particles at the current noise
   level. It therefore cannot replace a reverse step. Every "BD-accelerated" run
   in this repo still uses exactly the same number of model calls as its baseline.
2. **BD needs the log density of the current marginal, not just the score.** For a
   pretrained image model that quantity is unavailable. The only working paths in
   the repo are an analytic toy density and a finite empirical target.
3. **At matched NFE the quality gain is small and inconsistent.** On a 2-D toy
   mixture, BD changes MMD by roughly ±10% at 5–10 steps and is within noise at
   20–50 steps. NLL often improves, but that is mode/variance collapse: the exact
   duplicate fraction falls from 1.0 to 0.2–0.6.
4. **At matched wall-clock, spending the compute on more steps wins.** A 5-step
   BD run is both worse and slower than a 10-step baseline; a 10-step BD run is
   worse than a 20-step baseline at the same wall-clock.
5. **In high dimension the KDE is self-dominated, so BD degenerates** into
   "duplicate high-density particles". At d = 768 the behaviour is identical for
   bandwidths from 0.1 to 78, diversity drops, and MMD gets worse.
6. **The one genuine positive regime** is a known target density plus a
   *materially biased* score plus stochastic sampling (`eta = 1`). There BD acts
   as a model-error correction, not as a step-reduction method.

The idea is mathematically sound; the project's stated goal ("reduce the number
of diffusion sampling steps") is not achievable with BD as formulated. The rest
of this note gives the evidence and a concrete reframing.

---

## 1. What is implemented

`src/diffusion/acceleration/birth_death.py` implements Algorithm 1 of the paper:

- `beta_i = log KDE_h(x_i) - log pi(x_i)`, centred over the population;
- `p_i = 1 - exp(-|beta_i| * Delta_tau)`;
- if `beta_i > 0` (over-represented) kill `i` and duplicate a random partner;
  if `beta_i < 0` (under-represented) duplicate `i` and kill a random partner.

This matches the paper. I did not find a sign error, a rate error, or an
off-by-one in the event loop. The important caveats are already in the module
docstring: for DDPM/DDIM this is an "exploratory discrete splitting", and the
stationary-target convergence result from the Langevin paper does not imply it.

Two structural facts matter more than any tuning detail:

- **It is a resampling operator, not a transport operator.** It never moves a
  particle along the diffusion trajectory; it only changes which particles exist.
- **It compares the population to `pi_t`, the marginal at the current noise
  level.** `ToyGaussianMixtureTarget.log_prob` correctly evaluates the noised
  mixture at `sigma_t`, and `EmpiricalNoisedDataTarget` correctly evaluates the
  noised empirical target. But a standard pretrained model only supplies
  `grad log pi_t`; it does not supply `log pi_t`.

The `total_time` (BD pseudo-time) and `kernel_bandwidth` are free
hyperparameters with no natural diffusion analogue. The paper ties `Delta_tau`
to the ULA step size; in a diffusion sampler there is no such step size, so the
amount of BD is arbitrary. Results below depend strongly on it.

---

## 2. Experiments run

All experiments use the project's own pipeline, targets, and quality functions
unless noted. Raw runs are reproducible with the scripts in `tests/` (Section 7).

| # | Setting | What it isolates |
|---|---------|------------------|
| 1 | 2-D, 8-mode toy mixture, oracle score, DDIM/DDPM, N = 16/64/256, 3–50 steps, `eta` = 0/1, `h` = 0.3/0.7/1.2, `T` = 0.1/1/10, 4 seeds | Standard diffusion sampling at matched NFE |
| 2 | Paper's 4-mode anisotropic mixture, fixed-target ULA, N = 16/64/128/1000, 5k BD steps vs 5k–100k ULA steps, 4 seeds | The paper's actual regime, and the iteration-vs-wall-clock trade-off |
| 3 | Isotropic Gaussian target already sampled exactly, d = 2/8/32/128/768, N = 256/1024, absolute and median bandwidths | Why the KDE fails in high dimension (a null test: BD should do nothing) |
| 4 | Toy mixture + deliberately biased score `s_model = c * s_true`, c = 0.5–1.5, 5–20 steps, `eta` = 0/1 | Whether BD can correct model error, and when |
| 5 | `toy-images` empirical target, DDIM/Langevin, 25/100 steps, N = 16/64, bandwidths 0.3–1000, `T` = 0.01–1 | The repo's image-facing BD path |

Metrics: NLL (lower better), squared RBF multiscale MMD (lower better),
mode coverage and component-balance TV (lower better), exact-duplicate fraction
(1.0 = no exact copies), NFE, wall-clock.

---

## 3. Results

### 3.1 Toy diffusion: no consistent gain at matched NFE

2-D 8-mode mixture, DDIM, N = 256, mean of 4 seeds. `h = None` is the no-BD
baseline. MMD values around 0.003 are at the finite-reference floor.

| steps | eta | BD (h, T) | NLL | MMD | TV | unique | time (s) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 3 | 0 | — | 43.28 | 0.2486 | 0.080 | 1.00 | 0.000 |
| 5 | 0 | — | 10.33 | 0.0408 | 0.080 | 1.00 | 0.005 |
| 5 | 0 | 0.7, 0.1 | 8.59 | **0.0378** | 0.077 | 0.84 | 0.010 |
| 10 | 0 | — | 2.18 | 0.0060 | 0.080 | 1.00 | 0.004 |
| 10 | 0 | 0.3, 0.1 | 2.01 | 0.0057 | 0.077 | 0.93 | 0.018 |
| 10 | 1 | — | 2.29 | 0.0062 | 0.079 | 1.00 | 0.004 |
| 10 | 1 | 0.7, 1.0 | 1.20 | 0.0052 | 0.076 | 0.80 | 0.013 |
| 20 | 1 | — | 0.91 | 0.0032 | 0.052 | 1.00 | 0.019 |
| 20 | 1 | 0.3, 1.0 | 0.75 | 0.0029 | 0.054 | 0.96 | 0.040 |
| 50 | 1 | — | 0.735 | 0.0031 | 0.063 | 1.00 | 0.029 |
| 50 | 1 | 0.3, 0.1 | 0.734 | 0.0028 | 0.059 | 1.00 | 0.157 |

Readings:

- At 5 steps BD gives a ~7% MMD improvement. At 10 steps the best BD is ~5%
  better than baseline (eta = 0) or ~16% better (eta = 1, heavy BD), but the
  latter comes with 20% exact duplicates. At 20–50 steps the differences are at
  the MMD floor.
- The NLL gains at large `T` are not free: unique fraction falls to 0.2–0.6.
  NLL rewards concentration, so it is not a reliable headline metric here.
- **Matched compute is decisive.** 5-step BD with `T = 0.1` costs 10 ms and gets
  MMD 0.0378; the 10-step baseline costs 4 ms and gets 0.0060 — six times better
  and 2.3x faster. 10-step BD (`T = 0.1`, 18 ms, MMD 0.0057) versus 20-step
  baseline (19 ms, MMD 0.0032): same wall-clock, baseline is 1.8x better.
  In other words, the next reverse step is a better investment than a BD event.
- At N = 16 BD generally *worsens* MMD (e.g. 10 steps, eta = 1: 0.0377 → 0.041–0.048)
  and TV; at N = 64 the changes are small and mostly negative. The KDE needs a
  population large enough to be informative, which conflicts with cheap sampling.
- NFE is identical to baseline in every row.

### 3.2 The paper's ULA regime: real per-iteration gain, no general wall-clock gain

4-mode anisotropic mixture, fixed-target ULA, step size 1e-3, BD applied every
step with `h = 0.3`, `T = steps * 1e-3` (the paper's coupling), mean of 4 seeds.
MSE is the squared error of the sample mean; time is measured on CPU.

| N | method | steps | TV | MSE | time (s) |
|---:|---|---:|---:|---:|---:|
| 16 | ULA | 5,000 | 0.688 | 8.28 | 0.41 |
| 16 | BD | 5,000 | 0.688 | 8.49 | 1.20 |
| 16 | ULA | 20,000 | 0.358 | 2.90 | 1.59 |
| 64 | ULA | 5,000 | 0.614 | 7.73 | 0.51 |
| 64 | BD | 5,000 | 0.345 | 3.23 | 2.02 |
| 64 | ULA | 20,000 | 0.316 | 2.72 | 1.99 |
| 128 | ULA | 5,000 | 0.606 | 7.44 | 0.59 |
| 128 | **BD** | **5,000** | **0.223** | **1.45** | **2.47** |
| 128 | ULA | 20,000 | 0.308 | 2.66 | 2.31 |
| 128 | ULA | 50,000 | 0.097 | 0.37 | 5.78 |
| 1000 | ULA | 5,000 | 0.595 | 7.40 | 1.54 |
| 1000 | BD | 5,000 | 0.089 | 0.19 | 39.3 |
| 1000 | ULA | 20,000 | 0.313 | 2.75 | 5.90 |
| 1000 | ULA | 100,000 | 0.033 | 0.03 | 29.5 |

Readings:

- **Per iteration, BD is a genuine mixing accelerator for N >= 64.** At N = 128
  it reaches TV 0.223 in 5k iterations; ULA needs roughly 26k iterations for the
  same TV (~5x fewer). At N = 1000 it reaches 0.089 vs ULA's 0.595 at 5k
  (~10x fewer). At N = 16 there is no gain.
- **The cost per iteration grows faster than the iteration saving.** BD/ULA time
  per 5k iterations: 2.9x at N = 16, 4.0x at N = 64, 4.2x at N = 128, 25x at
  N = 1000. The KDE is O(N^2 D); ULA is O(N D).
- **Equal wall-clock:** ULA wins at N = 16 and N = 64; BD has a narrow, modest
  edge around N = 128 (~1.2–1.5x); ULA wins again at N = 1000 (BD 39 s for TV
  0.089; ULA reaches 0.108 in 15 s and 0.033 in 30 s).
- This diagnostic gives BD an analytic, essentially free `log pi_t`. In a
  diffusion model, evaluating `log pi_t` costs at least one model forward pass
  (or is impossible), so these timings are an *upper bound* on BD's usefulness.

The BD docstring's caveat is therefore the correct summary: BD accelerates
convergence of a *fixed-target* sampler, not a diffusion sampler.

### 3.3 High dimension: the KDE becomes self-dominated

Null test: draw N iid samples from a standard Gaussian, so the population is
already exactly right and any BD effect is pure finite-sample noise. One
intervention with `T = 1`; mean of 3 seeds. Values are for `h = 0.7`.

| d | N | self KDE mass | effective other neighbours | KDE/target rank corr. | unique | MMD before → after | NLL before → after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1024 | 8.5 | −0.80 | 1.00 | 0.82 | 0.0003 → 0.0026 | 2.84 → 2.61 |
| 8 | 1024 | 531 | −1.00 | 0.98 | 0.61 | 0.0003 → 0.0042 | 11.32 → 10.36 |
| 32 | 1024 | 1024 | −1.00 | 0.04 | 0.51 | 0.0003 → 0.0024 | 45.36 → 42.92 |
| 128 | 1024 | 1024 | −1.00 | 0.01 | 0.48 | 0.0003 → 0.0014 | 181.7 → 176.8 |
| 768 | 1024 | 1024 | −1.00 | 0.04 | 0.45 | 0.0003 → 0.0012 | 1090 → 1078 |

(MMD values use a median-scaled reference bandwidth and are comparable only
within a row.)

- "Self KDE mass" is the fraction of the KDE belonging to the particle's own
  kernel. When it is ~1, the KDE is essentially a sum of delta functions and
  `log KDE` carries no information.
- For d >= 32, even `h` = 0.1–0.7 (and median-based `h` up to ~78) leaves the
  self mass ~1. The ranking of KDE vs target is uninformative, so
  `beta ≈ -log pi_t(x) + const`: BD simply copies high-density particles.
- Consequently diversity falls (unique 0.45–0.51) and MMD gets 4–14x worse,
  while NLL *improves* because the population contracts onto high-density
  regions. This is the same NLL/MMD disagreement seen in the image runs.
- At d = 768, bandwidths 0.1, 0.3, 0.7, 19.6, 39.2 and 78.4 all give unique
  0.45–0.46 and MMD 0.0012–0.0013. **Bandwidth tuning cannot fix this regime.**
- This explains the image result in Section 3.5: all bandwidths give identical
  metrics.

### 3.4 The genuine positive regime: a badly biased score

2-D toy mixture, 10 DDIM steps, `eta = 1`, N = 256, 4 seeds. The model score is
`s_model = c * s_true`; `c = 1` is exact, `c < 1` under-disperses the score,
`c > 1` over-disperses it. BD uses `h = 0.7`, `T = 1`.

| model | baseline NLL | baseline MMD | BD NLL | BD MMD | BD unique |
|---|---:|---:|---:|---:|---:|
| exact (c = 1) | 2.29 | 0.0062 | 1.20 | 0.0052 | 0.80 |
| c = 0.5 | 235.8 | 0.1667 | **6.47** | **0.0260** | 0.65 |
| c = 0.7 | 5.37 | 0.0131 | 2.57 | **0.0064** | 0.79 |
| c = 0.8 | 2.82 | 0.0047 | 1.86 | 0.0052 | 0.81 |
| c = 1.2 | 1.93 | 0.0046 | 1.02 | 0.0052 | 0.82 |
| c = 1.3 | 1.79 | 0.0036 | 1.12 | 0.0036 | 0.83 |
| c = 1.5 | 2.10 | 0.0038 | 1.19 | 0.0052 | 0.82 |

- With a severe error (`c = 0.5`) BD cuts MMD by 6.4x and NLL by 36x: it really
  is correcting the population toward the target.
- With mild error (`c >= 0.8`) the MMD gain disappears or reverses, even though
  NLL keeps improving. This is the collapse bias again.
- Stochasticity is required. With `eta = 0` the same `c = 0.5` run has baseline
  MMD 0.286 and BD MMD 0.165 with unique fraction 0.04: heavy BD plus a
  deterministic sampler destroys diversity and never reaches the target.
- So BD is a **model-error/population correction**, and only when the error is
  large relative to the resampling noise. A well-trained diffusion model is
  closer to `c = 0.9–1.1`, precisely where BD does nothing.

### 3.5 Image-facing run: the empirical target is not evidence about image models

`--empirical-data-target` replaces the model with `target-score`
(`src/diffusion/cli.py`), i.e. the "generated" samples are driven by the score of
the finite empirical target itself. The BD term is then evaluated against the
same target density, and quality is measured against the same reference images.
This can test whether BD preserves a known finite target; it says nothing about
a pretrained CIFAR/CelebA model.

`toy-images`, DDIM, 25 steps, N = 64, 256 references:

| run | bandwidth | T | MMD | NLL | unique | time (s) |
|---|---:|---:|---:|---:|---:|---:|
| no BD | — | — | **0.0062** | −2825.5 | 56/64 | 0.10 |
| BD | 0.3 | 0.01 | 0.0071 | −2825.5 | 55/64 | 0.15 |
| BD | 0.3 | 0.1 | 0.0184 | −2825.5 | 32/64 | 0.16 |
| BD | 0.3 | 1.0 | 0.0619 | −2825.4 | 11/64 | 0.14 |
| BD | 10 | 0.1 | 0.0188 | −2825.5 | 32/64 | — |
| BD | 10 | 1.0 | 0.0865 | −2825.8 | 10/64 | — |

`toy-images`, Langevin, 100 steps, N = 16, 32 references:

| run | bandwidth | T | MMD | NLL | unique |
|---|---:|---:|---:|---:|---:|
| no BD | — | — | **0.149** | −2594.2 | 16/16 |
| BD | 0.3 | 1.0 | 0.759 | −2609.9 | 16/16 |
| BD | 20 | 1.0 | 0.759 | −2609.9 | 16/16 |
| BD | 1000 | 1.0 | 0.759 | −2609.9 | 16/16 |

- Identical results for bandwidths spanning four orders of magnitude confirm the
  high-dimensional self-domination in Section 3.3.
- BD makes MMD 5–14x worse and, at larger `T`, collapses the number of distinct
  tiles. NLL is flat or slightly better, which is the misleading-metric problem.
- Because the sampler is the target-score model, these numbers cannot support
  any claim about accelerating a *learned* diffusion model.

---

## 4. Why the stated goal is not reachable with this design

The project goal is to "reduce the number of diffusion sampling steps required".
BD fails that goal for four independent reasons:

1. **No transport.** A reverse step is a map from `pi_{t+1}` to `pi_t` (plus
   noise). BD only resamples within `pi_t`. Skipping a reverse step leaves
   particles at the wrong noise level; BD cannot move them to the right one.
   Hence NFE cannot go down.
2. **No target density.** The algorithm needs `log pi_t`, but a pretrained model
   provides only `grad log pi_t`. The repo's only exact densities are toy and
   empirical targets. A KDE of the particles is the *proposal*, not the target;
   using it in place of `pi_t` is circular, and in high dimension it is exactly
   the failure mode in Section 3.3.
3. **No mass creation.** BD duplicates existing particles; it cannot put mass in
   a mode that the population never reached. With DDIM `eta = 0`, a duplicate is
   permanent (identical particles follow the same deterministic map), which is
   why unique fraction collapses.
4. **Cost.** Each intervention is O(N^2 D) for the KDE plus a `log pi_t`
   evaluation. This is cheap for a 2-D analytic target and expensive for images,
   and it grows quadratically in the population.

The paper's result is not contradicted by this: it accelerates *mixing* of a
fixed-target ULA chain, where each iteration's score evaluation is the same and
BD reduces the number of iterations to stationarity. Diffusion adds a changing
target and removes the known density, and those are precisely the assumptions
that make BD work.

---

## 5. What a sound version of the idea looks like

If the project wants to keep BD, the defensible framing is **target-aware
particle correction**, not sampler acceleration.

1. **Use BD where `log pi_t` is genuinely known.** The natural case is
   conditional generation / inverse problems: for a Gaussian likelihood,
   `log pi_t(x | y) = log pi_t(x) + log p(y | x_t) + const`, and the likelihood
   term is cheap and exact. Test inpainting or denoising on a small pretrained
   diffusion model, with BD between reverse steps.
2. **Replace the KDE with the model's transition kernel.** In SMC for diffusion,
   the proposal is the reverse transition and the weights come from the known
   likelihood. A KDE of the particles is a poor density estimator in image
   dimension; the transition kernel is the object the model actually provides.
   This also turns BD into a principled resampling step rather than an ad hoc
   duplication rule.
3. **Report NFE and diversity-sensitive metrics.** NLL alone rewards collapse.
   Use MMD/recall/coverage/FID and the exact-duplicate fraction, always at
   matched NFE and matched wall-clock. A "win" that lowers NLL while halving
   unique samples is a loss.
4. **For pure unconditional speed, use a different mechanism.** If the course
   deliverable needs "fewer steps at equal quality", distillation, DPM-Solver /
   higher-order ODE solvers, or adaptive step sizes are the right tools. BD can
   remain as an ablation/analysis chapter, not the acceleration claim.
5. **Keep the negative results.** The high-dimensional analysis (Section 3.3)
   and the matched-compute comparison (Section 3.1) are genuinely useful
   findings: they explain *why* a plausible algorithm does not transfer, and
   they give a concrete diagnostic (self-KDE mass, unique fraction) for future
   work.

If the project must claim step reduction with BD, the minimum evidence bar is:

- a pretrained model on a real dataset (not an empirical target);
- BD quality at NFE = k compared with the baseline at NFE = k *and* at the same
  wall-clock;
- a density source that is not derived from the evaluation set;
- MMD/FID/recall plus unique fraction, not just NLL.

None of the current image runs meet that bar.

---

## 6. Bottom line for the report

- The BD code is correct; the experiments are not failing because of a bug.
- The paper's positive result is real in its own regime: fixed target, known
  density, low dimension, large population, many iterations. This repo
  reproduces it (Section 3.2).
- The transfer to diffusion acceleration fails for structural reasons, not
  tuning reasons: no NFE reduction, no known `log pi_t`, no mode creation, and
  an O(N^2 D) KDE that is statistically useless in high dimension.
- There is one honest positive result: with a known target and a badly biased
  score, BD corrects the population and improves MMD (Section 3.4). That is a
  model-error-correction story, and it is worth reporting as such.
- Recommendation: reframe the project as an empirical study of when
  target-aware resampling helps, and use a standard acceleration method
  (distillation/solver) for the "accelerated sampling" deliverable. If BD is
  kept as the acceleration mechanism, move to a conditional/inverse-problem
  setting where the target density is known and use SMC weights instead of a
  KDE.

---

## 7. Reproducing the numbers

New diagnostic scripts (no project source was modified):

```bash
# 1. Toy diffusion sweep (Section 3.1)
python3 tests/bd_diffusion_sweep.py \
  --counts 16,64,256 --steps 3,5,10,20,50 --etas 0,1 \
  --bandwidths 0.3,0.7,1.2 --total-times 0.1,1,10 \
  --output samples/bd-diffusion-sweep/metrics.json

# 2. Paper ULA setting, per-iteration and equal-wall-clock (Section 3.2)
python3 tests/bd_paper_equal_time.py \
  --counts 16,64,128,1000 --seeds 7,17,29,41 \
  --bd-steps 5000 --ula-steps 20000,50000,100000 \
  --output samples/bd-paper-equal-time/metrics.json

# 3. High-dimensional KDE self-domination (Section 3.3)
python3 tests/bd_kde_dim_diagnostic.py \
  --dims 2,8,32,128,768 --counts 256,1024 --seeds 7,17,29 \
  --total-times 0.1,1 --output samples/bd-kde-dim/metrics.json
```

The biased-score runs (Section 3.4) used the same toy pipeline with a wrapper
model whose score is `c * target.score`; the empirical-target runs (Section 3.5)
were produced with `sample.py --empirical-data-target --target-dataset toy-images`
and the `--birth-death-*` flags. Existing baselines:

```bash
python3 tests/toy_bd_experiment.py --sampler all --steps 50 --count 256
python3 tests/bd_paper_toy_diagnostic.py --count 128 --steps 5000 \
  --bandwidths 0.3,0.7,1.2
```

Raw JSON from the runs discussed here was written under `/tmp/opencode/`
(`sweep256.json`, `sweep_smallN.json`, `paper_n_seeds.json`, `ula_long_4seed.json`,
`kde_dim.json`, `model_error.json`, `model_error_supp.json`, and the
`grid-*.metrics.json` / `img-*.metrics.json` files). The scripts above regenerate
the three main tables; `samples/` is gitignored.
