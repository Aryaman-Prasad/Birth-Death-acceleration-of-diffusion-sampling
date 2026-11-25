# Accelerating Diffusion Sampling with Birth--Death Dynamics: Manifold Projections & Non-KDE Paradigms

**Course Project — IE 643: Deep Learning (Theory & Practice), IIT Bombay**  
**Team `idk`:** Aryaman Prasad (`24B0928`) & S Sabarinath (`24B1052`)  
*Department of Industrial Engineering & Operations Research, IIT Bombay*

> **Project Deliverables:**  
> - 📄 [**2-Page Novelty Assessment Report (PDF)**](idk_IE643_NoveltyAssessment_report.pdf)  
> - 🖼️ [**A0 Course Project Poster (PDF)**](idk_IE643_CourseProject_Poster.pdf)  
> - 📓 [**Comprehensive Benchmark Notebook (`run_all_bd_benchmarks.ipynb`)**](run_all_bd_benchmarks.ipynb)  
> - 📚 **Documentation:** [Framework Guide (`DOCS.md`)](DOCS.md) | [Birth--Death Notes (`DOCS-BD.md`)](DOCS-BD.md) | [Equations & Extensions (`BD-EXTENSIONS.md`)](BD-EXTENSIONS.md) | [Findings (`FINDINGS.md`)](FINDINGS.md)

---

## Overview: Prior Work & What Broke Down

Standard diffusion samplers (**DDPM**, **DDIM**) generate high-fidelity samples through iterative spatial transport drift along learned score fields $\nabla \log p_t(x)$. Under fast sampling budgets ($T \le 25$ steps), particles frequently become trapped in low-density valleys between separated data modes.

Prior theoretical work introduced continuous-time **Wasserstein--Fisher--Rao (WFR)** birth--death dynamics (Lu et al., 2019), pairing Langevin transport with non-local birth (duplication) and death (pruning) reactions. While effective on low-dimensional synthetic manifolds, porting naive birth--death directly to deep image diffusion triggers three severe pathologies:

1. **Curse of Dimensionality & Distance Concentration:** In deep generative models, target densities are intractable and approximated via empirical Kernel Density Estimation (KDE) in raw pixel space ($D = 3{,}072$ on CIFAR-10, $D = 196{,}608$ on CelebA-HQ 256). Pairwise Euclidean distances concentrate sharply around a spherical shell ($\sigma\sqrt{2D}$), turning Gaussian kernel weights into uniform noise and degrading 50-step CIFAR-10 $\text{MMD}^2$ by over $5\times$ ($0.2650$ vs. $0.0590$ baseline).
2. **Late-Stage Mode Collapse:** Applying unconstrained particle pruning at fine noise levels ($t < 100$) destroys high-frequency visual details and collapses trajectories onto duplicate modes.
3. **Quadratic Computational Overhead:** Naive empirical KDE computes pairwise pixel distances against cached reference banks at $\mathcal{O}(NMD)$ cost, offsetting few-step sampling speedups.

---

## Novel Contributions

To resolve these high-dimensional bottlenecks, we designed and benchmarked two manifold projection methods and three reference-free non-KDE paradigms under gentle temporal regulation:

- **1. Manifold Reduced-KDE (`src/diffusion/acceleration/reduced_kde_birth_death.py`):**  
  Extracts time-conditioned 128D representations from the native DDPM U-Net bottleneck (`DDPMUNetEncoderTrunk`) and compresses them to $d=25$ dimensions via offline SVD (`FeatureReducer`). Eliminates distance concentration without relying on clean-image external classifiers that misinterpret noisy latents.
- **2. Direct Linear Pixel-PCA (`--pca` / `--unet-layer pca`):**  
  Bypasses convolutional feature extraction by projecting flattened pixels directly onto principal eigenvectors $V_d \in \mathbb{R}^{D \times d}$ ($d \in \{10, 25, 50\}$) via a single GPU Tensor Core GEMM ($<0.015\text{ ms}$, $76.8\times$ FLOP cut).
- **3. SMC Reverse Transition Weighting (`src/diffusion/acceleration/smc_transition_modifier.py`):**  
  Scores reverse transitions analytically using $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta, \sigma_t^2 I)$ in $\mathcal{O}(N)$ linear time, completely eliminating reference set caching.
- **4. Continuous Normalizing Flow BD (`src/diffusion/acceleration/flow_birth_death.py`):**  
  Integrates exact or Hutchinson-approximated trace divergence along ODE paths for flow-based density evaluation (`ContinuousNormalizingFlowTarget`, `FlowBirthDeathAcceleration`).
- **5. Particle Filtering Explorer (`src/diffusion/acceleration/particle_filter_explorer.py`):**  
  Branches candidate reverse steps ($N \to N \times B$, default $B=2$) and selects optimal paths using transition likelihood potentials plus optional batch-dispersion bonuses.
- **6. Gentle Temporal Regulation:**  
  Disabling birth--death for $t < 100$ (`--bd-min-timestep 100`), capping pseudo-time per step (`--bd-max-time 0.005`), capping event probability at $\le 25\%$ (`--bd-max-event-prob 0.25`), and preserving stochasticity (`--eta 1.0`) permanently eliminates mode collapse across architectures.

---

## Key Empirical Results

### 1. 2D Gaussian Mixture Oracle Verification ($D = 2$, 8 Symmetric Modes)
Before scaling to pixels, we verified birth--death dynamics on an exact analytical 8-mode Gaussian ring ($r=2.0, \sigma^2=0.04$):
- **100% Mode Coverage** across all tested methods and sampling budgets ($T \in \{10, 25, 50\}$).
- **Rapid Convergence:** Particle Filtering achieves the lowest error at 10 steps ($\text{NLL} = \mathbf{2.1251}$, $\text{MMD}^2 = \mathbf{0.00129}$ vs. DDIM $2.1913 / 0.00444$); KDE Birth--Death reaches $\text{NLL} = \mathbf{0.6600}$ at 25 steps (a **32% error cut** vs. DDIM $0.9684$).
- **Score Bias Recovery:** Under biased score fields, birth--death reweighting restores mode symmetry ($\text{NLL} = 0.49$ vs. DDIM $17.82$, a $36\times$ improvement).

### 2. CIFAR-10 ($D = 3{,}072$) & CelebA-HQ 256 ($D = 196{,}608$) Benchmarks

| Regime / Method | Steps | Latent Space | Throughput | Diversity | MMD² ↓ |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **CIFAR-10** Baseline DDIM ($\eta=0$) | 25 | 3072D Pixels | 10.58 img/s | 100% | 0.0646 |
| **CIFAR-10** Baseline DDIM ($\eta=1$) | 25 | 3072D Pixels | 10.52 img/s | 100% | 0.0529 |
| **CIFAR-10 Reduced-KDE (Ours)** | 25 | 25D U-Net Bottleneck | 6.67 img/s | **100%** | **0.0446** *(global best)* |
| **CIFAR-10 Direct Pixel-PCA ($d=25$)** | 10 | 25D Linear PCA | **28.69 img/s** | 87.5% | **0.0851** *(beats DDIM $\eta=0$ 0.0926)* |
| **CIFAR-10 Direct Pixel-PCA ($d=10$)** | 10 | 10D Linear PCA | **28.82 img/s** | **100%** | **0.0875** *(beats DDIM $\eta=0$ 0.0926)* |
| **CIFAR-10 SMC Reverse Transition** | 10 | $\mathcal{O}(N)$ Analytical | 18.87 img/s | 87.5% | **0.0845** *(beats DDIM $\eta=0$ 0.0926)* |
| **CIFAR-10** Baseline DDIM ($\eta=0$) | 50 | 3072D Pixels | 5.25 img/s | 100% | 0.0590 |
| **CIFAR-10 Particle Filter Explorer** | 50 | Branching ($B=2$) | 3.66 img/s | **93.8%** | **0.0493** *(beats DDIM $\eta=0$)* |
| **CIFAR-10 Reduced-KDE (Ours)** | 50 | 25D U-Net Bottleneck | 3.18 img/s | **100%** | **0.0482** *(global best)* |
| **CIFAR-10** Naive Pixel-KDE (Failure) | 50 | 3072D Pixels | 5.25 img/s | 93.8% | 0.2650 *(distance concentration)* |
| **CelebA-HQ 256 Direct Pixel-PCA ($d=10$)** | 25 | 10D Linear PCA | **0.41 img/s** | **93.8%** | **0.0781** *(4×4 face grid)* |

*Note: Direct Linear Pixel-PCA also reaches up to **58.47 img/s** on CIFAR-10 at 5 sampling steps with zero convolutional feature overhead.*

---

## Installation & Setup

Requires **Python >= 3.10**. Dependencies are specified in [`pyproject.toml`](pyproject.toml).

```bash
# Editable base installation (PyTorch, torchvision, Pillow)
python3 -m pip install -e .

# Optional: Pretrained Hugging Face Diffusers models (e.g., google/ddpm-cifar10-32)
python3 -m pip install -e ".[pretrained]"

# Optional: FID evaluation support (torchmetrics[image])
python3 -m pip install -e ".[fid]"
```

> **Kaggle / Cloud GPU Note:** Ensure the notebook **Accelerator** is set to **GPU** (`T4` / `P100`) and avoid overwriting the preinstalled CUDA build of PyTorch (you can run `python3 -m pip install --no-deps -e .` on Kaggle, or pass `--device auto` so CPU-only environments fall back gracefully).

### Downloading Datasets (`wget`)

> **Note:** Synthetic datasets (`--dataset toy-mixture`, `--dataset toy-images`) and the 2D oracle (`--toy-mixture-oracle`) are generated in memory and require no external downloads.

#### 1. CIFAR-10 ($32 \times 32$, 50k train / 10k test)
Download and extract into `data/` using the fast GitHub CDN mirror configured in [`src/diffusion/datasets/cifar10.py`](src/diffusion/datasets/cifar10.py):

```bash
mkdir -p data
wget -O data/cifar-10-python.tar.gz https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz
tar -xzf data/cifar-10-python.tar.gz -C data
```

*(Official University of Toronto fallback mirror: `wget -O data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`)*

#### 2. CelebA-HQ 256 ($256 \times 256$, 30k faces)
Download and extract into `data/celeba/` using Kaggle's dataset download endpoint for `badasstechie/celebahq-resized-256x256` (as used in [`run_all_bd_benchmarks.ipynb`](run_all_bd_benchmarks.ipynb)). [`ImageFolderDataModule`](src/diffusion/datasets/image_folder.py) supports both class-subfolder and flat image directories:

```bash
mkdir -p data/celeba
wget -L -O data/celebahq-256.zip https://www.kaggle.com/api/v1/datasets/download/badasstechie/celebahq-resized-256x256
unzip -q data/celebahq-256.zip -d data/celeba
```

*(If Kaggle prompts for authentication or you prefer Python, use `python3 -c 'import kagglehub; print(kagglehub.dataset_download("badasstechie/celebahq-resized-256x256"))'` or `--target-download` / `--download` for automatic CIFAR-10 fetching.)*

---

## Complete Running Syntax & CLI Reference

### 1. Training Baseline Diffusion Models (`train.py`)

```bash
# Train on CIFAR-10 (downloads automatically with --download)
python3 train.py --dataset cifar10 --data-dir data --download --epochs 100 --batch-size 128 --output checkpoints/ddpm.pt

# Train on a custom ImageFolder dataset (e.g., CelebA-HQ)
python3 train.py --dataset imagefolder --data-dir data/celeba --image-size 64 --epochs 50 --batch-size 64 --output checkpoints/celeba.pt

# Quick dry-run on synthetic toy images or 2D Gaussian mixture
python3 train.py --dataset toy-images --toy-size 64 --image-size 16 --epochs 1 --batch-size 8 --timesteps 20 --scheduler cosine --base-channels 16 --output checkpoints/toy-images.pt
python3 train.py --dataset toy-mixture --toy-size 256 --epochs 10 --batch-size 64 --timesteps 20 --scheduler cosine --toy-hidden 128 --output checkpoints/toy-mixture.pt
```

<details>
<summary>All <code>train.py</code> CLI flags</summary>

- `--dataset`: `cifar10`, `imagefolder`, `toy-images`, or `toy-mixture` (default: `cifar10`)
- `--data-dir`, `--download`, `--image-size`, `--toy-size`, `--toy-seed`
- `--model`: `auto`, `simple-unet`, `toy-mlp`, `huggingface-unet`
- `--base-channels` (default: `64`), `--toy-hidden` (default: `128`)
- `--pretrained-model-id`, `--pretrained-revision`, `--pretrained-unet-subfolder`, `--pretrained-scheduler-subfolder`
- `--scheduler`: `auto`, `linear`, `cosine`, `huggingface`; `--timesteps` (default: `1000`), `--beta-start` (`1e-4`), `--beta-end` (`2e-2`)
- `--epochs` (default: `100`), `--batch-size` (`128`), `--learning-rate` (`2e-4`), `--grad-clip` (`1.0`), `--num-workers` (`4`), `--save-every` (`10`)
- `--output` (default: `checkpoints/ddpm.pt`), `--device` (`auto`, `cuda`, `cpu`)

</details>

---

### 2. Core Sampling & Empirical Birth--Death (`sample.py`)

`sample.py` requires exactly one model/target source (`--checkpoint`, `--pretrained-model-id`, `--toy-mixture-oracle`, `--empirical-data-target`, or `--bd-unet`).

```bash
# Baseline DDPM (1000 steps) and DDIM (50 steps) from a trained checkpoint
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler ddpm --count 16 --output samples/ddpm.png
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler ddim --steps 50 --eta 0.0 --seed 7 --count 16

# Sample directly from pretrained Hugging Face DDPM U-Net (google/ddpm-cifar10-32)
python3 sample.py --pretrained-model-id google/ddpm-cifar10-32 --sampler ddim --steps 25 --eta 1.0 --count 16 --device auto

# Annealed Langevin or fixed-target Langevin sampling
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler annealed-langevin --steps 20 --langevin-steps-per-level 5

# 2D Gaussian Mixture Oracle with Birth-Death dynamics
python3 sample.py --toy-mixture-oracle --sampler ddim --steps 25 --eta 1.0 --count 512 \
    --birth-death --bd-every-k 2 --bd-total-time 0.1 --bd-kernel-bandwidth 0.3 --output samples/toy_bd.pt

# Pretrained CIFAR-10 U-Net + Empirical Pixel-KDE Birth-Death (--bd-unet) with gentle regulation
python3 sample.py --bd-unet --pretrained-model-id google/ddpm-cifar10-32 --target-dataset cifar10 --target-reference-count 256 \
    --sampler ddim --steps 25 --eta 1.0 --count 16 --bd-min-timestep 100 --bd-max-time 0.005 --bd-max-event-prob 0.25 --bd-every-k 1

# Empirical data target with FID-2048 evaluation
python3 sample.py --empirical-data-target --target-dataset cifar10 --target-download --sampler ddim --steps 25 --eta 1.0 \
    --birth-death --fid --fid-dataset cifar10 --fid-samples 1000 --fid-batch-size 64 --device auto
```

<details>
<summary>All Birth--Death & Gentle Regulation flags in <code>sample.py</code></summary>

- `--birth-death`: enable fixed-population BD updates (`ddpm`, `ddim`, `langevin`, `ula`)
- `--bd-kernel-bandwidth` (default: `0.3`), `--bd-every-k` (default: `1`), `--bd-total-time` (default: `1.0`)
- `--bd-min-timestep` (default: `100`), `--bd-max-timestep` (default: `None`), `--bd-max-time` (default: `0.005`), `--bd-max-event-prob` (default: `0.25`)
- `--add-noise [STD]`: optional perturbation noise on cloned particles (default `0.05` if flag passed without value)

</details>

---

### 3. Novel Acceleration Sampling (`sample_reduced_kde.py`)

Supports **Manifold Reduced-KDE** (`reduced_kde`), **Direct Linear Pixel-PCA** (`--pca`), **SMC Reverse Transition** (`smc_transition`), **Particle Filtering Explorer** (`particle_filter`), and **Flow Birth--Death** (`flow_exact`).

```bash
# 1. Manifold Reduced-KDE (25D U-Net Bottleneck + offline SVD)
python3 sample_reduced_kde.py --empirical-data-target --target-dataset cifar10 --target-download \
    --sampler ddim --steps 25 --eta 1.0 --birth-death --kde_reduce 25 --unet-layer bottleneck --projection-method pca --count 16 --device auto

# 2. Direct Linear Pixel-PCA (zero convolutions, single GPU Tensor Core GEMM)
python3 sample_reduced_kde.py --empirical-data-target --target-dataset cifar10 \
    --sampler ddim --steps 10 --eta 1.0 --birth-death --pca --kde_reduce 25 --count 16 --device auto

# 3. Direct Pixel-PCA on high-resolution ImageFolder (e.g., CelebA-HQ 256x256, 4x4 grid)
python3 sample_reduced_kde.py --empirical-data-target --target-dataset imagefolder --target-data-dir data/celeba --target-image-size 256 \
    --sampler ddim --steps 25 --eta 1.0 --birth-death --pca --kde_reduce 10 --count 16 --nrow 4 --device auto

# 4. SMC Reverse Transition Weighting (O(N), no reference caching)
python3 sample_reduced_kde.py --pretrained-model-id google/ddpm-cifar10-32 \
    --sampler ddim --steps 10 --eta 1.0 --acceleration-method smc_transition --smc-temp 1.0 --bd-every-k 2 --bd-total-time 0.01 --count 16 --device auto

# 5. Particle Filtering Solution Space Explorer (branching B=2)
python3 sample_reduced_kde.py --pretrained-model-id google/ddpm-cifar10-32 \
    --sampler ddim --steps 50 --eta 1.0 --acceleration-method particle_filter --filter-branch-factor 2 --bd-every-k 2 --count 16 --device auto

# 6. Flow Birth-Death (flow_exact) on 2D Toy Oracle
python3 sample_reduced_kde.py --toy-mixture-oracle --sampler ddim --steps 25 --eta 1.0 \
    --acceleration-method flow_exact --count 512 --output samples/toy_flow.pt
```

<details>
<summary>All novel acceleration flags in <code>sample_reduced_kde.py</code></summary>

- `--acceleration-method`: `reduced_kde` (default), `smc_transition`, `particle_filter`, `flow_exact`
- `--kde_reduce` / `--kde-reduce`: target latent dimension (default: `25`)
- `--pca` / `--pixel-pca`: shortcut to enable Direct Linear Pixel-PCA (`--unet-layer pca --projection-method pca`)
- `--unet-layer` / `--unet-backbone` / `--cnn-backbone`: `bottleneck` (default), `down_block2`, `down_block1`, or `pca`
- `--projection-method`: `pca` (default), `orthogonal`, `linear`, `mlp`
- `--noise-references` / `--no-noise-references`: re-noise empirical references at timestep $t$ (default: `True`)
- `--smc-temp` (default: `1.0`), `--filter-branch-factor` (default: `2`)

</details>

---

### 4. Automated Evaluation & Benchmark Suites (`scripts/` & `benchmark.py`)

```bash
# Baseline DDPM & DDIM throughput / timing sweep
python3 benchmark.py --pretrained-model-id google/ddpm-cifar10-32 --steps 100 50 25 10 --eta 0.0 --count 16 --device auto --output-dir baseline_data

# 2D Gaussian Mixture Ring verification + scatter & bar plots
python3 scripts/run_and_plot_toy.py --count 512 --seed 42 --device auto --output-dir toy_results

# Unified Multi-Method Benchmark (Baseline, Pixel-KDE, Reduced-KDE, Direct Pixel-PCA, SMC, Particle Filter, Flow-Exact)
python3 scripts/run_reduced_kde_eval.py --dataset cifar10 --download --kde_reduce 25 --pca --compare-all-paradigms \
    --steps 10 25 50 --bd-min-timestep 100 --bd-max-time 0.005 --bd-max-event-prob 0.25 --count 16 --no-fid --device auto --output-dir reduced_kde_results

# Direct Pixel-PCA-only sweep (fast throughput benchmark across steps)
python3 scripts/run_reduced_kde_eval.py --dataset cifar10 --pca-only --kde_reduce 10 --steps 5 10 25 50 --no-fid --device auto

# CelebA-HQ 256 extreme-scale Pixel-PCA evaluation (4x4 grids)
python3 scripts/run_reduced_kde_eval.py --dataset imagefolder --data-dir data/celeba --image-size 256 \
    --pca-only --kde_reduce 10 --steps 25 --count 16 --no-fid --device auto

# Pretrained U-Net + Empirical KDE Birth-Death evaluation (BD-UNet)
python3 scripts/run_bd_unet_eval.py --steps 10 25 50 --reference-count 512 --eta 1.0 \
    --bd-min-timestep 100 --bd-max-time 0.005 --bd-max-event-prob 0.25 --device auto --output-dir bd_unet_results
# (Add --fid --fid-samples 512 for FID evaluation)

# Standalone Non-KDE Paradigm Benchmarks
python3 scripts/run_smc_eval.py --model-id google/ddpm-cifar10-32 --dataset cifar10 --steps 10 25 50 \
    --smc-every-k 2 --smc-total-time 0.01 --smc-temp 1.0 --bd-min-timestep 100 --no-fid --device auto --output-dir smc_benchmark_results

python3 scripts/run_particle_filter_eval.py --model-id google/ddpm-cifar10-32 --dataset cifar10 --steps 10 25 50 \
    --branch-factor 2 --filter-every-k 2 --filter-temp 1.0 --diversity-weight 0.1 --bd-min-timestep 100 --no-fid --device auto --output-dir pf_results

python3 scripts/run_exact_flow_eval.py --model-id google/ddpm-cifar10-32 --dataset cifar10 --steps 10 25 50 \
    --flow-every-k 2 --flow-total-time 0.01 --flow-bandwidth 0.3 --bd-min-timestep 100 --no-fid --device auto --output-dir flow_results

# Deliverable 4 Ablation Grid (steps, cadence k, budget tau, stochasticity eta)
python3 scripts/run_deliverable4_eval.py --dataset cifar10 --device auto --output-dir deliverable4_results
python3 scripts/run_deliverable4_eval.py --dataset cifar10 --quick --fid-samples 0 --device auto
```

---

### 5. Unit Tests & Diagnostics (`tests/`)

```bash
# Run unit test suites for Reduced-KDE, SMC, CNF Flow-BD, and Particle Filtering
pytest tests/test_reduced_kde_birth_death.py tests/test_novel_acceleration_methods.py

# Run standalone toy & high-dimensional diagnostic scripts
python3 tests/toy_bd_experiment.py
python3 tests/toy_acceleration_extensions.py
python3 tests/bd_kde_dim_diagnostic.py
python3 tests/bd_diffusion_sweep.py
python3 tests/bd_paper_equal_time.py
python3 tests/bd_paper_toy_diagnostic.py
```

---

## Repository Structure

```text
├── src/diffusion/
│   ├── acceleration/
│   │   ├── base.py                        # SamplingModifier & SamplingState interfaces
│   │   ├── birth_death.py                 # Naive/empirical target KDE Birth-Death
│   │   ├── reduced_kde_birth_death.py     # U-Net bottleneck Reduced-KDE & Direct Pixel-PCA
│   │   ├── smc_transition_modifier.py     # Analytical SMC reverse-transition BD modifier
│   │   ├── particle_filter_explorer.py    # Branching Particle Filtering Explorer
│   │   ├── flow_birth_death.py            # Continuous Normalizing Flow & Flow-BD
│   │   └── reverse_transition.py          # Reverse transition Gaussian density helpers
│   ├── datasets/                          # CIFAR-10, ImageFolder, Toy Images, Toy Mixture
│   ├── metrics/                           # MMD², FID, NLL, and throughput profiling
│   ├── models/                            # SimpleUNet, ToyMLP, HuggingFace U-Net wrapper
│   ├── samplers/                          # DDPM, DDIM, Annealed Langevin, Fixed-Target Langevin
│   ├── schedulers/                        # Linear, Cosine, HuggingFace schedulers
│   ├── targets/                           # Analytic Gaussian Mixture & Empirical Noised Targets
│   ├── cli.py                             # Root CLI backbone for train.py / sample.py
│   └── pipeline.py                        # Modular reverse sampling loop + modifiers
├── scripts/                               # Standalone evaluation & plotting suites
├── tests/                                 # Unit tests & high-dimensional diagnostics
├── train.py / sample.py / benchmark.py    # Baseline training, sampling, and benchmarking entrypoints
├── sample_reduced_kde.py                  # Reduced-KDE, Pixel-PCA & non-KDE CLI entrypoint
├── run_all_bd_benchmarks.ipynb            # End-to-end Kaggle/GPU benchmark suite notebook
├── idk_IE643_NoveltyAssessment_report.pdf # Compiled 2-page Novelty Assessment report
└── idk_IE643_CourseProject_Poster.pdf     # Compiled A0 Course Project poster
```

---

## License

This project is released under the [LICENSE](LICENSE) included in this repository.
