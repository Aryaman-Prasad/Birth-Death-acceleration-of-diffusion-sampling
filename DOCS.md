# Birth-death acceleration of diffusion sampling

IE 643 course project by Aryaman and Sabarinath.

The framework setup, toy-data dry runs, pretrained model options, and extension points are documented in [DOCS.md](DOCS.md).

```bash
python3 -m pip install -e .
python3 train.py --dataset toy-images --toy-size 64 --image-size 16 --epochs 1 --batch-size 8 --timesteps 20 --scheduler cosine --base-channels 16
```

# Framework guide

This repository implements the standard diffusion training and sampling baseline, plus an annealed Langevin sampler and an experimental, fixed-target birth-death particle update. The code is split so models, data sources, noise schedules, samplers, and optional sampling-time methods can be extended independently.

## Setup and running

Use Python 3.10 or newer. The dependency list and package metadata live in `pyproject.toml`; there is no separate requirements file.

Install the project and its dependencies once, from the repository root:

```bash
python3 -m pip install -e .
```

For optional pretrained Hugging Face Diffusers models, install the `pretrained` extra:

```bash
python3 -m pip install -e ".[pretrained]"
```

For optional FID evaluation, install the `fid` extra:

```bash
python3 -m pip install -e ".[fid]"
```

On Windows, use `py -3 -m pip install -e .` if `python3` is not on PATH. After installation, run the root-level scripts directly. They add `src/` to the import path, so no separate CLI command is required.

### Train on CIFAR-10

```bash
python3 train.py --dataset cifar10 --data-dir data --download --epochs 100
```

### Train on an image folder

Place images under class subfolders, as expected by `torchvision.datasets.ImageFolder`:

```text
data/my-images/
  class-a/image-001.png
  class-a/image-002.jpg
  class-b/image-003.jpg
```

Then run:

```bash
python3 train.py --dataset imagefolder --data-dir data/my-images --image-size 32
```

Images are resized/cropped to a square, converted to RGB tensors, and normalized to `[-1, 1]`. CIFAR-10 and ImageFolder currently provide the two dataset adapters. The U-Net baseline expects three-channel images.

Training options include `--model`, `--scheduler`, `--timesteps`, `--beta-start`, `--beta-end`, `--base-channels`, `--batch-size`, `--learning-rate`, `--epochs`, `--device`, and `--output`. `--model auto` selects the image U-Net for image datasets and the vector MLP for the toy mixture. `--scheduler auto` selects a linear schedule for locally initialized models and the matching repository schedule for a pretrained U-Net. Defaults favor a simple 32-pixel experiment; adjust batch size and model width for available hardware. Checkpoints are written to `checkpoints/ddpm.pt` by default, with model weights and configuration needed for sampling.

### Dry-run with toy data

The toy datasets use the same `DiffusionDataModule` interface as CIFAR-10 and ImageFolder and are generated locally without downloads:

```bash
python3 train.py --dataset toy-images --toy-size 64 --image-size 16 --epochs 1 --batch-size 8 --timesteps 20 --scheduler cosine --base-channels 16 --output checkpoints/toy-images.pt
python3 sample.py --checkpoint checkpoints/toy-images.pt --sampler ddim --steps 5 --count 8
```

`toy-images` creates a small in-memory dataset of simple RGB shapes and exercises the image U-Net path. `toy-mixture` creates a low-dimensional, eight-mode Gaussian mixture and automatically selects a small MLP denoiser:

```bash
python3 train.py --dataset toy-mixture --toy-size 256 --epochs 1 --batch-size 64 --timesteps 20 --scheduler cosine --output checkpoints/toy-mixture.pt
python3 sample.py --checkpoint checkpoints/toy-mixture.pt --sampler ddim --steps 10 --count 256 --output samples/mixture.pt
```

Vector samples are saved as a PyTorch tensor (`.pt`) for plotting or numerical checks. These small datasets make it practical to exercise training, checkpoint loading, sampling, and future algorithm changes without downloading CIFAR-10. They are debugging aids, not substitutes for image-quality benchmarks.

### Generate samples

Use the full ancestral DDPM chain:

```bash
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler ddpm --count 16
```

Or compare a smaller number of DDIM model evaluations:

```bash
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler ddim --steps 50 --eta 0 --seed 7
```

The `annealed-langevin` sampler is also available for epsilon-prediction models:

```bash
python3 sample.py --checkpoint checkpoints/ddpm.pt --sampler annealed-langevin --steps 100 --langevin-steps-per-level 2 --count 16
```

Here `--steps` is the number of descending noise levels; NFE is that count multiplied by `--langevin-steps-per-level`. The per-level step size is `base_step_size * (sigma_t / sigma_min)^2`; the base parameter defaults to `1e-7`, so the effective step at high noise can be much larger. Tune it for the model and schedule in use. Annealed Langevin is an additional score-based sampler and is not the DDPM/DDIM reverse update.

The fixed-noise `langevin` sampler is primarily provided for controlled target-density experiments. With an epsilon model it runs ULA against the model's score at one fixed noise level; it does not implement a conventional full image-generation trajectory. A small exact-score toy experiment can be run without training:

```bash
python3 sample.py --toy-mixture-oracle --sampler langevin --steps 1000 --count 256 --seed 7 --output samples/mixture-ula.pt
python3 sample.py --toy-mixture-oracle --sampler langevin --steps 1000 --count 256 --birth-death --bd-kernel-bandwidth 0.3 --seed 7 --output samples/mixture-bd-ula.pt
```

These commands use an analytic eight-mode Gaussian mixture, a matching exact epsilon predictor, and the same fixed-noise ULA updates; the second inserts birth-death resampling after each update. See [DOCS-BD.md](DOCS-BD.md) for the implemented rates, assumptions, limitations, and details on what is faithful to the paper. Birth-death is deliberately rejected for learned image models and for DDPM, DDIM, or annealed Langevin: those paths do not expose a justified stationary target log density in the current framework.

The sample grid is saved under `samples/`. `--steps` selects reverse updates for DDIM and fixed Langevin, and noise levels for annealed Langevin; DDPM uses every training timestep. Each run prints and saves a metrics JSON sidecar next to the sample output. For example, `samples/sample.png` produces `samples/sample.metrics.json`; set `--metrics-output path/to/run.json` to choose another location.

The sampling metrics include:

- **NFE:** model evaluations per generated sampling batch/trajectory. For standard DDPM and DDIM, this matches the sampler step count. Calls made by sampling-time modifiers are counted too.
- **Model forward calls:** the total observed denoiser invocations. This equals NFE for an ordinary single-batch run; with FID's streamed generation, it covers all batches.
- **Sampler steps:** the number of reverse updates selected for this run.
- **Elapsed time:** wall-clock duration of the sampling call, with CUDA synchronized before and after measurement when running on a CUDA device.
- **Throughput:** generated samples per second, plus seconds per sample and model calls per second.

These measurements add no model evaluations and do not change generated samples. For fair speed comparisons, use the same checkpoint, sample count, device, and seed, and repeat runs to account for warm-up and device load. NFE is useful for comparing algorithmic work independently from hardware speed.

FID is optional because it adds an Inception feature-extraction pass and needs a real image reference set. Install the `fid` extra above, then opt in when sampling. If sampling a Hugging Face model, install both extras with `python3 -m pip install -e ".[pretrained,fid]"`. For a pretrained CIFAR-10 model, the following uses CIFAR-10's test split and generates 10,000 images in batches of 64:

```bash
python3 sample.py --pretrained-model-id google/ddpm-cifar10-32 --sampler ddim --steps 50 --count 16 --fid --fid-dataset cifar10 --fid-data-dir data --fid-download
```

The ordinary `--count` still controls the saved preview grid. `--fid-samples` controls the number of generated and real images used for the score (default `10000`), and `--fid-batch-size` controls streaming memory use. The command saves only the first `--count` generated images but evaluates FID over the full set. For an ImageFolder reference set, pass `--fid-dataset imagefolder --fid-data-dir data/my-images`; the directory must contain class subfolders. Set `--fid-samples` no higher than the number of available reference images. FID requires RGB image models and uses feature dimension 2048. Its evaluation time is reported separately from diffusion sampling time. The JSON sidecar includes the FID score and reference configuration.

FID compares Inception feature statistics for real and generated images and is sensitive to sample count and preprocessing. Use the same reference split, number of samples, and image resolution when comparing runs. Tiny sample sets or procedural toy images do not provide a meaningful image-quality benchmark. TorchMetrics expects normalized RGB images and its default Inception feature extractor uses the optional `torch-fidelity` dependency; see the [TorchMetrics FID documentation](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).

### Use a pretrained Diffusers model

The optional `HuggingFaceUNetModel` adapter wraps unconditional Diffusers `UNet2DModel` checkpoints behind the same `DiffusionModel` interface. The default example is `google/ddpm-cifar10-32`, a 32×32 unconditional CIFAR-10 DDPM model. The repository supplies both the U-Net and its DDPM scheduler configuration; the framework loads their matching weights and beta schedule. [Model card](https://huggingface.co/google/ddpm-cifar10-32), [Diffusers U-Net API](https://huggingface.co/docs/diffusers/api/models/unet2d)

Generate samples directly without first creating a local checkpoint:

```bash
python3 sample.py --pretrained-model-id google/ddpm-cifar10-32 --sampler ddim --steps 50 --count 16 --seed 7
```

The same pretrained epsilon model can be used with annealed Langevin:

```bash
python3 sample.py --pretrained-model-id google/ddpm-cifar10-32 --sampler annealed-langevin --steps 100 --langevin-steps-per-level 2 --count 16 --seed 7
```

Fine-tune the pretrained model with the ordinary training pipeline:

```bash
python3 train.py --dataset cifar10 --download --model huggingface-unet --pretrained-model-id google/ddpm-cifar10-32 --epochs 1 --batch-size 32
```

The adapter currently supports unconditional Diffusers `UNet2DModel` checkpoints whose output is an epsilon/noise prediction. Text-conditioned Stable Diffusion pipelines and other prediction parameterizations need dedicated adapters. Pretrained model downloads require network access on the first run; the Hugging Face cache is reused later.

The default CIFAR-10 repository stores its U-Net and scheduler files at the repository root. For a Diffusers repository with components in subfolders, pass `--pretrained-unet-subfolder unet --pretrained-scheduler-subfolder scheduler` to `train.py` or `sample.py`.

## How the baseline works

For a clean training image `x_0`, the scheduler samples a time `t` and produces

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

where `epsilon` is standard Gaussian noise and `alpha_bar_t` is determined by the selected beta schedule. The model receives `(x_t, t)` and predicts `epsilon`; training minimizes mean squared error. This is the DDPM noise-prediction objective.

At generation time, the sampler starts from Gaussian noise and applies reverse updates using the trained model. DDPM makes a stochastic ancestral update at every training timestep. DDIM supports a reduced set of timesteps and optional stochasticity (`--eta`), making it useful for quick inference-step comparisons. A `SamplingPipeline` owns the loop and calls the chosen sampler's update rule at each selected time.

## Layout and extension points

```text
train.py, sample.py           Small entry points for training and generation
src/diffusion/
  models/                     Denoiser interface, image U-Net, vector MLP, pretrained and target-score adapters
  datasets/                   Data-module interface, CIFAR-10, ImageFolder, toy data
  schedulers/                 NoiseScheduler interface, linear, cosine, pretrained
  samplers/                   Sampler interface, DDPM, DDIM, ULA, and annealed Langevin
  metrics/                    Sampling NFE, timing, throughput, optional FID, and JSON output
  targets/                    Log-density/score interface, analytic toy mixture, and empirical noised-data target
  acceleration/               SamplingModifier interface and KDE birth-death update
  pipeline.py                 Sampling loop; composes sampler and modifiers
  engine.py                   Training epoch and checkpoint helpers
  cli.py                      Argument handling for the two root scripts
```

- **Model:** subclass `DiffusionModel` and add it to `models.make_model` to make an architecture available by name. Its name can then be passed as `--model`. The current choices include an image U-Net, a vector MLP, the optional pretrained Diffusers adapter, and `TargetScoreModel`, which adapts a known target score to epsilon prediction.
- **Dataset:** subclass `DiffusionDataModule` and add it to `datasets.make_data_module`. A dataset should yield a tensor or `(tensor, label)` pair and expose `sample_shape`; image tensors should be normalized to `[-1, 1]`. The same interface supports image-shaped and vector-shaped data.
- **Scheduler:** subclass `NoiseScheduler`, implement `make_betas`, and register it in `schedulers.make_scheduler`. It supplies the forward noising process and coefficients consumed by reverse updates. The pretrained adapter imports betas from the model repository's Diffusers scheduler configuration.
- **Sampler:** subclass `Sampler`, implement `inference_timesteps` and one `step` update, then register it in `samplers.make_sampler`. Override `initial_samples` for sampler-specific initialization and `sampling_metadata` to share update information with modifiers. The pipeline handles model evaluation order and iteration. Its name can then be passed as `--sampler`.
- **Sampling modifier:** subclass `SamplingModifier` and attach it to a `SamplingPipeline`. The pipeline applies modifiers after each sampler update during generation. A modifier can change the sample population size and does not participate in training.
- **Target distribution:** implement `TargetDistribution.log_prob` and `.score` for a target on the sampler's state coordinates. The analytic toy mixture and empirical noised-data mixture implement this interface. Birth-death needs the log density; `TargetScoreModel` can also turn a target score into the epsilon-prediction interface used by the samplers.

## Birth-death implementation scope

The birth-death method now runs as an optional `SamplingModifier` after fixed-target Langevin updates. In addition to the analytic toy target, `--empirical-data-target` constructs an exact Gaussian-mixture density and score from a seeded subset of a selected dataset. The subset density is a clear target for experiments but only approximates the full data distribution; small-subset KDE in image space may be poor. A pretrained DDPM model's score is not automatically consistent with this empirical target, so the empirical-target mode uses its own analytic score adapter. For the equations, commands, and limitations, see [DOCS-BD.md](DOCS-BD.md). Birth-death changes particle ancestry while preserving population size; this implementation does not remove Langevin time steps from a trajectory. Training never invokes it.
