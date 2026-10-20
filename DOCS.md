# Birth-death acceleration of diffusion sampling

IE 643 course project by Aryaman and Sabarinath.

The framework setup, toy-data dry runs, pretrained model options, and extension points are documented in [DOCS.md](DOCS.md).

```bash
python3 -m pip install -e .
python3 train.py --dataset toy-images --toy-size 64 --image-size 16 --epochs 1 --batch-size 8 --timesteps 20 --scheduler cosine --base-channels 16
```

# Framework guide

This repository currently implements the first course-project deliverable: a standard diffusion training and sampling baseline. The code is split into components so that models, data sources, noise schedules, reverse samplers, and optional sampling-time methods can be extended independently.

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

The sample grid is saved under `samples/`. `--steps` selects inference steps for DDIM; DDPM uses every training timestep. Sampling reports wall-clock time. For comparisons, keep the checkpoint, sample count, device, and seed fixed and repeat timings. The project currently reports training noise-prediction MSE and sampling time; FID and other image-quality metrics are not implemented yet.

### Use a pretrained Diffusers model

The optional `HuggingFaceUNetModel` adapter wraps unconditional Diffusers `UNet2DModel` checkpoints behind the same `DiffusionModel` interface. The default example is `google/ddpm-cifar10-32`, a 32×32 unconditional CIFAR-10 DDPM model. The repository supplies both the U-Net and its DDPM scheduler configuration; the framework loads their matching weights and beta schedule. [Model card](https://huggingface.co/google/ddpm-cifar10-32), [Diffusers U-Net API](https://huggingface.co/docs/diffusers/api/models/unet2d)

Generate samples directly without first creating a local checkpoint:

```bash
python3 sample.py --pretrained-model-id google/ddpm-cifar10-32 --sampler ddim --steps 50 --count 16 --seed 7
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
  models/                     Denoiser interface, image U-Net, vector MLP, pretrained adapter
  datasets/                   Data-module interface, CIFAR-10, ImageFolder, toy data
  schedulers/                 NoiseScheduler interface, linear, cosine, pretrained
  samplers/                   Sampler interface, DDPM and DDIM reverse updates
  acceleration/               SamplingModifier interface and birth-death placeholder
  pipeline.py                 Sampling loop; composes sampler and modifiers
  engine.py                   Training epoch and checkpoint helpers
  cli.py                      Argument handling for the two root scripts
```

- **Model:** subclass `DiffusionModel` and add it to `models.make_model` to make an architecture available by name. Its name can then be passed as `--model`. The current choices include an image U-Net, a vector MLP, and the optional pretrained Diffusers adapter.
- **Dataset:** subclass `DiffusionDataModule` and add it to `datasets.make_data_module`. A dataset should yield a tensor or `(tensor, label)` pair and expose `sample_shape`; image tensors should be normalized to `[-1, 1]`. The same interface supports image-shaped and vector-shaped data.
- **Scheduler:** subclass `NoiseScheduler`, implement `make_betas`, and register it in `schedulers.make_scheduler`. It supplies the forward noising process and coefficients consumed by reverse updates. The pretrained adapter imports betas from the model repository's Diffusers scheduler configuration.
- **Sampler:** subclass `Sampler`, implement `inference_timesteps` and one `step` update, then register it in `samplers.make_sampler`. The pipeline handles model evaluation order and iteration. Its name can then be passed as `--sampler`.
- **Sampling modifier:** subclass `SamplingModifier` and attach it to a `SamplingPipeline`. The pipeline applies modifiers after each sampler update during generation. A modifier can change the sample population size and does not participate in training.

## Birth-death direction

The cited [Lu, Lu, and Nolen paper](https://arxiv.org/abs/1905.09863) augments Langevin dynamics with a nonlocal birth-death term for a population of samples. That supports keeping birth-death logic separate from the learned denoiser and training objective, and treating it as a sampling-time population modifier. The paper itself does not specify how to skip DDPM denoising steps. Applying its idea to a diffusion trajectory therefore requires a project-specific rule for the time-dependent noisy distributions and particle scores; that research and implementation are future deliverables.

`src/diffusion/acceleration/birth_death.py` contains only the placeholder class. The currently usable pipeline has no modifier enabled by default, so baseline DDPM and DDIM behavior is unaffected. A future modifier can be passed to `SamplingPipeline(..., modifiers=[...])` without changing either sampler's reverse-update implementation. Training code never invokes sampling modifiers.
