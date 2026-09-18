# High level plan

- Read up on various relevant things (diffusion models, existing acceleration methods etc) through papers or other sources
- Implement a fairly simple framework for testing diffusion models with room for making tweaks to the algorithms
- Aim to develop a potential idea / theory in order to improve the performance of diffusion sampling in some way

# Reading resources

This is the initial big obstacle, reading up on things and gaining a good understanding will take time, look to spend **~15 days** to gain enough grasp so as to reason and think about the main algorithm that we'll be tweaking

## Diffusion foundations (can skim if familiar but get a solid understanding)

- [Ho, Jain, Abbeel — Denoising Diffusion Probabilistic Models (2020)](https://arxiv.org/pdf/2006.11239)
- Song et al. — Score-Based Generative Modeling through SDEs (ICLR 2021)
- Song & Ermon — DDIM (2021)
- Karras et al. — Elucidating the Design Space of Diffusion-Based Generative Models (EDM, NeurIPS 2022) — good unified notation + strong pretrained CIFAR-10 checkpoints to build on
- **[3B1B's video on diffusion models](https://www.youtube.com/watch?v=iv-5mZ_9CPY)** - the description of this video also links some other useful resources
- **[MIT's online lectures on diffusion models](https://www.youtube.com/@MIT-6.S183)**

## Existing fast samplers

- Lu et al. — DPM-Solver / DPM-Solver++ (2022)
- Zhao et al. — UniPC (2023)
- Song et al. — Consistency Models (2023)
- Xu et al. — Restart Sampling for Improving Generative Processes (NeurIPS 2023)
- [PFDiff — training-free timestep-skipping](https://arxiv.org/pdf/2408.08822)
- ParaDiGMS — parallel/Picard-iteration sampling that trades extra compute for wall-clock speed and is compatible with existing fast samplers like DDIM and DPM-Solver — conceptually close (population/parallel idea), good contrast case

## Birth-death / WFR theory (the main theoretical crux)

- **[Lu, Lu, Nolen (2019) — Accelerating Langevin Sampling with Birth-Death](https://arxiv.org/pdf/2305.05529) - read this closely, multiple times, also the 2023 exploration-component follow-up**
- **[Efficient Bayesian Sampling with Langevin Birth-Death Dynamics (Sept 2025)](https://arxiv.org/pdf/2509.01942) — shows this line of work is still actively developing; useful as the most recent related-work anchor**
- Rotskoff, Jelassi, Bruna, Vanden-Eijnden — mean-field/neuronal birth-death & Wasserstein-Fisher-Rao gradient descent — for the geometry background

## SMC for diffusion (an alternate idea, read to position against, not to copy)

- Del Moral, Doucet, Jasra — Sequential Monte Carlo Samplers (2006), the classical reference
- Wu et al. — Twisted Diffusion Sampler (NeurIPS 2023)
- Trippe et al. — SMC for protein motif scaffolding (2023)
- Cardoso et al. — Monte Carlo Guided Denoising Diffusion (ICLR 2024)
- Self-Rewarding SMC for Masked Diffusion Language Models (2026)

# Framework implementation

This repo is for implementing the framework for testing purposes, need to map out exact structure and implementation plan (should not be too hard, **~10 days should suffice**), can be done in parallel with reading and (if necessary) with some help from AI

# Potential ideas

For now, pray to God