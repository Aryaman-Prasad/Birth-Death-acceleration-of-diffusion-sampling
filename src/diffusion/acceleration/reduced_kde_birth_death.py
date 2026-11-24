"""Low-dimensional DDPM U-Net feature-space birth-death particle updates.

Computes KDE and target density in a compressed semantic feature space
(e.g., dimension 128 or 25 instead of pixel dimension 3072 or 196608) using the
native DDPM U-Net encoder/bottleneck layers (< 1024 dimensions) and PCA / orthogonal
projection.

In high-dimensional pixel space (d >= 32, d = 3072, d = 196608), standard Gaussian
KDE suffers from curse-of-dimensionality self-domination: the self-kernel mass
is ~1.0 for every particle, meaning the KDE conveys zero density information and
BD degenerates into mode collapse. Extracting features from the DDPM U-Net's own
encoder/bottleneck layers (< 1024 dims) provides noise-aware, semantic representations
that restore meaningful neighborhood overlap and healthy particle diversity.
"""

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from diffusion.acceleration.base import (
    SamplingModifier,
    SamplingState,
    coordinate_noise_std,
)
from diffusion.models.time_embedding import SinusoidalTimeEmbedding
from diffusion.models.unet import ResidualBlock
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


class DDPMUNetEncoderTrunk(nn.Module):
    """Native DDPM U-Net encoder and bottleneck feature trunk (dimension < 1024).

    Extracts time-conditioned latent representations directly from the DDPM
    U-Net architecture, completely replacing external ImageNet-trained classifiers
    (ResNet, VGG, MobileNet).

    Supported layer outputs:
    - 'bottleneck' (default): Output of middle blocks, pooled to 128D (< 1024).
    - 'down_block2' / 'deep_encoder': Output of deepest downsampling stage (128D).
    - 'down_block1': Output of first encoder stage (64D).
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 64,
        time_channels: int = 256,
        layer: str = "bottleneck",
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        self.layer = layer.lower()
        self.base_channels = base_channels

        # DDPM time conditioning MLP
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        # DDPM U-Net Encoder Layers
        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down_block1 = ResidualBlock(base_channels, base_channels, time_channels)  # 64 ch
        self.downsample = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.down_block2 = ResidualBlock(base_channels, base_channels * 2, time_channels)  # 128 ch
        self.middle1 = ResidualBlock(base_channels * 2, base_channels * 2, time_channels)   # 128 ch
        self.middle2 = ResidualBlock(base_channels * 2, base_channels * 2, time_channels)   # 128 ch

        # Global average pool: reduces (B, C, H, W) -> (B, C) where C < 1024
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Determine output feature dimension
        if self.layer in ("down_block1", "first_encoder"):
            self.out_dim = base_channels  # 64 dims
        else:
            self.out_dim = base_channels * 2  # 128 dims for bottleneck and down_block2

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str):
        """Loads weights from a trained SimpleUNet or framework checkpoint."""
        ckpt = Path(path)
        if not ckpt.exists():
            return
        state = torch.load(str(ckpt), map_location="cpu")
        weights = state.get("model", state)
        filtered = {}
        for k, v in weights.items():
            if k.startswith("input."):
                filtered[k.replace("input.", "input_conv.")] = v
            elif any(k.startswith(prefix) for prefix in (
                "time_mlp.", "input_conv.", "down_block1.", "downsample.",
                "down_block2.", "middle1.", "middle2."
            )):
                filtered[k] = v
        if filtered:
            self.load_state_dict(filtered, strict=False)

    def load_from_unet(self, unet_model: nn.Module):
        """Copies encoder weights directly from an active SimpleUNet instance."""
        target_dict = self.state_dict()
        source_dict = unet_model.state_dict()
        transferred = {}
        for k, v in source_dict.items():
            mapped_k = k.replace("input.", "input_conv.") if k.startswith("input.") else k
            if mapped_k in target_dict and target_dict[mapped_k].shape == v.shape:
                transferred[mapped_k] = v
        if transferred:
            self.load_state_dict(transferred, strict=False)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        # Build timestep tensor
        if timesteps is None:
            ts = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        elif isinstance(timesteps, int):
            ts = torch.full((x.shape[0],), timesteps, dtype=torch.long, device=x.device)
        elif timesteps.ndim == 0:
            ts = timesteps.repeat(x.shape[0]).to(dtype=torch.long, device=x.device)
        else:
            ts = timesteps.to(dtype=torch.long, device=x.device)
            if ts.shape[0] != x.shape[0]:
                ts = ts[0].repeat(x.shape[0])

        t = self.time_mlp(ts)
        h = self.down_block1(self.input_conv(x), t)
        if self.layer in ("down_block1", "first_encoder"):
            return self.pool(h).flatten(1)

        h = self.down_block2(self.downsample(h), t)
        if self.layer in ("down_block2", "deep_encoder"):
            return self.pool(h).flatten(1)

        # Bottleneck (middle block)
        h = self.middle2(self.middle1(h, t), t)
        return self.pool(h).flatten(1)  # shape: (B, 128) - strictly < 1024 dims


class FeatureReducer(nn.Module):
    """Extracts DDPM U-Net encoder features (< 1024 dims) and compresses them (e.g. to 25 dims)."""

    def __init__(
        self,
        reduced_dim: int = 25,
        backbone: str = "auto",
        projection_method: str = "pca",
        checkpoint_path: str | None = None,
        seed: int = 42,
    ):
        super().__init__()
        if reduced_dim < 1:
            raise ValueError(f"kde_reduce dimension must be >= 1, got {reduced_dim}")
        self.reduced_dim = int(reduced_dim)
        self.backbone_name = backbone
        self.projection_method = projection_method
        self.seed = seed

        # Check if backbone is direct pixel-PCA (zero convolutions, projects raw pixels directly)
        self.is_pixel_pca = str(backbone).lower() in (
            "pca", "pixel_pca", "pixel-pca", "pixel", "none", "linear_pca", "linear-pca"
        )

        if self.is_pixel_pca:
            self.trunk = None
            self.raw_dim = None
        else:
            # Map layer choice
            layer = "bottleneck"
            if backbone in ("down_block1", "down_block2", "bottleneck"):
                layer = backbone
            elif backbone in ("convnet", "resnet18", "vgg16", "mobilenet", "mobilenet_v3", "auto"):
                # Cleanly replace external CNN classification backbones with U-Net bottleneck
                layer = "bottleneck"

            # Native DDPM U-Net encoder trunk (< 1024 dimensions)
            self.trunk = DDPMUNetEncoderTrunk(
                image_channels=3,
                base_channels=64,
                time_channels=256,
                layer=layer,
                checkpoint_path=checkpoint_path,
            )
            self.raw_dim = self.trunk.out_dim  # 128 dimensions (< 1024)
            self.trunk.eval()
            for param in self.trunk.parameters():
                param.requires_grad = False

            # Initialize fixed orthogonal projection as fallback
            self._init_orthogonal(self.raw_dim, self.reduced_dim, seed=seed)

        self.is_pca_fitted = False

        # Neural Network projection heads (Linear or MLP)
        if self.projection_method == "linear" and not self.is_pixel_pca:
            torch.manual_seed(seed)
            self.nn_head = nn.Linear(self.raw_dim, self.reduced_dim)
            for p in self.nn_head.parameters():
                p.requires_grad = False
        elif self.projection_method == "mlp" and not self.is_pixel_pca:
            torch.manual_seed(seed)
            hidden_dim = max(self.reduced_dim * 2, 64)
            self.nn_head = nn.Sequential(
                nn.Linear(self.raw_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.reduced_dim),
            )
            for p in self.nn_head.parameters():
                p.requires_grad = False
        else:
            self.nn_head = None

    def _init_orthogonal(self, in_dim: int, out_dim: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        k = min(in_dim, out_dim)
        random_mat = torch.randn(in_dim, k, generator=gen)
        q, _ = torch.linalg.qr(random_mat)
        if out_dim > in_dim:
            padded = torch.zeros(in_dim, out_dim)
            padded[:, :k] = q
            q = padded
        self.register_buffer("ortho_proj", q)

    def fit_pca(self, reference_samples: torch.Tensor, timesteps: Any = 0):
        """Fit PCA projection matrix on real reference samples once."""
        self.to(device=reference_samples.device)
        with torch.no_grad():
            ref_raw = self._extract_raw(reference_samples, timesteps=timesteps)  # (M, raw_dim)
            if self.raw_dim is None:
                self.raw_dim = ref_raw.shape[1]
                self._init_orthogonal(self.raw_dim, self.reduced_dim, seed=self.seed)

            mean = ref_raw.mean(dim=0, keepdim=True)
            centered = ref_raw - mean
            # SVD on centered features
            _, s, vh = torch.linalg.svd(centered, full_matrices=False)
            k = min(self.reduced_dim, vh.shape[0], vh.shape[1])
            proj = vh[:k, :].T  # (raw_dim, k)
            stds = (s[:k] / math.sqrt(max(1, ref_raw.shape[0] - 1))).clamp_min(1e-5)
            self.register_buffer("pca_mean", mean)
            self.register_buffer("pca_proj", proj)
            self.register_buffer("pca_stds", stds)
            self.is_pca_fitted = True

    def _extract_raw(
        self, x: torch.Tensor, timesteps: Any = None, chunk_size: int = 64
    ) -> torch.Tensor:
        """Extract raw features (flattened pixels for PCA, or DDPM U-Net encoder/bottleneck features)."""
        if x.ndim == 2:
            return x

        if self.is_pixel_pca:
            # Flatten spatial image directly (N, C*H*W) with zero convolutions
            return x.flatten(start_dim=1)

        self.trunk.to(device=x.device, dtype=x.dtype)

        # Ensure 3-channel image
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]

        # Standard diffusion image normalization [-1, 1]
        x_norm = x.clamp(-1.0, 1.0)

        # Chunk large batches to bound activation memory
        if x_norm.shape[0] > chunk_size:
            chunks = []
            for start in range(0, x_norm.shape[0], chunk_size):
                sub_x = x_norm[start : start + chunk_size]
                chunks.append(self.trunk(sub_x, timesteps=timesteps))
            return torch.cat(chunks, dim=0)

        return self.trunk(x_norm, timesteps=timesteps)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        """Projects input images or vectors to the reduced_dim feature space (< 1024)."""
        if x.ndim == 2 and x.shape[1] <= self.reduced_dim:
            # Low-dimensional vector data (e.g. 2D toy mixture)
            return x

        self.to(device=x.device)
        raw = self._extract_raw(x, timesteps=timesteps)  # (N, raw_dim) where raw_dim = 128 < 1024

        if self.projection_method in ("linear", "mlp") and self.nn_head is not None:
            self.nn_head.to(device=raw.device, dtype=raw.dtype)
            return self.nn_head(raw)

        if self.projection_method == "pca" and self.is_pca_fitted:
            mean = self.pca_mean.to(device=raw.device, dtype=raw.dtype)
            proj = self.pca_proj.to(device=raw.device, dtype=raw.dtype)
            stds = self.pca_stds.to(device=raw.device, dtype=raw.dtype)
            centered = raw - mean
            reduced = (centered @ proj) / stds
            return reduced

        # Fallback to orthogonal projection if available
        if hasattr(self, "ortho_proj"):
            ortho = self.ortho_proj.to(device=raw.device, dtype=raw.dtype)
            if raw.shape[1] == ortho.shape[0]:
                return raw @ ortho
        k = min(raw.shape[1], self.reduced_dim)
        return raw[:, :k]


class ReducedKDEBirthDeathAcceleration(SamplingModifier):
    """Fixed-population birth-death particle updates using DDPM U-Net latent features.

    Compresses particle population and reference target into a compact semantic
    space (< 1024 dims, default: 25 dims) via DDPM U-Net encoder/bottleneck features
    and PCA projection. This eliminates the curse of dimensionality and distance
    concentration without relying on external classification backbones.
    """

    def __init__(
        self,
        target: TargetDistribution,
        kernel_bandwidth: float = 0.3,
        *,
        kde_reduce: int = 25,
        every_k_steps: int = 1,
        total_time: float = 1.0,
        backbone: str = "auto",
        projection_method: str = "pca",
        noise_references: bool = True,
        checkpoint_path: str | None = None,
        seed: int = 42,
        min_timestep: int | None = 100,
        max_timestep: int | None = None,
        max_delta_tau: float = 0.005,
        max_event_probability: float = 0.25,
        add_noise: float = 0.0,
    ):
        if kernel_bandwidth <= 0:
            raise ValueError("kernel_bandwidth must be positive")
        if kde_reduce < 1:
            raise ValueError("kde_reduce must be at least 1")
        if every_k_steps < 1:
            raise ValueError("every_k_steps must be positive")
        if total_time <= 0:
            raise ValueError("total_time must be positive")

        self.target = target
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.reduced_dim = int(kde_reduce)
        self.every_k_steps = int(every_k_steps)
        self.total_time = float(total_time)
        self.noise_references = bool(noise_references)
        self.cached_ref_features = None
        self.min_timestep = int(min_timestep) if min_timestep is not None else None
        self.max_timestep = int(max_timestep) if max_timestep is not None else None
        self.max_delta_tau = float(max_delta_tau)
        self.max_event_probability = (
            float(max_event_probability) if max_event_probability is not None else None
        )
        self.add_noise = float(add_noise)

        self.feature_reducer = FeatureReducer(
            reduced_dim=self.reduced_dim,
            backbone=backbone,
            projection_method=projection_method,
            checkpoint_path=checkpoint_path,
            seed=seed,
        )

        # Pre-fit PCA if reference samples exist in the target
        if hasattr(target, "reference_samples") and target.reference_samples is not None:
            refs = target.reference_samples
            if refs.ndim >= 3:
                self.feature_reducer.to(device=refs.device)
                self.feature_reducer.fit_pca(refs, timesteps=0)

    def _target_coordinates(
        self, samples: torch.Tensor, context: SamplingContext, scheduler
    ) -> torch.Tensor:
        target_coordinates = getattr(self.target, "state_coordinates", None)
        if target_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError(
                "birth-death target.state_coordinates must be 'x_t' or 'normalized_x_t'"
            )
        if context.state_coordinates == target_coordinates:
            return samples
        if context.state_coordinates not in {"x_t", "normalized_x_t"}:
            raise ValueError(
                f"cannot evaluate birth-death at state coordinates {context.state_coordinates!r}"
            )
        timestep = context.state_timestep
        if timestep is None:
            raise ValueError("noisy-coordinate conversion requires a state timestep")
        alpha_bar = scheduler.alpha_bars[timestep].to(samples)
        scale = alpha_bar.sqrt()
        if context.state_coordinates == "x_t":
            return samples / scale
        return samples * scale

    def apply(
        self, state: SamplingState, *, model, scheduler, generator
    ) -> SamplingState:
        samples = state.samples
        context = state.context
        if context is None:
            raise ValueError(
                "birth-death requires the sampler's post-update SamplingContext"
            )
        if (
            context.is_clean_sample
            or context.state_timestep is None
            or context.state_coordinates == "clean_x0"
        ):
            return state

        if not 0 <= context.step_index < context.num_steps:
            raise ValueError("SamplingContext has an invalid trajectory position")
        if (context.step_index + 1) % self.every_k_steps != 0:
            return state

        timestep = context.state_timestep
        if timestep is not None:
            if self.min_timestep is not None and timestep < self.min_timestep:
                return state
            if self.max_timestep is not None and timestep > self.max_timestep:
                return state

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = eligible_updates // self.every_k_steps
        if intervention_count < 1:
            return state
        raw_delta_tau = self.total_time / intervention_count
        delta_tau = min(raw_delta_tau, self.max_delta_tau)

        population = samples.shape[0]
        if population < 2:
            raise ValueError("birth-death updates require at least two particles")
        target_samples = self._target_coordinates(samples, context, scheduler)

        # Sync weights if model is a SimpleUNet
        if hasattr(model, "down_block1") and hasattr(model, "middle2"):
            self.feature_reducer.trunk.load_from_unet(model)

        # 1. Project particles using DDPM U-Net encoder/bottleneck features (< 1024 dims)
        self.feature_reducer.to(device=samples.device, dtype=samples.dtype)
        particle_features = self.feature_reducer(
            target_samples, timesteps=timestep
        )  # (N, reduced_dim)

        # 2. Compute particle population KDE in reduced U-Net latent space
        pop_dist_sq = torch.cdist(particle_features, particle_features).square()
        log_kernel_pop = -0.5 * pop_dist_sq / (self.kernel_bandwidth**2)
        log_kde = torch.logsumexp(log_kernel_pop, dim=1) - math.log(population)

        # 3. Compute target density in reduced U-Net latent space
        if (
            hasattr(self.target, "reference_samples")
            and self.target.reference_samples is not None
        ):
            refs = self.target.reference_samples.to(
                device=samples.device, dtype=samples.dtype
            )
            if self.noise_references and timestep is not None:
                sigma = self.target.noise_std(
                    int(timestep), scheduler, device=samples.device, dtype=samples.dtype
                )
                ref_noise = torch.randn(
                    refs.shape,
                    device=refs.device,
                    dtype=refs.dtype,
                    generator=generator,
                )
                noisy_refs = refs + sigma * ref_noise
                ref_features = self.feature_reducer(noisy_refs, timesteps=timestep)
            else:
                if (
                    self.cached_ref_features is None
                    or self.cached_ref_features.device != samples.device
                ):
                    self.cached_ref_features = self.feature_reducer(refs, timesteps=0)
                ref_features = self.cached_ref_features

            ref_dist_sq = torch.cdist(particle_features, ref_features).square()
            log_kernel_ref = -0.5 * ref_dist_sq / (self.kernel_bandwidth**2)
            log_target = torch.logsumexp(log_kernel_ref, dim=1) - math.log(refs.shape[0])
        else:
            # Analytic target (e.g. 2D toy mixture)
            log_target = self.target.log_prob(target_samples, int(timestep), scheduler)

        if log_target.shape != (population,):
            raise ValueError("target density must produce one value per particle")

        # 4. Birth-death rate and particle updates
        beta = log_kde - log_target
        beta = beta - beta.mean()

        event_probability = -torch.expm1(-beta.abs() * delta_tau)
        if self.max_event_probability is not None and self.max_event_probability > 0:
            event_probability = event_probability.clamp(max=self.max_event_probability)
        uniforms = torch.rand(population, device=samples.device, generator=generator)
        selected = uniforms < event_probability
        partners = torch.randint(
            population - 1, (population,), device=samples.device, generator=generator
        )

        event_data = torch.stack(
            (
                selected.to(torch.float32),
                beta.gt(0).to(torch.float32),
                partners.to(torch.float32),
            ),
            dim=1,
        ).cpu().tolist()

        source_indices = list(range(population))
        cloned_positions = []
        births = 0
        deaths = 0
        for index, (is_selected, is_overrepresented, raw_partner) in enumerate(
            event_data
        ):
            if not is_selected:
                continue
            partner = int(raw_partner)
            if partner >= index:
                partner += 1
            if is_overrepresented:
                # Over-represented particle in latent space is killed; partner duplicated
                source_indices[index] = source_indices[partner]
                cloned_positions.append(index)
                deaths += 1
            else:
                # Under-represented particle in latent space is duplicated; partner killed
                source_indices[partner] = source_indices[index]
                cloned_positions.append(partner)
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )
        if self.add_noise > 0 and cloned_positions:
            unique_cloned = sorted(list(set(cloned_positions)))
            noise = torch.randn(
                (len(unique_cloned), *samples.shape[1:]),
                device=samples.device,
                dtype=samples.dtype,
                generator=generator,
            )
            scale = self.add_noise * coordinate_noise_std(
                samples, context.state_timestep, scheduler, context.state_coordinates
            )
            cloned_tensor_idx = torch.tensor(unique_cloned, device=samples.device, dtype=torch.long)
            updated = updated.clone()
            updated[cloned_tensor_idx] = updated[cloned_tensor_idx] + scale * noise
        metadata = dict(state.metadata)
        metadata.update(
            {
                "birth_death_events": births + deaths,
                "birth_death_births": births,
                "birth_death_deaths": deaths,
                "birth_death_population": population,
                "reduced_kde_dim": self.reduced_dim,
                "reduced_kde_bandwidth": self.kernel_bandwidth,
                "reduced_kde_feature_type": "ddpm_unet_encoder",
            }
        )
        return SamplingState(
            updated,
            state.timestep,
            state.previous_timestep,
            metadata,
            context=state.context,
            source_samples=state.source_samples,
        )
