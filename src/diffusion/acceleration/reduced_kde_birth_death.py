"""Low-dimensional feature-space birth-death particle updates.

Computes KDE and target density in a compressed semantic feature space
(e.g., dimension 25 instead of pixel dimension 3072 or 196608) using a
pretrained CNN backbone (VGG-16, ResNet-18, or ConvNet) and PCA / orthogonal
projection.

In high-dimensional pixel space (d >= 32, d = 3072, d = 196608), standard Gaussian
KDE suffers from curse-of-dimensionality self-domination: the self-kernel mass
is ~1.0 for every particle, meaning the KDE conveys zero density information and
BD degenerates into mode collapse. Compressing images into a semantic 25-dimensional
feature space restores meaningful neighborhood overlap, rank correlation with the
data distribution, and healthy particle diversity.
"""

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diffusion.acceleration.base import SamplingModifier, SamplingState
from diffusion.sampling import SamplingContext
from diffusion.targets.base import TargetDistribution


class LightweightConvTrunk(nn.Module):
    """Self-contained convolutional feature trunk (always available offline)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureReducer(nn.Module):
    """Extracts CNN features from images and compresses them to a low dimension (e.g. 25)."""

    def __init__(
        self,
        reduced_dim: int = 25,
        backbone: str = "auto",
        projection_method: str = "pca",
        seed: int = 42,
    ):
        super().__init__()
        if reduced_dim < 1:
            raise ValueError(f"kde_reduce dimension must be >= 1, got {reduced_dim}")
        self.reduced_dim = int(reduced_dim)
        self.backbone_name = backbone
        self.projection_method = projection_method
        self.seed = seed

        self.trunk, self.raw_dim = self._build_trunk(backbone)
        self.trunk.eval()
        for param in self.trunk.parameters():
            param.requires_grad = False

        # ImageNet normalization statistics
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # Initialize fixed orthogonal projection as fallback
        self._init_orthogonal(self.raw_dim, self.reduced_dim, seed=seed)
        self.is_pca_fitted = False

        # Neural Network projection heads (Linear or MLP)
        if self.projection_method == "linear":
            torch.manual_seed(seed)
            self.nn_head = nn.Linear(self.raw_dim, self.reduced_dim)
            for p in self.nn_head.parameters():
                p.requires_grad = False
        elif self.projection_method == "mlp":
            torch.manual_seed(seed)
            hidden_dim = max(self.reduced_dim * 2, 128)
            self.nn_head = nn.Sequential(
                nn.Linear(self.raw_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.reduced_dim),
            )
            for p in self.nn_head.parameters():
                p.requires_grad = False
        else:
            self.nn_head = None

    def _build_trunk(self, backbone: str) -> tuple[nn.Module, int]:
        """Constructs the feature extractor backbone with fail-safe fallbacks."""
        if backbone in ("vgg16", "auto"):
            try:
                import torchvision.models as tv_models
                cache_ckpt = (
                    Path.home()
                    / ".cache"
                    / "torch"
                    / "hub"
                    / "checkpoints"
                    / "vgg16-397923af.pth"
                )
                if cache_ckpt.exists():
                    vgg = tv_models.vgg16(weights=None)
                    state = torch.load(str(cache_ckpt), map_location="cpu")
                    vgg.load_state_dict(state)
                else:
                    vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
                trunk = nn.Sequential(
                    vgg.features,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                )
                return trunk, 512
            except Exception:
                if backbone == "vgg16":
                    pass  # Fall through to lightweight ConvNet if torchvision fails
        if backbone == "resnet18":
            try:
                import torchvision.models as tv_models
                resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                trunk = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
                return trunk, 512
            except Exception:
                pass

        if backbone in ("mobilenet", "mobilenet_v3"):
            try:
                import torchvision.models as tv_models
                mb = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
                trunk = nn.Sequential(mb.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
                return trunk, 576
            except Exception:
                pass

        # Standalone lightweight ConvNet (zero dependencies, completely offline)
        return LightweightConvTrunk(in_channels=3, out_dim=512), 512

    def _init_orthogonal(self, in_dim: int, out_dim: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        k = min(in_dim, out_dim)
        random_mat = torch.randn(in_dim, k, generator=gen)
        q, _ = torch.linalg.qr(random_mat)
        if out_dim > in_dim:
            # Pad with zeros if requested dimension is larger than input
            padded = torch.zeros(in_dim, out_dim)
            padded[:, :k] = q
            q = padded
        self.register_buffer("ortho_proj", q)

    def fit_pca(self, reference_samples: torch.Tensor):
        """Fit PCA projection matrix on real reference samples once."""
        self.to(device=reference_samples.device)
        with torch.no_grad():
            ref_raw = self._extract_raw(reference_samples)  # (M, raw_dim)
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

    def _extract_raw(self, x: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
        """Extract high-dimensional raw CNN features, chunked to prevent OOM."""
        if x.ndim == 2:
            # Already flat vector data
            return x

        # Ensure trunk is on the same device and dtype as input
        self.trunk.to(device=x.device, dtype=x.dtype)

        # Ensure 3-channel image
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]

        # Rescale [-1, 1] to [0, 1] then ImageNet normalize
        x_01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
        x_norm = (x_01 - mean) / std

        # Chunk large batches (e.g. references) to prevent activation OOM
        if x_norm.shape[0] > chunk_size:
            chunks = []
            for start in range(0, x_norm.shape[0], chunk_size):
                chunks.append(self.trunk(x_norm[start : start + chunk_size]))
            return torch.cat(chunks, dim=0)

        return self.trunk(x_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input images or vectors to the reduced_dim feature space."""
        if x.ndim == 2 and x.shape[1] <= self.reduced_dim:
            # Low-dimensional vector data (e.g. 2D toy mixture)
            return x

        self.to(device=x.device)
        raw = self._extract_raw(x)
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

        # Fallback to orthogonal projection
        ortho = self.ortho_proj.to(device=raw.device, dtype=raw.dtype)
        if raw.shape[1] != ortho.shape[0]:
            # Vector input dimension mismatch: adaptively project
            k = min(raw.shape[1], self.reduced_dim)
            return raw[:, :k]
        return raw @ ortho


class ReducedKDEBirthDeathAcceleration(SamplingModifier):
    """Fixed-population birth-death particle updates using reduced-dimensional KDE.

    Compresses particle population and reference target into a low-dimensional
    semantic space (default: 25 dims) via a pretrained CNN backbone and PCA.
    This resolves the curse of dimensionality and self-domination that plagues
    pixel-space KDE on image datasets.
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
        seed: int = 42,
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

        self.feature_reducer = FeatureReducer(
            reduced_dim=self.reduced_dim,
            backbone=backbone,
            projection_method=projection_method,
            seed=seed,
        )

        # Pre-fit PCA if reference samples exist in the target
        if hasattr(target, "reference_samples") and target.reference_samples is not None:
            refs = target.reference_samples
            if refs.ndim >= 3:
                self.feature_reducer.to(device=refs.device)
                self.feature_reducer.fit_pca(refs)

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

        eligible_updates = context.num_steps - int(context.has_clean_endpoint)
        intervention_count = eligible_updates // self.every_k_steps
        if intervention_count < 1:
            return state
        delta_tau = self.total_time / intervention_count

        population = samples.shape[0]
        if population < 2:
            raise ValueError("birth-death updates require at least two particles")
        timestep = context.state_timestep
        target_samples = self._target_coordinates(samples, context, scheduler)

        # 1. Project particle samples into reduced semantic feature space (e.g. 25D)
        self.feature_reducer.to(device=samples.device, dtype=samples.dtype)
        particle_features = self.feature_reducer(target_samples)  # (N, reduced_dim)

        # 2. Compute particle population KDE in reduced space
        pop_dist_sq = torch.cdist(particle_features, particle_features).square()
        log_kernel_pop = -0.5 * pop_dist_sq / (self.kernel_bandwidth**2)
        log_kde = torch.logsumexp(log_kernel_pop, dim=1) - math.log(population)

        # 3. Compute target density in reduced space
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
                ref_noise = torch.randn_like(refs, generator=generator)
                noisy_refs = refs + sigma * ref_noise
                ref_features = self.feature_reducer(noisy_refs)
            else:
                if (
                    self.cached_ref_features is None
                    or self.cached_ref_features.device != samples.device
                ):
                    self.cached_ref_features = self.feature_reducer(refs)
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
                # Over-represented particle in feature space is killed; partner duplicated
                source_indices[index] = source_indices[partner]
                deaths += 1
            else:
                # Under-represented particle in feature space is duplicated; partner killed
                source_indices[partner] = source_indices[index]
                births += 1

        updated = samples.index_select(
            0, torch.tensor(source_indices, device=samples.device, dtype=torch.long)
        )
        metadata = dict(state.metadata)
        metadata.update(
            {
                "birth_death_events": births + deaths,
                "birth_death_births": births,
                "birth_death_deaths": deaths,
                "birth_death_population": population,
                "reduced_kde_dim": self.reduced_dim,
                "reduced_kde_bandwidth": self.kernel_bandwidth,
            }
        )
        return SamplingState(
            updated,
            state.timestep,
            state.previous_timestep,
            metadata,
            context=state.context,
        )
