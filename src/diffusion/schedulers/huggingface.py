"""Adapter that imports the training betas from a Diffusers model repository."""

import torch

from diffusion.schedulers.base import NoiseScheduler


class HuggingFaceScheduler(NoiseScheduler):
    def __init__(self, pretrained_model_id: str, revision: str | None = None,
                 subfolder: str | None = None):
        try:
            from diffusers import DDPMScheduler
        except ImportError as exc:
            raise ImportError(
                "Pretrained Hugging Face schedules require the optional dependencies; "
                "install with `python -m pip install -e '.[pretrained]'`."
            ) from exc
        self.pretrained_model_id = pretrained_model_id
        source = DDPMScheduler.from_pretrained(pretrained_model_id, revision=revision,
                                               subfolder=subfolder)
        self._pretrained_betas = source.betas.detach().cpu().float()
        super().__init__(len(self._pretrained_betas))
        self.prediction_type = source.config.prediction_type
        self.variance_type = source.config.variance_type
        self.clip_sample = source.config.clip_sample
        self.clip_sample_range = source.config.clip_sample_range
        if self.prediction_type != "epsilon":
            raise ValueError("This framework currently supports pretrained epsilon-prediction models only")

    def make_betas(self, num_train_timesteps: int) -> torch.Tensor:
        if num_train_timesteps != len(self._pretrained_betas):
            raise ValueError("Pretrained scheduler timestep count changed unexpectedly")
        return self._pretrained_betas.clone()
