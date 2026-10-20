"""Forward noising schedules."""

from diffusion.schedulers.base import NoiseScheduler
from diffusion.schedulers.cosine import CosineScheduler
from diffusion.schedulers.huggingface import HuggingFaceScheduler
from diffusion.schedulers.linear import LinearScheduler


def make_scheduler(name: str, num_train_timesteps: int = 1000,
                   beta_start: float = 1e-4, beta_end: float = 2e-2,
                   pretrained_model_id: str | None = None,
                   revision: str | None = None,
                   pretrained_subfolder: str | None = None) -> NoiseScheduler:
    if name == "linear":
        return LinearScheduler(num_train_timesteps, beta_start, beta_end)
    if name == "cosine":
        return CosineScheduler(num_train_timesteps)
    if name in {"huggingface", "pretrained"}:
        if not pretrained_model_id:
            raise ValueError("pretrained_model_id is required for the Hugging Face scheduler")
        return HuggingFaceScheduler(pretrained_model_id, revision, pretrained_subfolder)
    raise ValueError(f"Unknown scheduler: {name}")


__all__ = ["NoiseScheduler", "LinearScheduler", "CosineScheduler", "HuggingFaceScheduler",
           "make_scheduler"]
