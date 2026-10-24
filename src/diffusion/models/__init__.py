"""Diffusion denoiser architectures."""

from diffusion.models.base import DiffusionModel
from diffusion.models.huggingface import HuggingFaceUNetModel
from diffusion.models.toy_mlp import ToyMLP
from diffusion.models.toy_mixture_oracle import ToyMixtureOracleModel
from diffusion.models.target_score import TargetScoreModel
from diffusion.models.unet import SimpleUNet


def make_model(name: str, **kwargs) -> DiffusionModel:
    if name in {"simple-unet", "unet"}:
        return SimpleUNet(**kwargs)
    if name == "toy-mlp":
        return ToyMLP(**kwargs)
    if name == "toy-mixture-oracle":
        return ToyMixtureOracleModel(**kwargs)
    if name == "target-score":
        return TargetScoreModel(**kwargs)
    if name in {"huggingface-unet", "hf-unet"}:
        return HuggingFaceUNetModel(**kwargs)
    raise ValueError(f"Unknown model: {name}")


__all__ = ["DiffusionModel", "SimpleUNet", "ToyMLP", "ToyMixtureOracleModel",
           "TargetScoreModel", "HuggingFaceUNetModel", "make_model"]
