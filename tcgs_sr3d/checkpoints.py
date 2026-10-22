

import torch

from .model import TCGSSR3D

MODEL_KEYS = ("scale", "hidden_channels", "feature_channels", "num_blocks",
              "hr_blocks", "levels", "mid_radius", "coarse_radius")

def load_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Expected a checkpoint containing a model state dictionary")
    source_config = payload.get("model_config", payload.get("args"))
    if not isinstance(source_config, dict):
        raise ValueError("Missing model_config/args in checkpoint")
    config = {key: source_config[key] for key in MODEL_KEYS}
    config["gradient_checkpointing"] = False
    model = TCGSSR3D(**config)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    metadata = {"epoch": payload.get("epoch"), "model_config": config,
                "parameter_count": model.parameter_count(),
                "normalization": payload.get("normalization", "input_zscore" if "args" in payload else "none")}
    return model, metadata

def normalize_pair(low, high=None):
    
    if low.ndim != 5 or low.shape[1] != 1 or min(low.shape[-3:]) < 2:
        raise ValueError("Expected [B,1,inline,time,crossline] with dimensions >=2")
    mean = low.mean(dim=(1,2,3,4), keepdim=True)
    std = low.flatten(1).std(dim=1, correction=1).view(-1,1,1,1,1).clamp_min(1e-6)
    return (low-mean)/std, None if high is None else (high-mean)/std, mean, std

@torch.inference_mode()
def predict(model, low, normalization):
    if normalization == "input_zscore":
        normalized, _, mean, std = normalize_pair(low)
        return model(normalized).float()*std + mean
    if normalization == "none":
        return model(low).float()
    raise ValueError(f"Unknown normalization: {normalization}")
