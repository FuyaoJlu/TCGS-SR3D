from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tcgs_sr3d.checkpoints import load_checkpoint, predict
from tcgs_sr3d.losses import metrics3d


def score(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = metrics3d(prediction, target)
    metrics["mae"] = float((prediction - target).abs().mean())
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the packaged demo on one precomputed low/high-resolution patch.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tcgs_sr3d_epoch145.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/demo_patch.npz"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    with np.load(args.data, allow_pickle=False) as payload:
        low = torch.from_numpy(payload["lr"].copy()).float().unsqueeze(0)
        high = torch.from_numpy(payload["hr"].copy()).float().unsqueeze(0)
    if low.ndim != 5 or high.ndim != 5 or low.shape[1] != 1 or high.shape[1] != 1:
        raise ValueError("Expected lr and hr arrays shaped [1, 1, D, H, W]")
    if tuple(high.shape[-3:]) != tuple(2 * value for value in low.shape[-3:]):
        raise ValueError("The demo expects a 2x spatial upsampling factor")

    model, metadata = load_checkpoint(args.checkpoint, args.device)
    low, high = low.to(args.device), high.to(args.device)
    with torch.inference_mode():
        prediction = predict(model, low, metadata["normalization"])
    baseline = F.interpolate(low, size=high.shape[-3:], mode="trilinear", align_corners=False)
    result = {
        "data": str(args.data),
        "input_shape": list(low.shape),
        "output_shape": list(prediction.shape),
        "checkpoint_epoch": metadata["epoch"],
        "parameters": metadata["parameter_count"],
        "trilinear": score(baseline, high),
        "model": score(prediction, high),
    }
    output = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
