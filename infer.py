

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from tcgs_sr3d.checkpoints import load_checkpoint, predict

def load_input(path):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if "lr" not in data:
                raise ValueError("NPZ input must contain an lr array")
            volume = data["lr"].copy()
    elif path.suffix.lower() == ".npy":
        volume = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Input must be .npy or .npz")
    if volume.ndim == 3:
        volume = volume[None]
    if volume.ndim != 4 or volume.shape[0] != 1 or min(volume.shape[1:]) < 2:
        raise ValueError("Expected [inline,time,crossline] or [1,inline,time,crossline]")
    if not np.issubdtype(volume.dtype, np.number) or np.iscomplexobj(volume):
        raise ValueError("Input must contain real amplitudes")
    volume = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError("Input has NaN/Inf amplitudes")
    return torch.from_numpy(np.ascontiguousarray(volume)).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tcgs_sr3d_epoch145.pt"))
    parser.add_argument("--input", type=Path, default=Path("data/simulated_validation/simulated_000.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/inference/sr.npy"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy":
        parser.error("Output must use the .npy extension")
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    low = load_input(args.input).to(args.device)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    started = time.perf_counter()
    result = predict(model, low, metadata["normalization"])
    # Moving to CPU also synchronizes CUDA before measuring elapsed time.
    output = result[0,0].cpu().numpy()
    elapsed = time.perf_counter()-started
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite prediction; no output written")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, output, allow_pickle=False)
    print(json.dumps({"input_shape": list(low.shape), "output_shape": list(output.shape),
                      "checkpoint_epoch": metadata["epoch"], "parameters": metadata["parameter_count"],
                      "normalization": metadata["normalization"], "device": args.device,
                      "forward_seconds": elapsed, "finite": True}, indent=2))

if __name__ == "__main__":
    main()
