

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def make_volume(seed: int, shape=(16, 32, 16)) -> torch.Tensor:
    
    if len(shape) != 3 or any(n < 8 or n % 2 for n in shape):
        raise ValueError("All three target dimensions must be even and at least 8")
    rng = np.random.default_rng(seed)
    i, t, x = np.meshgrid(
        np.linspace(-1, 1, shape[0]), np.arange(shape[1]),
        np.linspace(-1, 1, shape[2]), indexing="ij",
    )
    volume = np.zeros(shape, dtype=np.float64)
    fault = rng.uniform(-0.4, 0.4)
    throw = rng.uniform(-1.5, 1.5)
    for centre in np.linspace(3, shape[1] - 4, 7):
        surface = centre + rng.uniform(-2, 2)*i + rng.uniform(-2, 2)*x
        surface += rng.uniform(-2, 2)*(i*i + 0.5*x*x)
        surface += throw * (i > fault)
        width = rng.uniform(0.7, 1.3)
        z = (t - surface) / width
        wavelet = (1 - 2*z*z)*np.exp(-z*z)
        amplitude = rng.uniform(0.5, 1.0) * rng.choice([-1, 1])
        volume += amplitude * wavelet
    volume /= max(float(np.max(np.abs(volume))), 1e-8)
    return torch.from_numpy(volume.astype(np.float32)).unsqueeze(0)

def degrade(high: torch.Tensor) -> torch.Tensor:
    
    if high.ndim != 4 or high.shape[0] != 1:
        raise ValueError("Expected [1, inline, time, crossline]")
    if any(n < 8 or n % 2 for n in high.shape[1:]):
        raise ValueError("Target dimensions must be even and at least 8")
    low = high.unsqueeze(0)
    for axis, sigma in enumerate((0.6, 2.0, 0.6)):
        radius = int(np.ceil(3*sigma))
        coordinates = torch.arange(-radius, radius+1, dtype=low.dtype, device=low.device)
        kernel = torch.exp(-0.5*(coordinates/sigma)**2)
        kernel /= kernel.sum()
        kernel_shape = [1, 1, 1, 1, 1]
        kernel_shape[axis+2] = kernel.numel()
        padding = [0]*6
        padding[2*(2-axis):2*(2-axis)+2] = [radius, radius]
        low = F.conv3d(F.pad(low, padding, mode="replicate"), kernel.reshape(kernel_shape))
    return F.interpolate(low, scale_factor=0.5, mode="trilinear", align_corners=False)[0]

class SyntheticTraining(Dataset):
    def __init__(self, count=16, shape=(16, 32, 16), seed=42):
        self.count, self.shape, self.seed = count, tuple(shape), seed

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        # Validation uses even seeds; training always uses odd seeds.
        high = make_volume(2*(self.seed + index)+1, self.shape)
        return {"lr": degrade(high), "hr": high}

class ValidationVolumes(Dataset):
    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No simulated validation volumes in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            low, high = data["lr"].copy(), data["hr"].copy()
        if high.ndim != 4 or high.shape[0] != 1 or low.shape != (1, *(n//2 for n in high.shape[1:])):
            raise ValueError("Invalid paired volume shapes")
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("Volume contains non-finite amplitudes")
        return {"lr": torch.from_numpy(low).float(), "hr": torch.from_numpy(high).float()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/simulated_validation"))
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--shape", nargs=3, type=int, default=(16, 32, 16))
    args = parser.parse_args()
    if args.count < 1 or args.seed < 0:
        parser.error("count must be positive and seed nonnegative")
    make_volume(2*args.seed, args.shape)  # Validate before creating outputs.
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Output directory must be empty to avoid replacing data")
    torch.set_num_threads(2)
    records = []
    for index in range(args.count):
        seed = 2*(args.seed+index)
        high = make_volume(seed, args.shape)
        filename = f"simulated_{index:03d}.npz"
        np.savez_compressed(args.output/filename, hr=high.numpy(), lr=degrade(high).numpy())
        records.append({"file": filename, "seed": seed})
    manifest = {"origin": "procedural Ricker reflections; no acquired data", "shape": args.shape,
                "axes": ["channel", "inline", "time", "crossline"],
                "blur_sigma": [0.6, 2.0, 0.6], "scale": 2, "volumes": records}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {args.count} simulated validation volumes")

if __name__ == "__main__":
    main()
