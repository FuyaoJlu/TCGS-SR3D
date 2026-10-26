from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

@dataclass(frozen=True)
class VolumeRecord:
    dataset: str
    split: str
    volume_id: str
    path: str
    shape: tuple[int, int, int]
    storage: str
    axis_order: str

def private_split_ids(root: str | Path, seed: int = 20260813) -> dict[str, list[str]]:
    root = Path(root)
    seismic_ids = {path.stem for path in (root / "seis").glob("*.dat")}
    index_ids = {path.stem for path in (root / "index").glob("*.dat")}
    paired = sorted(seismic_ids & index_ids, key=lambda value: int(value))
    if len(paired) < 20:
        raise RuntimeError(f"Expected a multi-volume Private corpus, found {len(paired)} paired volumes")
    generator = np.random.default_rng(seed)
    shuffled = [paired[index] for index in generator.permutation(len(paired))]
    train_end = int(round(0.8 * len(shuffled)))
    val_end = train_end + int(round(0.1 * len(shuffled)))
    return {"train": shuffled[:train_end], "val": shuffled[train_end:val_end], "test": shuffled[val_end:]}

def build_volume_records(
    cigbench_root: str | Path,
    private_root: str | Path,
    split: str,
    split_seed: int = 20260813,
) -> list[VolumeRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError(split)
    cigbench_root = Path(cigbench_root)
    private_root = Path(private_root)
    records = []
    for path in sorted((cigbench_root / split / "GT").glob("*.npy")):
        shape = tuple(int(value) for value in np.load(path, mmap_mode="r").shape)
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D CIGBench volume: {path} has {shape}")
        records.append(
            VolumeRecord(
                "cigbench", split, path.stem, str(path),
                shape, "npy", "inline,time,xline",
            )
        )
    index_ids = private_split_ids(private_root, seed=split_seed)[split]
    for volume_id in index_ids:
        path = private_root / "seis" / f"{volume_id}.dat"
        expected_bytes = 128**3 * np.dtype("<f4").itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"Unexpected Private file size: {path} ({path.stat().st_size})")
        records.append(
            VolumeRecord(
                "private", split, volume_id, str(path),
                (128, 128, 128), "dat-float32-le",
                "raw(n3,n2,n1)->inline(n2),time(n1),xline(n3)",
            )
        )
    return records

def write_data_manifest(
    path: str | Path,
    cigbench_root: str | Path,
    private_root: str | Path,
    split_seed: int = 20260813,
) -> dict:
    split_records = {
        split: build_volume_records(cigbench_root, private_root, split, split_seed)
        for split in ("train", "val", "test")
    }
    manifest = {
        "axis_order": "inline,time,xline",
        "split_unit": "independent 3D volume",
        "private_split_seed": split_seed,
        "counts": {
            split: {
                dataset: sum(record.dataset == dataset for record in records)
                for dataset in ("cigbench", "private")
            }
            for split, records in split_records.items()
        },
        "records": {
            split: [asdict(record) for record in records]
            for split, records in split_records.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

class ExpandedSeismicVolumeDataset(Dataset):
    

    def __init__(
        self,
        cigbench_root: str,
        private_root: str,
        split: str,
        patch_size: Sequence[int] = (48, 48, 48),
        scale: int = 2,
        samples_per_epoch: int = 2048,
        eval_patches_per_volume: int = 1,
        cigbench_probability: float = 0.35,
        sigma_time: tuple[float, float] = (1.2, 2.4),
        sigma_lateral: tuple[float, float] = (0.35, 0.90),
        noise_std: tuple[float, float] = (0.0, 0.05),
        augment: bool = True,
        seed: int = 20260813,
        only_dataset: str | None = None,
    ):
        self.split = split
        self.patch_size = tuple(int(value) for value in patch_size)
        self.scale = int(scale)
        self.samples_per_epoch = int(samples_per_epoch)
        self.eval_patches_per_volume = int(eval_patches_per_volume)
        self.cigbench_probability = float(cigbench_probability)
        self.sigma_time = sigma_time
        self.sigma_lateral = sigma_lateral
        self.noise_std = noise_std
        self.augment = bool(augment and split == "train")
        self.seed = int(seed)
        if any(size % self.scale for size in self.patch_size):
            raise ValueError(f"patch_size={self.patch_size} must be divisible by scale={self.scale}")
        records = build_volume_records(cigbench_root, private_root, split, split_seed=seed)
        if only_dataset is not None:
            records = [record for record in records if record.dataset == only_dataset]
        if not records:
            raise RuntimeError(f"No records for split={split}, only_dataset={only_dataset}")
        self.records = records
        self.records_by_dataset = {
            dataset: [record for record in records if record.dataset == dataset]
            for dataset in ("cigbench", "private")
        }
        self._cache: dict[str, np.ndarray] = {}
        self.eval_items = [
            (record_index, patch_index)
            for record_index in range(len(records))
            for patch_index in range(self.eval_patches_per_volume)
        ]

    def close(self) -> None:
        for volume in self._cache.values():
            base = volume
            closed = set()
            while base is not None:
                mmap = getattr(base, "_mmap", None)
                if mmap is not None and id(mmap) not in closed:
                    mmap.close()
                    closed.add(id(mmap))
                base = getattr(base, "base", None)
        self._cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.samples_per_epoch if self.split == "train" else len(self.eval_items)

    def _load_volume(self, record: VolumeRecord) -> np.ndarray:
        if record.path in self._cache:
            return self._cache[record.path]
        if record.storage == "npy":
            volume = np.load(record.path, mmap_mode="r")
        else:
            # Official storage is a[n3][n2][n1]. Canonical order is
            # (inline=n2, time=n1, xline=n3).
            raw = np.memmap(record.path, dtype="<f4", mode="r", shape=(128, 128, 128))
            volume = raw.transpose(1, 2, 0)
        self._cache[record.path] = volume
        return volume

    def _select_train_record(self) -> VolumeRecord:
        cigbench = self.records_by_dataset["cigbench"]
        private = self.records_by_dataset["private"]
        if cigbench and private:
            pool = cigbench if random.random() < self.cigbench_probability else private
        else:
            pool = cigbench or private
        return random.choice(pool)

    @staticmethod
    def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
        token = ":".join(str(part) for part in (seed, *parts)).encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=8).digest()
        return np.random.default_rng(int.from_bytes(digest, "little"))

    def _crop_start(self, shape: Sequence[int], rng: np.random.Generator) -> tuple[int, int, int]:
        starts = []
        for length, patch in zip(shape, self.patch_size):
            if length < patch:
                raise ValueError(f"Volume shape {tuple(shape)} is smaller than patch {self.patch_size}")
            starts.append(int(rng.integers(0, length - patch + 1)))
        return tuple(starts)

    def _degrade(self, source: np.ndarray, rng: np.random.Generator) -> tuple[torch.Tensor, dict[str, float]]:
        sigma_inline = float(rng.uniform(*self.sigma_lateral))
        sigma_time = float(rng.uniform(*self.sigma_time))
        sigma_xline = float(rng.uniform(*self.sigma_lateral))
        degraded = gaussian_filter(
            source.astype(np.float32, copy=False),
            sigma=(sigma_inline, sigma_time, sigma_xline),
            mode="reflect",
        ).astype(np.float32)
        noise = float(rng.uniform(*self.noise_std))
        if noise > 0:
            degraded += rng.normal(0.0, noise * max(float(degraded.std()), 1e-6), degraded.shape).astype(np.float32)
        missing_count = 0
        if self.split == "train" and rng.random() < 0.20:
            missing_count = int(rng.integers(1, 4))
            if rng.random() < 0.5:
                indices = rng.choice(degraded.shape[0], size=missing_count, replace=False)
                degraded[indices, :, :] = 0.0
            else:
                indices = rng.choice(degraded.shape[2], size=missing_count, replace=False)
                degraded[:, :, indices] = 0.0
        tensor = torch.from_numpy(np.ascontiguousarray(degraded)).unsqueeze(0).unsqueeze(0)
        low = F.interpolate(
            tensor,
            size=tuple(size // self.scale for size in self.patch_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return low, {
            "sigma_inline": sigma_inline,
            "sigma_time": sigma_time,
            "sigma_xline": sigma_xline,
            "noise_std_ratio": noise,
            "missing_traces": float(missing_count),
        }

    def __getitem__(self, index: int) -> dict:
        if self.split == "train":
            record = self._select_train_record()
            rng = np.random.default_rng(random.getrandbits(64))
        else:
            record_index, patch_index = self.eval_items[index]
            record = self.records[record_index]
            rng = self._stable_rng(self.seed, self.split, record.dataset, record.volume_id, patch_index)
        volume = self._load_volume(record)
        source_volume = volume
        start = self._crop_start(volume.shape, rng)
        slices = tuple(slice(offset, offset + size) for offset, size in zip(start, self.patch_size))
        target = np.asarray(volume[slices], dtype=np.float32).copy()
        source = np.asarray(source_volume[slices], dtype=np.float32).copy()
        target = np.nan_to_num(target)
        source = np.nan_to_num(source)

        if self.augment:
            if rng.random() < 0.5:
                target = target[::-1, :, :].copy()
                source = source[::-1, :, :].copy()
            if rng.random() < 0.5:
                target = target[:, :, ::-1].copy()
                source = source[:, :, ::-1].copy()
            if rng.random() < 0.5:
                target = target.transpose(2, 1, 0).copy()
                source = source.transpose(2, 1, 0).copy()
            if rng.random() < 0.5:
                target = -target
                source = -source
            amplitude = float(rng.uniform(0.75, 1.25))
            target *= amplitude
            source *= amplitude

        low, degradation = self._degrade(source, rng)
        mean = low.mean()
        std = low.std().clamp_min(1e-6)
        low = (low - mean) / std
        high = (torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0) - mean) / std
        return {
            "lr": low.contiguous(),
            "hr": high.contiguous(),
            "mean": mean,
            "std": std,
            "dataset": record.dataset,
            "volume_id": record.volume_id,
            "crop_start": torch.tensor(start, dtype=torch.int32),
            "degradation": {key: torch.tensor(value, dtype=torch.float32) for key, value in degradation.items()},
        }
