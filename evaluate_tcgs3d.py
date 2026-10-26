from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tcgs_sr3d import TCGSSR3D
from tcgs_sr3d.data import ExpandedSeismicVolumeDataset
from tcgs_sr3d.losses import metrics3d

def load_model(path: Path, device: torch.device) -> tuple[TCGSSR3D, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    cfg = checkpoint["args"]
    model = TCGSSR3D(
        scale=cfg["scale"],
        hidden_channels=cfg["hidden_channels"],
        feature_channels=cfg["feature_channels"],
        num_blocks=cfg["num_blocks"],
        hr_blocks=cfg["hr_blocks"],
        levels=cfg["levels"],
        gradient_checkpointing=False,
        mid_radius=cfg["mid_radius"],
        coarse_radius=cfg["coarse_radius"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, cfg

def spectrum3d(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = volume.astype(np.float64, copy=False)
    values = values - values.mean(axis=1, keepdims=True)
    window = np.hanning(values.shape[1])[None, :, None]
    amplitude = np.abs(np.fft.rfft(values * window, axis=1)).mean(axis=(0, 2))
    amplitude = amplitude / max(float(amplitude.max()), 1e-12)
    frequency = np.linspace(0.0, 0.5, len(amplitude))
    return frequency, amplitude

def save_slices(
    path: Path,
    low: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    aux: dict[str, torch.Tensor],
) -> None:
    up = F.interpolate(low, size=target.shape[-3:], mode="trilinear", align_corners=False)
    inline_index = target.shape[-3] // 2
    xline_index = target.shape[-1] // 2
    arrays = [
        ("Input", up[0, 0, inline_index].cpu().numpy()),
        ("TCGS-SR3D", pred[0, 0, inline_index].float().cpu().numpy()),
        ("Ground truth", target[0, 0, inline_index].cpu().numpy()),
        ("Absolute error", (pred - target).abs()[0, 0, inline_index].float().cpu().numpy()),
        ("Input", up[0, 0, :, :, xline_index].T.cpu().numpy()),
        ("TCGS-SR3D", pred[0, 0, :, :, xline_index].T.float().cpu().numpy()),
        ("Ground truth", target[0, 0, :, :, xline_index].T.cpu().numpy()),
        ("Absolute error", (pred - target).abs()[0, 0, :, :, xline_index].T.float().cpu().numpy()),
    ]
    seismic = np.concatenate([arrays[index][1].ravel() for index in (0, 1, 2, 4, 5, 6)])
    limit = float(np.percentile(np.abs(seismic), 99))
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="white")
    for index, (axis, (name, image)) in enumerate(zip(axes.flat, arrays)):
        if index in {3, 7}:
            axis.imshow(image, cmap="magma", vmin=0, vmax=float(np.percentile(image, 99)), aspect="auto")
        else:
            axis.imshow(image, cmap="gray", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_title(name, fontsize=10)
        axis.axis("off")
        axis.set_facecolor("white")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.10)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(path.with_suffix(suffix), dpi=220, facecolor="white")
    plt.close(figure)

    feature_panels = [
        ("Fine gate", aux["gates"][0, 0, inline_index]),
        ("Middle gate", aux["gates"][0, 1, inline_index]),
        ("Coarse gate", aux["gates"][0, 2, inline_index]),
        ("Inline dip", aux["coarse_dip_inline"][0, 0, inline_index]),
        ("Xline dip", aux["coarse_dip_xline"][0, 0, inline_index]),
        ("Visibility", aux["coarse_visibility"][0, 0, inline_index]),
        ("Termination", aux["coarse_termination"][0, 0, inline_index]),
        ("Feature norm", aux["feature_norm"][0, 0, inline_index]),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), facecolor="white")
    for axis, (name, image) in zip(axes.flat, feature_panels):
        axis.imshow(image.float().cpu().numpy(), cmap="viridis", aspect="auto")
        axis.set_title(name, fontsize=10)
        axis.axis("off")
        axis.set_facecolor("white")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.03, hspace=0.10)
    for suffix in (".png", ".pdf", ".svg"):
        figure.savefig(path.with_name(path.name + "_features").with_suffix(suffix), dpi=220, facecolor="white")
    plt.close(figure)

@torch.inference_mode()
def evaluate_patch_sets(args: argparse.Namespace, model: TCGSSR3D, cfg: dict, device: torch.device) -> dict:
    output = Path(args.out_dir)
    all_results = {}
    for split in args.splits:
        for dataset_name in ("cigbench", "private"):
            dataset = ExpandedSeismicVolumeDataset(
                args.cigbench_root,
                args.private_root,
                split,
                patch_size=cfg["patch_size"],
                scale=cfg["scale"],
                eval_patches_per_volume=args.eval_patches_per_volume,
                augment=False,
                seed=cfg["seed"],
                only_dataset=dataset_name,
            )
            loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
            rows = []
            representative = None
            spectra = {"Input": [], "TCGS-SR3D": [], "Ground truth": []}
            for batch in loader:
                low = batch["lr"].to(device, non_blocking=True)
                high = batch["hr"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred, aux = model(low, return_aux=True)
                row = {
                    "split": split,
                    "dataset": dataset_name,
                    "volume_id": batch["volume_id"][0],
                    **metrics3d(pred, high),
                }
                for index, branch in enumerate(("fine", "mid", "coarse")):
                    row[f"gate_{branch}"] = float(aux["gates"][:, index].mean())
                row["visibility_mean"] = float(aux["coarse_visibility"].mean())
                row["termination_mean"] = float(aux["coarse_termination"].mean())
                rows.append(row)
                up = F.interpolate(low.float(), size=high.shape[-3:], mode="trilinear", align_corners=False)
                for name, values in (
                    ("Input", up[0, 0]), ("TCGS-SR3D", pred[0, 0].float()), ("Ground truth", high[0, 0]),
                ):
                    spectra[name].append(spectrum3d(values.cpu().numpy())[1])
                if representative is None:
                    representative = (low, pred, high, aux)
            key = f"{dataset_name}_{split}"
            case_dir = output / key
            case_dir.mkdir(parents=True, exist_ok=True)
            with (case_dir / "patch_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
            summary = {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys}
            summary["volumes"] = len({row["volume_id"] for row in rows})
            summary["patches"] = len(rows)
            (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            if representative is not None:
                save_slices(case_dir / "representative", *representative)
            figure, axis = plt.subplots(figsize=(6.5, 4.2), facecolor="white")
            for name, arrays in spectra.items():
                mean_spectrum = np.mean(np.stack(arrays), axis=0)
                frequency = np.linspace(0.0, 0.5, len(mean_spectrum))
                axis.plot(frequency, 20 * np.log10(np.maximum(mean_spectrum, 1e-6)), label=name)
            axis.set_xlabel("Normalized temporal frequency")
            axis.set_ylabel("Normalized amplitude (dB)")
            axis.grid(True, alpha=0.25)
            axis.legend(frameon=False)
            figure.tight_layout()
            for suffix in (".png", ".pdf", ".svg"):
                figure.savefig(case_dir / f"spectrum{suffix}", dpi=220, facecolor="white")
            plt.close(figure)
            all_results[key] = summary
            dataset.close()
    return all_results

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cigbench-root", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    parser.add_argument("--eval-patches-per-volume", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_model(Path(args.checkpoint), device)
    report = {
        "checkpoint": args.checkpoint,
        "model_args": cfg,
        "protocol": "volume-level held-out 3D patches; metrics reported separately per dataset",
        "results": evaluate_patch_sets(args, model, cfg, device),
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "EVALUATION_COMPLETE").write_text("complete", encoding="utf-8")

if __name__ == "__main__":
    main()

