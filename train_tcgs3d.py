from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tcgs_sr3d import TCGSSR3D
from tcgs_sr3d.data import ExpandedSeismicVolumeDataset, write_data_manifest
from tcgs_sr3d.losses import metrics3d, tri_level_loss3d

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dataset(args: argparse.Namespace, split: str, only_dataset: str | None = None):
    return ExpandedSeismicVolumeDataset(
        cigbench_root=args.cigbench_root,
        private_root=args.private_root,
        split=split,
        patch_size=args.patch_size,
        scale=args.scale,
        samples_per_epoch=args.samples_per_epoch,
        eval_patches_per_volume=args.eval_patches_per_volume,
        cigbench_probability=args.cigbench_probability,
        sigma_time=(args.sigma_time_min, args.sigma_time_max),
        sigma_lateral=(args.sigma_lateral_min, args.sigma_lateral_max),
        noise_std=(args.noise_std_min, args.noise_std_max),
        augment=split == "train",
        seed=args.seed,
        only_dataset=only_dataset,
    )

def make_loader(args: argparse.Namespace, split: str, only_dataset: str | None = None) -> DataLoader:
    dataset = make_dataset(args, split, only_dataset)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if split == "train" else args.val_batch_size,
        shuffle=split == "train",
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=split == "train",
    )

def make_model(args: argparse.Namespace) -> TCGSSR3D:
    return TCGSSR3D(
        scale=args.scale,
        hidden_channels=args.hidden_channels,
        feature_channels=args.feature_channels,
        num_blocks=args.num_blocks,
        hr_blocks=args.hr_blocks,
        levels=args.levels,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mid_radius=args.mid_radius,
        coarse_radius=args.coarse_radius,
    )

@torch.inference_mode()
def evaluate_loader(
    model: TCGSSR3D,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    structure_weight: float,
) -> tuple[dict[str, float], tuple | None]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    snapshot = None
    for batch in loader:
        low = batch["lr"].to(device, non_blocking=True)
        high = batch["hr"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            pred, aux = model(low, return_aux=True)
            loss, parts = tri_level_loss3d(
                pred, high, aux, structure_weight=structure_weight, levels=model.levels
            )
        item = {"loss": float(loss), **metrics3d(pred, high)}
        item.update({f"loss_{key}": float(value) for key, value in parts.items()})
        item["selection_loss"] = (
            item["loss_voxel"] + 0.55 * item["loss_grad"]
            + 0.25 * item["loss_lap"] + 0.20 * item["loss_hf"]
        )
        for branch_index, branch in enumerate(("fine", "mid", "coarse")):
            item[f"gate_{branch}"] = float(aux["gates"][:, branch_index].mean())
        item["visibility_mean"] = float(aux["coarse_visibility"].mean())
        item["termination_mean"] = float(aux["coarse_termination"].mean())
        batch_size = low.shape[0]
        for key, value in item.items():
            sums[key] = sums.get(key, 0.0) + value * batch_size
        count += batch_size
        if snapshot is None:
            snapshot = (
                low[0:1].float().cpu(), high[0:1].float().cpu(), pred[0:1].float().cpu(),
                {key: value[0:1].float().cpu() for key, value in aux.items()},
            )
    model.train()
    return {key: value / max(count, 1) for key, value in sums.items()}, snapshot

def save_snapshot(snapshot: tuple, path: Path) -> None:
    low, high, pred, aux = snapshot
    up = torch.nn.functional.interpolate(low, size=high.shape[-3:], mode="trilinear", align_corners=False)
    inline_index = high.shape[-3] // 2
    xline_index = high.shape[-1] // 2
    panels = [
        ("Input", up[0, 0, inline_index]),
        ("Prediction", pred[0, 0, inline_index]),
        ("Ground truth", high[0, 0, inline_index]),
        ("Absolute error", (pred - high).abs()[0, 0, inline_index]),
        ("Fine gate", aux["gates"][0, 0, inline_index]),
        ("Middle gate", aux["gates"][0, 1, inline_index]),
        ("Coarse gate", aux["gates"][0, 2, inline_index]),
        ("Visibility", aux["coarse_visibility"][0, 0, inline_index]),
        ("Termination", aux["coarse_termination"][0, 0, inline_index]),
        ("Feature norm", aux["feature_norm"][0, 0, inline_index]),
        ("Xline prediction", pred[0, 0, :, :, xline_index].T),
        ("Xline truth", high[0, 0, :, :, xline_index].T),
    ]
    values = torch.cat((high.flatten(), pred.flatten())).numpy()
    limit = float(np.percentile(np.abs(values), 99))
    figure, axes = plt.subplots(3, 4, figsize=(13.5, 9.2), facecolor="white")
    for index, (axis, (name, image)) in enumerate(zip(axes.flat, panels)):
        array = image.numpy()
        if index in {0, 1, 2, 10, 11}:
            axis.imshow(array, cmap="gray", vmin=-limit, vmax=limit, aspect="auto")
        elif index == 3:
            axis.imshow(array, cmap="magma", vmin=0, vmax=float(np.percentile(array, 99)), aspect="auto")
        else:
            axis.imshow(array, cmap="viridis", aspect="auto")
        axis.set_title(name, fontsize=9)
        axis.axis("off")
        axis.set_facecolor("white")
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96, wspace=0.03, hspace=0.12)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)

def checkpoint_payload(model: TCGSSR3D, args: argparse.Namespace, epoch: int, row: dict) -> dict:
    return {
        "model": {
            key: value.detach().cpu().to(torch.bfloat16) if value.is_floating_point() else value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "args": vars(args),
        "epoch": epoch,
        "metrics": row,
        "parameter_count": model.parameter_count(),
        "axis_order": "inline,time,xline",
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cigbench-root", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--levels", choices=("fine", "fine_mid", "tri"), default="tri")
    parser.add_argument("--structure-weight", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--patch-size", nargs=3, type=int, default=(48, 48, 48))
    parser.add_argument("--hidden-channels", type=int, default=192)
    parser.add_argument("--feature-channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=24)
    parser.add_argument("--hr-blocks", type=int, default=4)
    parser.add_argument("--mid-radius", type=int, default=2)
    parser.add_argument("--coarse-radius", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--eval-patches-per-volume", type=int, default=1)
    parser.add_argument("--cigbench-probability", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--structure-warmup-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--sigma-time-min", type=float, default=1.2)
    parser.add_argument("--sigma-time-max", type=float, default=2.4)
    parser.add_argument("--sigma-lateral-min", type=float, default=0.35)
    parser.add_argument("--sigma-lateral-max", type=float, default=0.90)
    parser.add_argument("--noise-std-min", type=float, default=0.0)
    parser.add_argument("--noise-std-max", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-state", default=None)
    parser.add_argument("--resume-save-every", type=int, default=5)
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "snapshots").mkdir(exist_ok=True)
    (output / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    write_data_manifest(output / "data_manifest.json", args.cigbench_root, args.private_root, args.seed)

    device = torch.device(args.device)
    train_loader = make_loader(args, "train")
    val_loaders = {
        "cigbench": make_loader(args, "val", "cigbench"),
        "private": make_loader(args, "val", "private"),
    }
    model = make_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
    )
    log_path = output / "train_log.jsonl"
    best = math.inf
    stale = 0
    step = 0
    start_epoch = 1
    started = time.time()
    resume_path = Path(args.resume_state) if args.resume_state else None
    if resume_path is not None and resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        step = int(state["step"])
        best = float(state["best"])
        stale = int(state["stale"])
        print(json.dumps({"event": "resume", "start_epoch": start_epoch}), flush=True)
    print(json.dumps({
        "event": "start", "parameter_count": model.parameter_count(),
        "axis_order": "inline,time,xline", "args": vars(args),
    }), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if epoch <= args.warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = args.lr * epoch / args.warmup_epochs
        structure_ramp = min(1.0, max(0.0, (epoch - args.warmup_epochs) / max(args.structure_warmup_epochs, 1)))
        active_structure_weight = args.structure_weight * structure_ramp
        optimizer.zero_grad(set_to_none=True)
        train_sum = 0.0
        train_count = 0
        for batch_index, batch in enumerate(train_loader, 1):
            low = batch["lr"].to(device, non_blocking=True)
            high = batch["hr"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                pred, aux = model(low, return_aux=True)
                loss, _ = tri_level_loss3d(
                    pred, high, aux, structure_weight=active_structure_weight, levels=args.levels
                )
                scaled_loss = loss / args.accumulation_steps
            scaled_loss.backward()
            if batch_index % args.accumulation_steps == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            train_sum += float(loss.detach()) * low.shape[0]
            train_count += low.shape[0]

        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_sum / max(train_count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "structure_weight": active_structure_weight,
            "elapsed_sec": time.time() - started,
        }
        snapshots = {}
        for dataset, loader in val_loaders.items():
            values, snapshots[dataset] = evaluate_loader(
                model, loader, device, args.amp, active_structure_weight
            )
            row.update({f"val_{dataset}_{key}": value for key, value in values.items()})
        row["val_selection_loss"] = (
            0.7 * row["val_cigbench_selection_loss"]
            + 0.3 * row["val_private_selection_loss"]
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        improved = row["val_selection_loss"] < best - args.early_stop_min_delta
        if improved:
            best = row["val_selection_loss"]
            stale = 0
            torch.save(checkpoint_payload(model, args, epoch, row), output / "best.pt")
            for dataset, snapshot in snapshots.items():
                if snapshot is not None:
                    save_snapshot(snapshot, output / "snapshots" / f"best_{dataset}.png")
        else:
            stale += 1
        if epoch % 10 == 0:
            for dataset, snapshot in snapshots.items():
                if snapshot is not None:
                    save_snapshot(snapshot, output / "snapshots" / f"epoch_{epoch:04d}_{dataset}.png")
        if resume_path is not None and epoch % args.resume_save_every == 0:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step,
                "best": best, "stale": stale,
            }, resume_path)
        if epoch > args.warmup_epochs:
            scheduler.step()
        if args.early_stop_patience > 0 and stale >= args.early_stop_patience:
            event = {"event": "early_stop", "epoch": epoch, "best_val_loss": best, "stale_epochs": stale}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            break

    (output / "TRAINING_COMPLETE").write_text(json.dumps({"best_val_loss": best}), encoding="utf-8")
    if resume_path is not None and resume_path.exists():
        resume_path.unlink()

if __name__ == "__main__":
    main()

