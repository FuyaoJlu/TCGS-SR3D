

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tcgs_sr3d import TCGSSR3D
from tcgs_sr3d.checkpoints import normalize_pair
from tcgs_sr3d.losses import tri_level_loss3d
from tcgs_sr3d.synthetic import SyntheticTraining, ValidationVolumes
from evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--validation", type=Path, default=Path("data/simulated_validation"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.samples is not None:
        config["samples_per_epoch"] = args.samples
    if config["epochs"] < 1 or config["samples_per_epoch"] < 1:
        parser.error("epochs and samples must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new output directory; existing runs are never overwritten")
    torch.set_num_threads(config.get("threads", 2))
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    model = TCGSSR3D(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    validation = ValidationVolumes(args.validation)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best = math.inf
    for epoch in range(1, config["epochs"]+1):
        model.train()
        dataset = SyntheticTraining(config["samples_per_epoch"], config["shape"],
                                    config["seed"]+(epoch-1)*config["samples_per_epoch"])
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        warmup = config["warmup_epochs"]
        if epoch <= warmup:
            lr_factor = epoch/max(warmup, 1)
        else:
            progress = (epoch-warmup-1)/max(config["epochs"]-warmup, 1)
            lr_factor = 0.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = config["lr"]*lr_factor
        event_weight = min(1.0, max(0.0, (epoch-warmup)/config["event_ramp_epochs"]))
        total = 0.0
        for batch in loader:
            low, high = batch["lr"].to(device), batch["hr"].to(device)
            low, high, _, _ = normalize_pair(low, high)
            optimizer.zero_grad(set_to_none=True)
            pred, aux = model(low, return_aux=True)
            loss, _ = tri_level_loss3d(pred, high, aux, structure_weight=event_weight, levels=model.levels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*low.shape[0]
        result = evaluate(model, validation, device, normalization="input_zscore")
        record = {"epoch": epoch, "train_loss": total/len(dataset),
                  "event_weight": event_weight, "validation": result}
        with (args.output/"history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
        if result["model"]["reconstruction_loss"] < best:
            best = result["model"]["reconstruction_loss"]
            torch.save({"model": model.state_dict(), "model_config": config["model"],
                        "epoch": epoch, "config": config,
                        "normalization": "input_zscore"}, args.output/"best.pt")
        print(json.dumps(record, allow_nan=False), flush=True)

if __name__ == "__main__":
    main()
