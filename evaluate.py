

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tcgs_sr3d import TCGSSR3D
from tcgs_sr3d.checkpoints import load_checkpoint, predict
from tcgs_sr3d.losses import charbonnier, gradients3d, laplacian3d, high_frequency3d, metrics3d
from tcgs_sr3d.synthetic import ValidationVolumes

def scores(pred, target):
    grad = sum(charbonnier(a, b) for a, b in zip(gradients3d(pred), gradients3d(target)))
    reconstruction = charbonnier(pred, target) + 0.55*grad
    reconstruction += 0.25*charbonnier(laplacian3d(pred), laplacian3d(target))
    reconstruction += 0.20*charbonnier(high_frequency3d(pred), high_frequency3d(target))
    return {**metrics3d(pred, target), "mae": float((pred-target).abs().mean()),
            "reconstruction_loss": float(reconstruction)}

@torch.inference_mode()
def evaluate(model, dataset, device, normalization="none"):
    model.eval()
    totals = {"model": {}, "trilinear": {}}
    for index in range(len(dataset)):
        sample = dataset[index]
        low, target = sample["lr"].unsqueeze(0).to(device), sample["hr"].unsqueeze(0).to(device)
        pred = predict(model, low, normalization)
        baseline = F.interpolate(low, size=target.shape[-3:], mode="trilinear", align_corners=False)
        for name, values in (("model", scores(pred, target)), ("trilinear", scores(baseline, target))):
            for key, value in values.items():
                totals[name][key] = totals[name].get(key, 0.0) + value/len(dataset)
    return {"n_volumes": len(dataset), **totals}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tcgs_sr3d_epoch145.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/simulated_validation"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    result = evaluate(model, ValidationVolumes(args.data), torch.device(args.device), metadata["normalization"])
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(output+"\n")
    print(output)

if __name__ == "__main__":
    main()
