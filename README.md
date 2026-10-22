# TCGS-SR3D

Reflection-event-aware tri-level Gaussian reconstruction for 3D seismic
bandwidth enhancement and interpolation.

## Setup

Requires Python 3.10 or later and Git LFS.

```bash
git lfs install
git clone https://github.com/FuyaoJlu/TCGS-SR3D.git
cd TCGS-SR3D
git lfs pull
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

## Inference

```bash
python infer.py
```

The script loads `checkpoints/tcgs_sr3d_epoch145.pt` and the bundled example,
and writes `runs/inference/sr.npy`. For your own input:

```bash
python infer.py --input input.npy --output output.npy
```

Input: a real-valued `.npy` array in `[inline, time, crossline]` order, or
`[1, inline, time, crossline]`; `.npz` inputs use the key `lr`. Output: a float32
`.npy` array in the same axis order. Use `--device cuda` for GPU inference.

## Evaluation

```bash
python evaluate.py --output runs/evaluation/pretrained.json
```

## Training demo

```bash
python train.py --config configs/smoke.json --output runs/demo
```

For the full-size architecture:

```bash
python train.py --config configs/paper_architecture.json --device cuda --output runs/full_architecture
```

## Data layout

CIGBench: `CIG_ROOT/{train,val,test}/GT/*.npy`.
Private: `PRIVATE_ROOT/seis/*.dat` and `PRIVATE_ROOT/index/*.dat`.

## Training and evaluation

```bash
python train_tcgs3d.py --cigbench-root CIG_ROOT --private-root PRIVATE_ROOT --out-dir runs/train --levels tri --patch-size 64 64 64 --hidden-channels 224 --feature-channels 112 --num-blocks 28 --hr-blocks 4 --batch-size 3 --accumulation-steps 1
python evaluate_tcgs3d.py --checkpoint runs/train/best.pt --cigbench-root CIG_ROOT --private-root PRIVATE_ROOT --out-dir runs/evaluation
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) for the license notice.
