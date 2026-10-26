# TCGS-SR3D

Reference implementation of **Tri-level Coarse-to-fine Gaussian Splatting
(TCGS-SR3D)** for post-stack seismic bandwidth enhancement and regular 2x
grid refinement.

The package contains:

- training and validation code for CIGBench/Private volumes;
- dataset validation/manifest preparation;
- the paper's preprocessing and random-degradation pipeline for training and
  validation;
- a minimal demo that runs the packaged checkpoint on one precomputed patch.

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

## Data layout

CIGBench:

```text
CIG_ROOT/
  train/GT/*.npy
  val/GT/*.npy
  test/GT/*.npy
```

Private:

```text
PRIVATE_ROOT/
  seis/*.dat
  index/*.dat
```

Private volumes are 128 x 128 x 128 little-endian float32 arrays. The data
loader applies the split seed, random Gaussian degradation, noise, and training
trace dropout described in the paper.

## Dataset preparation

Validate the data roots and write the split manifest:

```bash
python prepare_data.py \
  --cigbench-root CIG_ROOT \
  --private-root PRIVATE_ROOT \
  --output data/manifest.json
```

## Training

```bash
python train_tcgs3d.py \
  --cigbench-root CIG_ROOT \
  --private-root PRIVATE_ROOT \
  --out-dir runs/train \
  --levels tri \
  --patch-size 64 64 64 \
  --hidden-channels 224 \
  --feature-channels 112 \
  --num-blocks 28 \
  --hr-blocks 4 \
  --batch-size 3 \
  --accumulation-steps 1
```

## Validation / evaluation

```bash
python evaluate_tcgs3d.py \
  --checkpoint runs/train/best.pt \
  --cigbench-root CIG_ROOT \
  --private-root PRIVATE_ROOT \
  --out-dir runs/evaluation
```

## Minimal demo

The demo uses one precomputed example patch rather than generating or degrading
data at run time:

```bash
python demo.py \
  --checkpoint checkpoints/tcgs_sr3d_epoch145.pt \
  --data data/demo_patch.npz
```

The patch contains a 64 x 64 x 64 target volume and its 32 x 32 x 32
low-resolution input. `demo.py` reports PSNR, SNR, MAE, gradient ratio,
Laplacian L1, and high-frequency L1 for the trilinear baseline and the model.
It does not perform any degradation.

The example is a single CIGBench synthetic patch. Its metrics are patch-level
and are not directly comparable to the full-volume Table 2 results in the
paper.

## Checkpoint

`checkpoints/tcgs_sr3d_epoch145.pt` is the released checkpoint (80,763,573
parameters). See `checkpoints/metadata.json` for its hash and configuration.

## License

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See [LICENSE](LICENSE) for the license notice.
