import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from infer import load_input
from tcgs_sr3d.checkpoints import load_checkpoint, normalize_pair, predict
from tcgs_sr3d import TCGSSR3D

class ReleaseTests(unittest.TestCase):
    def test_packaged_checkpoint_hash(self):
        meta = json.loads(Path("checkpoints/metadata.json").read_text())
        with Path(meta["checkpoint"]).open("rb") as stream:
            checksum = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                checksum.update(chunk)
        self.assertEqual(checksum.hexdigest(), meta["release_sha256"])

    def test_normalization_roundtrip(self):
        low = torch.arange(64).reshape(1,1,4,4,4).float()
        normalized, high, mean, std = normalize_pair(low, low)
        self.assertTrue(torch.allclose(normalized*std + mean, low, atol=1e-5))
        self.assertTrue(torch.equal(normalized, high))
        constant = torch.ones_like(low)
        self.assertTrue(torch.isfinite(normalize_pair(constant)[0]).all())

    def test_original_training_checkpoint_normalization(self):
        config = dict(scale=2, hidden_channels=8, feature_channels=8, num_blocks=1,
                      hr_blocks=1, levels="tri", mid_radius=2, coarse_radius=2)
        model = TCGSSR3D(**config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"legacy.pt"
            torch.save({"model": model.state_dict(), "args": config, "epoch": 1}, path)
            _, metadata = load_checkpoint(path)
            self.assertEqual(metadata["normalization"], "input_zscore")

    def test_real_checkpoint_inference(self):
        torch.set_num_threads(2)
        model, metadata = load_checkpoint("checkpoints/tcgs_sr3d_epoch145.pt")
        self.assertEqual(metadata["epoch"], 145)
        self.assertEqual(metadata["parameter_count"], 80763573)
        low = load_input(Path("data/simulated_validation/simulated_000.npz"))
        prediction = predict(model, low, metadata["normalization"])
        self.assertEqual(tuple(prediction.shape), (1,1,16,32,16))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_cli_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"sr.npy"
            command = [sys.executable, "infer.py", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            data = np.load(output, allow_pickle=False)
            self.assertEqual(data.shape, (16,32,16))
            self.assertEqual(data.dtype, np.dtype("float32"))
            self.assertTrue(np.isfinite(data).all())
            retry = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertTrue(np.array_equal(data, np.load(output, allow_pickle=False)))

    def test_invalid_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input.npy"
            np.save(path, np.full((4,4,4), np.nan, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_input(path)

if __name__ == "__main__":
    unittest.main()
