import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from tcgs_sr3d import TCGSSR3D
from tcgs_sr3d.losses import multiscale_gate_target3d, tri_level_loss3d
from tcgs_sr3d.synthetic import make_volume, degrade, SyntheticTraining, ValidationVolumes

class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_synthetic_determinism_and_shape(self):
        high = make_volume(3, (8, 16, 8))
        self.assertTrue(torch.equal(high, make_volume(3, (8, 16, 8))))
        self.assertFalse(torch.equal(high, make_volume(4, (8, 16, 8))))
        self.assertEqual(tuple(degrade(high).shape), (1, 4, 8, 4))
        self.assertLessEqual(high.abs().max().item(), 1.0)
        self.assertTrue(torch.equal(SyntheticTraining(1, (8,16,8), 1)[0]["hr"], high))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            make_volume(3, (7, 16, 8))

    def test_gate_target(self):
        target = multiscale_gate_target3d(make_volume(3, (8, 16, 8)).unsqueeze(0))
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.allclose(target.sum(1), torch.ones_like(target[:, 0]), atol=1e-6))

    def test_forward_backward_all_levels(self):
        high = make_volume(3, (8, 16, 8)).unsqueeze(0)
        low = degrade(high[0]).unsqueeze(0)
        for levels in ("fine", "fine_mid", "tri"):
            with self.subTest(levels=levels):
                torch.manual_seed(42)
                model = TCGSSR3D(hidden_channels=8, feature_channels=8, num_blocks=1,
                                hr_blocks=1, levels=levels, gradient_checkpointing=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                pred, aux = model(low, return_aux=True)
                self.assertEqual(pred.shape, high.shape)
                self.assertTrue(torch.allclose(pred, aux["baseline"], atol=1e-6))
                self.assertTrue(torch.allclose(aux["gates"].sum(1), torch.ones_like(pred[:,0]), atol=1e-6))
                loss, parts = tri_level_loss3d(pred, high, aux, levels=levels)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
                self.assertGreater(model.fine_value.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
                self.assertTrue(torch.isfinite(model(low)).all())

    def test_bundled_validation(self):
        data = ValidationVolumes("data/simulated_validation")
        self.assertEqual(len(data), 2)
        for sample in data:
            self.assertTrue(torch.allclose(sample["lr"], degrade(sample["hr"]), atol=1e-6))

    def test_training_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/"run"
            subprocess.run([sys.executable, "train.py", "--epochs", "2", "--samples", "1",
                            "--output", str(run)], check=True, capture_output=True, text=True)
            report = run/"evaluation.json"
            subprocess.run([sys.executable, "evaluate.py", "--checkpoint", str(run/"best.pt"),
                            "--output", str(report)], check=True, capture_output=True, text=True)
            result = json.loads(report.read_text())
            self.assertEqual(result["n_volumes"], 2)
            self.assertIn("trilinear", result)
            self.assertGreater(result["model"]["reconstruction_loss"], 0)

if __name__ == "__main__":
    unittest.main()
