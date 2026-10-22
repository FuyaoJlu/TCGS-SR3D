

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

class EntrypointTests(unittest.TestCase):
    def check_help(self, script):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cigbench-root", result.stdout)
        self.assertIn("--private-root", result.stdout)

    def test_external_training_help(self):
        self.check_help("train_tcgs3d.py")

    def test_external_evaluation_help(self):
        self.check_help("evaluate_tcgs3d.py")

if __name__ == "__main__":
    unittest.main()
