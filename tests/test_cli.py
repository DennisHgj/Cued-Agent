from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_help_does_not_import_ml_runtime(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "run_inference.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--checkpoint", completed.stdout)
        self.assertIn("--lip-only", completed.stdout)


if __name__ == "__main__":
    unittest.main()
