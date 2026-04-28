import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trainer import Trainer


class TestTrainerCheckpoint(unittest.TestCase):
    def test_checkpoint_includes_reproducibility_metadata(self):
        run_metadata = {
            "run_type": "train_ca_doa",
            "config": {"seed": 42, "epochs": 3},
            "git": {"commit": "abc123", "branch": "master", "dirty": False},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(nn.Linear(2, 1), device="cpu", save_dir=tmpdir, run_metadata=run_metadata)
            trainer.compile(epochs=3)
            trainer.save_checkpoint(
                "checkpoint.pth",
                epoch=2,
                metrics={"loss": 0.5, "rmse": 1.25, "success_rate": 0.75},
                best_val_loss=0.5,
                best_epoch=2,
            )

            checkpoint = torch.load(Path(tmpdir) / "checkpoint.pth", map_location="cpu")

        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["metrics"]["rmse"], 1.25)
        self.assertEqual(checkpoint["best_val_loss"], 0.5)
        self.assertEqual(checkpoint["best_epoch"], 2)
        self.assertEqual(checkpoint["run_metadata"]["config"]["seed"], 42)
        self.assertEqual(checkpoint["run_metadata"]["git"]["commit"], "abc123")


if __name__ == "__main__":
    unittest.main()
