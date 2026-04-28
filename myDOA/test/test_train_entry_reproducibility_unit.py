import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_ca_doa, train_vanilla_cnn


class TestTrainEntryReproducibility(unittest.TestCase):
    def test_ca_train_entry_writes_metadata_and_checkpoint_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "train_ca_doa.py",
                "--device", "cpu",
                "--epochs", "1",
                "--train_samples", "1",
                "--val_samples", "1",
                "--batch_size", "1",
                "--base_channels", "16",
                "--num_blocks", "1",
                "--save_dir", tmpdir,
                "--exp_name", "unit",
            ]

            with patch.object(sys, "argv", argv):
                train_ca_doa.main()

            save_dir = Path(tmpdir) / "unit"
            metadata = json.loads((save_dir / "run_metadata.json").read_text(encoding="utf-8"))
            command = (save_dir / "command.txt").read_text(encoding="utf-8")
            checkpoint = torch.load(save_dir / "final_model.pth", map_location="cpu")

        self.assertEqual(metadata["run_type"], "train_ca_doa")
        self.assertEqual(metadata["config"]["epochs"], 1)
        self.assertIn("--epochs 1", command)
        self.assertEqual(checkpoint["run_metadata"]["run_type"], "train_ca_doa")
        self.assertEqual(checkpoint["run_metadata"]["config"]["exp_name"], "unit")
        self.assertIsNotNone(checkpoint["best_val_loss"])

    def test_vanilla_train_entry_writes_metadata_and_checkpoint_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "train_vanilla_cnn.py",
                "--device", "cpu",
                "--epochs", "1",
                "--train_samples", "1",
                "--val_samples", "1",
                "--batch_size", "1",
                "--save_dir", tmpdir,
                "--exp_name", "vanilla_unit",
            ]

            with patch.object(sys, "argv", argv):
                train_vanilla_cnn.main()

            save_dir = Path(tmpdir) / "vanilla_unit"
            metadata = json.loads((save_dir / "run_metadata.json").read_text(encoding="utf-8"))
            command = (save_dir / "command.txt").read_text(encoding="utf-8")
            checkpoint = torch.load(save_dir / "final_model.pth", map_location="cpu")

        self.assertEqual(metadata["run_type"], "train_vanilla_cnn")
        self.assertEqual(metadata["config"]["epochs"], 1)
        self.assertIn("--epochs 1", command)
        self.assertEqual(checkpoint["run_metadata"]["run_type"], "train_vanilla_cnn")
        self.assertEqual(checkpoint["run_metadata"]["config"]["exp_name"], "vanilla_unit")


if __name__ == "__main__":
    unittest.main()
