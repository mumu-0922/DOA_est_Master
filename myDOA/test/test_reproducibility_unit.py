import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.reproducibility import build_run_metadata, save_run_metadata


class TestReproducibilityMetadata(unittest.TestCase):
    def test_build_run_metadata_records_command_config_and_git(self):
        args = Namespace(seed=42, snr_list=[-20, -10, 0], weights=Path("model.pth"))

        metadata = build_run_metadata(
            args=args,
            run_type="test_snr",
            argv=["test_snr.py", "--weights", "model.pth"],
            cwd="D:/AI/DOA_est_Master/myDOA/test",
            git_info={"commit": "abc123", "branch": "master", "dirty": False},
            timestamp="2026-04-29T12:00:00",
            extra={"outputs": {"csv": "snr_test_results.csv"}},
        )

        self.assertEqual(metadata["run_type"], "test_snr")
        self.assertEqual(metadata["command"], "test_snr.py --weights model.pth")
        self.assertEqual(metadata["config"]["seed"], 42)
        self.assertEqual(metadata["config"]["weights"], "model.pth")
        self.assertEqual(metadata["git"]["commit"], "abc123")
        self.assertEqual(metadata["extra"]["outputs"]["csv"], "snr_test_results.csv")

    def test_save_run_metadata_writes_json_and_command_file(self):
        args = Namespace(seed=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = save_run_metadata(
                save_dir=tmpdir,
                args=args,
                run_type="compare_methods",
                argv=["compare_methods.py", "--seed", "7"],
                git_info={"commit": "abc123", "branch": "master", "dirty": True},
                timestamp="2026-04-29T12:00:00",
            )

            command_path = Path(tmpdir) / "command.txt"
            self.assertTrue(Path(metadata_path).exists())
            self.assertTrue(command_path.exists())

            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["config"]["seed"], 7)
            self.assertEqual(command_path.read_text(encoding="utf-8"), "compare_methods.py --seed 7\n")


if __name__ == "__main__":
    unittest.main()
