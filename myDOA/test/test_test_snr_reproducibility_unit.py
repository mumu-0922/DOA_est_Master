import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import test_snr


class TestSnrReproducibility(unittest.TestCase):
    def test_parse_args_exposes_test_seed(self):
        with patch.object(sys, "argv", ["test_snr.py", "--weights", "best_model.pth"]):
            args = test_snr.parse_args()

        self.assertEqual(args.seed, 42)


if __name__ == "__main__":
    unittest.main()
