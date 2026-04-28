import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import compare_methods


class TestCompareMethodsMetrics(unittest.TestCase):
    def test_compare_methods_uses_unified_sample_evaluator(self):
        with patch("test.compare_methods.evaluate_doa_sample") as evaluator:
            evaluator.return_value = {"rmse": 1.25, "success": True}

            rmse, success = compare_methods.compute_rmse_with_hungarian(
                np.array([20.0, -10.0, 5.0]),
                np.array([-10.0, 5.0, 20.0]),
                tol=2.0,
            )

        evaluator.assert_called_once()
        np.testing.assert_allclose(evaluator.call_args.args[0], np.array([20.0, -10.0, 5.0]))
        np.testing.assert_allclose(evaluator.call_args.args[1], np.array([-10.0, 5.0, 20.0]))
        self.assertEqual(evaluator.call_args.kwargs["tol"], 2.0)
        self.assertEqual(rmse, 1.25)
        self.assertTrue(success)

    def test_default_refinement_is_none(self):
        with patch.object(sys, "argv", ["compare_methods.py"]):
            args = compare_methods.parse_args()

        self.assertEqual(args.refinement, "none")

    def test_result_record_includes_evaluation_protocol(self):
        args = SimpleNamespace(refinement="none", min_sep=5.0, grid_step=1.0, tol=2.0)

        result = compare_methods.build_result_record(
            snr=-10.0,
            args=args,
            music=(11.0, 0.2),
            esprit=(12.0, 0.3),
            vanilla=(1.5, 0.8),
            ca_doa=(1.2, 0.9),
        )

        self.assertEqual(result["refinement"], "none")
        self.assertEqual(result["min_sep"], 5.0)
        self.assertEqual(result["grid_step"], 1.0)
        self.assertEqual(result["tol"], 2.0)


if __name__ == "__main__":
    unittest.main()
