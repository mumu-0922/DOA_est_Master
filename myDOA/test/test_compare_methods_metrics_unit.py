import os
import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
