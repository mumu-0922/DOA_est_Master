import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.doa_evaluation import evaluate_doa_batch, evaluate_doa_sample


class TestDOAEvaluation(unittest.TestCase):
    def test_sample_rmse_is_order_invariant(self):
        result = evaluate_doa_sample(
            doa_est=np.array([20.0, -10.0, 5.0]),
            doa_true=np.array([-10.0, 5.0, 20.0]),
            tol=2.0,
        )

        self.assertEqual(result["rmse"], 0.0)
        self.assertTrue(result["success"])
        self.assertTrue(np.allclose(result["matched_errors"], [0.0, 0.0, 0.0]))

    def test_success_requires_all_matched_errors_below_tolerance(self):
        result = evaluate_doa_sample(
            doa_est=np.array([-10.5, 5.5, 23.0]),
            doa_true=np.array([-10.0, 5.0, 20.0]),
            tol=2.0,
        )

        self.assertFalse(result["success"])
        self.assertAlmostEqual(result["rmse"], np.sqrt((0.5**2 + 0.5**2 + 3.0**2) / 3))

    def test_batch_uses_peak_success_mask_and_all_samples_for_success_rate(self):
        result = evaluate_doa_batch(
            doa_est=np.array([
                [20.0, -10.0, 5.0],
                [-10.0, 5.0, 20.0],
            ]),
            doa_true=np.array([
                [-10.0, 5.0, 20.0],
                [-10.0, 5.0, 20.0],
            ]),
            peak_success=np.array([True, False]),
            tol=2.0,
        )

        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["num_valid"], 1)
        self.assertEqual(result["num_success"], 1)
        self.assertEqual(result["num_total"], 2)
        self.assertEqual(result["success_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
