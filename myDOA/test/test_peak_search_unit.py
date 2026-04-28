import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_network import GridBasedNetwork
from utils.peak_search import (
    min_sep_to_bins,
    refine_peak_centroid,
    refine_peak_parabolic,
    select_peaks_with_suppression,
)


class TestPeakSearch(unittest.TestCase):
    def test_min_sep_uses_ceiling_bins(self):
        self.assertEqual(min_sep_to_bins(min_sep=5.0, grid_step=2.0), 3)
        self.assertEqual(min_sep_to_bins(min_sep=0.0, grid_step=1.0), 0)

    def test_suppression_selects_sorted_top_peaks(self):
        spectrum = np.zeros(21)
        spectrum[5] = 10.0
        spectrum[7] = 9.0
        spectrum[15] = 8.0

        peaks = select_peaks_with_suppression(spectrum, k=2, min_sep_bins=3)

        self.assertEqual(peaks.tolist(), [5, 15])

    def test_numpy_nms_matches_grid_network_nms_without_refinement(self):
        spectrum = np.zeros(121, dtype=np.float32)
        spectrum[40] = 5.0
        spectrum[45] = 4.5
        spectrum[80] = 4.0

        model = GridBasedNetwork(start_angle=-60, end_angle=60, step=1)
        success, theta = model.spectrum_to_doa(
            torch.tensor(spectrum[None, :]),
            k=2,
            min_sep=5.0,
        )

        peaks = select_peaks_with_suppression(spectrum, k=2, min_sep_bins=5)
        grid = np.arange(-60, 60 + 0.001, 1.0)

        self.assertTrue(bool(success.item()))
        self.assertTrue(np.allclose(theta.squeeze(0).numpy(), grid[peaks]))

    def test_parabolic_refinement_clamps_to_half_bin(self):
        grid = np.arange(-2, 3, dtype=np.float64)
        spectrum = np.array([0.0, 1.0, 3.0, 2.0, 0.0])

        refined = refine_peak_parabolic(spectrum, grid, peak_idx=2)

        self.assertGreater(refined, 0.0)
        self.assertLessEqual(refined, 0.5)

    def test_centroid_refinement_uses_non_negative_weights(self):
        grid = np.arange(-3, 4, dtype=np.float64)
        spectrum = np.array([0.0, -10.0, 1.0, 4.0, 3.0, -10.0, 0.0])

        refined = refine_peak_centroid(spectrum, grid, peak_idx=3, window=2)

        expected = ((-1.0 * 1.0) + (0.0 * 4.0) + (1.0 * 3.0)) / (1.0 + 4.0 + 3.0)
        self.assertAlmostEqual(refined, expected)


if __name__ == "__main__":
    unittest.main()
