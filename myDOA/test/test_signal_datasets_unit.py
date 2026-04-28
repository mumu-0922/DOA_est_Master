import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.signal_datasets import DOADataset


class TestDOADatasetGeneration(unittest.TestCase):
    def test_snr_is_sample_level_and_broadcast_to_sources(self):
        dataset = DOADataset(
            M=8,
            k=3,
            snr_range=(-20, 0),
            snap=16,
            num_samples=12,
            seed=123,
        )

        snr = dataset.data["snr"].numpy()

        self.assertEqual(snr.shape, (12, 3))
        self.assertTrue(np.allclose(snr, snr[:, [0]]))
        self.assertGreater(np.unique(snr[:, 0]).size, 1)

    def test_default_sigma_matches_grid_step(self):
        dataset = DOADataset(
            M=8,
            k=3,
            grid_step=1.0,
            snap=16,
            num_samples=2,
            seed=123,
        )

        self.assertEqual(dataset.sigma, 1.0)

    def test_rejects_invalid_input_channels(self):
        with self.assertRaises(ValueError):
            DOADataset(
                M=8,
                k=3,
                input_channels=3,
                snap=16,
                num_samples=2,
                seed=123,
            )


if __name__ == "__main__":
    unittest.main()
