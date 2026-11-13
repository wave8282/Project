"""Tests for body-relative temporal pose features."""
from __future__ import annotations

import unittest

import numpy as np

from src.preprocessing.engineer_features import engineer_sequence


class EngineerFeatureTests(unittest.TestCase):
    def test_feature_shape_and_finite_values(self):
        rng = np.random.default_rng(7)
        sequence = rng.uniform(0.2, 0.8, size=(30, 66)).astype(np.float32)
        result = engineer_sequence(sequence)
        self.assertEqual(result.shape, (30, 137))
        self.assertTrue(np.isfinite(result).all())

    def test_first_velocity_frame_is_zero(self):
        sequence = np.full((30, 66), 0.4, dtype=np.float32)
        result = engineer_sequence(sequence)
        np.testing.assert_allclose(result[0, 66:132], 0.0)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            engineer_sequence(np.zeros((30, 65), dtype=np.float32))
        invalid = np.zeros((30, 66), dtype=np.float32)
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            engineer_sequence(invalid)


if __name__ == "__main__":
    unittest.main()
