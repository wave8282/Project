"""Unit tests for frame-aware pose sequence generation.

Run with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.preprocessing.build_sequences import build_sequences, load_pose_csv


def write_pose_csv(path: Path, frame_indices: list[int]) -> None:
    header = ["frame_index"]
    for i in range(33):
        header.extend([f"lm_{i:02d}_x", f"lm_{i:02d}_y"])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for frame_index in frame_indices:
            writer.writerow([frame_index] + [0.25, 0.5] * 33)


class BuildSequencesTests(unittest.TestCase):
    def test_contiguous_frames_create_all_valid_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "normal.csv"
            write_pose_csv(path, list(range(40)))
            sequences, labels, groups, reports = build_sequences([path], 0, 30)

            self.assertEqual(len(sequences), 11)
            self.assertEqual(sequences[0].shape, (30, 66))
            self.assertEqual(labels, [0] * 11)
            self.assertEqual(set(groups), {"0:normal"})
            self.assertEqual(reports[0]["sequence_windows"], 11)

    def test_missing_frame_breaks_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "gap.csv"
            write_pose_csv(path, list(range(20)) + list(range(21, 51)))
            sequences, _, _, _ = build_sequences([path], 1, 30)

            # The 20-frame run is too short; the 30-frame run yields one window.
            self.assertEqual(len(sequences), 1)
            self.assertTrue(np.all(sequences[0] == sequences[0][0]))

    def test_legacy_66_column_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow([f"c{i}" for i in range(66)])
                writer.writerow([0.1] * 66)

            with self.assertRaisesRegex(ValueError, "frame_index"):
                load_pose_csv(path)

    def test_non_finite_values_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nan.csv"
            write_pose_csv(path, list(range(30)))
            text = path.read_text(encoding="utf-8")
            text = text.replace("0.25", "nan", 1)
            path.write_text(text, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                load_pose_csv(path)


if __name__ == "__main__":
    unittest.main()
