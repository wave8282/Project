"""Tests for external-video manifest validation."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.preprocessing.build_dataset_from_manifest import read_manifest, safe_name


class ManifestTests(unittest.TestCase):
    def test_reads_reviewed_manifest_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            path.write_text(
                "video_path,label,source_id,dataset\n"
                "clip_a.mp4,normal,subject_a,Demo\n"
                "clip_b.mp4,fall,subject_b,Demo\n",
                encoding="utf-8",
            )
            rows = read_manifest(path)
            self.assertEqual([row["label"] for row in rows], ["normal", "fall"])

    def test_rejects_duplicate_source_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            path.write_text(
                "video_path,label,source_id,dataset\n"
                "a.mp4,normal,same,Demo\n"
                "b.mp4,normal,same,Demo\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_manifest(path)

    def test_safe_name_removes_path_separators(self):
        self.assertEqual(safe_name("../subject 01"), "subject_01")


    def test_accepts_frame_range_and_rejects_incomplete_range(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            path.write_text(
                "video_path,label,source_id,dataset,start_frame,end_frame\\n"
                "a.mp4,fall,event_a,Demo,100,180\\n",
                encoding="utf-8",
            )
            self.assertEqual(read_manifest(path)[0]["start_frame"], "100")
            path.write_text(
                "video_path,label,source_id,dataset,start_frame,end_frame\\n"
                "a.mp4,fall,event_a,Demo,100,\\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both be set"):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
