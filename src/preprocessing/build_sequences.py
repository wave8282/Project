"""Build labeled 30-frame pose sequences from per-video extraction CSVs.

Only CSVs produced by src/preprocessing/extract_pose.py are accepted. The
frame_index column is required so windows never bridge frames where pose
detection failed. Each input video is kept as a group for leakage-safe splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

LANDMARK_COLUMNS = 66


def load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted frame indices and (N, 66) coordinates from extractor CSV."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        header = file.readline().strip().split(",")
        if len(header) != LANDMARK_COLUMNS + 1 or header[0] != "frame_index":
            raise ValueError(
                f"{path}: expected extractor CSV with frame_index + 66 coordinates; "
                "legacy 66-column CSVs are not accepted because frame continuity "
                "and source provenance cannot be verified."
            )

        expected = ["frame_index"]
        for i in range(33):
            expected.extend([f"lm_{i:02d}_x", f"lm_{i:02d}_y"])
        if header != expected:
            raise ValueError(f"{path}: unexpected column names or order")

        rows: list[list[float]] = []
        for line_number, line in enumerate(file, start=2):
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split(",")
            if len(parts) != LANDMARK_COLUMNS + 1:
                raise ValueError(f"{path}:{line_number}: expected 67 columns")
            try:
                rows.append([float(value) for value in parts])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from exc

    if not rows:
        raise ValueError(f"{path}: no detected pose frames")

    values = np.asarray(rows, dtype=np.float64)
    frame_indices_float = values[:, 0]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: contains NaN or infinite values")
    if not np.all(frame_indices_float == np.floor(frame_indices_float)):
        raise ValueError(f"{path}: frame_index must contain integers")

    frame_indices = frame_indices_float.astype(np.int64)
    if np.any(np.diff(frame_indices) <= 0):
        raise ValueError(f"{path}: frame_index must be strictly increasing")

    coordinates = values[:, 1:]
    if np.any(coordinates < 0) or np.any(coordinates > 1):
        raise ValueError(f"{path}: coordinates outside normalized [0, 1] range")
    return frame_indices, coordinates.astype(np.float32)


def contiguous_runs(frame_indices: np.ndarray, coordinates: np.ndarray) -> Iterable[np.ndarray]:
    """Yield coordinate runs without gaps in original frame indices."""
    if len(frame_indices) == 0:
        return
    breaks = np.flatnonzero(np.diff(frame_indices) != 1) + 1
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(frame_indices)]
    for start, end in zip(starts, ends):
        yield coordinates[start:end]


def build_sequences(
    csv_paths: list[Path],
    label: int,
    sequence_length: int,
) -> tuple[list[np.ndarray], list[int], list[str], list[dict]]:
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    reports: list[dict] = []
    used_ids: set[str] = set()

    for path in csv_paths:
        source_id = f"{label}:{path.stem}"
        if source_id in used_ids:
            raise ValueError(f"Duplicate source ID: {source_id}")
        used_ids.add(source_id)

        frame_indices, coordinates = load_pose_csv(path)
        video_windows = 0
        for run in contiguous_runs(frame_indices, coordinates):
            for start in range(0, len(run) - sequence_length + 1):
                sequences.append(run[start : start + sequence_length])
                labels.append(label)
                groups.append(source_id)
                video_windows += 1

        reports.append({
            "source_id": source_id,
            "csv": str(path),
            "label": label,
            "detected_pose_frames": int(len(frame_indices)),
            "sequence_windows": int(video_windows),
            "sequence_length": sequence_length,
        })

    return sequences, labels, groups, reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled pose sequences without crossing video or frame gaps."
    )
    parser.add_argument("--normal", nargs="+", required=True, type=Path,
                        help="One or more normal-video extractor CSVs")
    parser.add_argument("--fall", nargs="+", required=True, type=Path,
                        help="One or more fall-video extractor CSVs")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/processed/pose_sequences.npz"))
    args = parser.parse_args()

    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")

    normal = build_sequences(args.normal, 0, args.sequence_length)
    fall = build_sequences(args.fall, 1, args.sequence_length)

    sequences = normal[0] + fall[0]
    labels = normal[1] + fall[1]
    groups = normal[2] + fall[2]
    reports = normal[3] + fall[3]

    if not sequences:
        raise RuntimeError("No complete sequences generated. Check video length and pose detection.")

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    group_ids = np.asarray(groups, dtype=str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y, groups=group_ids)

    class_counts = {
        "normal_0": int(np.sum(y == 0)),
        "fall_1": int(np.sum(y == 1)),
    }
    source_counts = {
        "normal_videos": int(sum(item["label"] == 0 for item in reports)),
        "fall_videos": int(sum(item["label"] == 1 for item in reports)),
    }
    summary = {
        "output": str(args.output),
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "class_sequence_counts": class_counts,
        "source_video_counts": source_counts,
        "groups_saved_for_video_level_split": True,
        "reports": reports,
        "warning": (
            "Do not randomly split overlapping windows. Split by source video using "
            "the saved groups. The current train/validation/test trainer requires "
            "at least three independent source videos per class."
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
