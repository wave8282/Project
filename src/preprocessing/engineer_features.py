"""Camera-shift and body-scale robust features for 33-point pose sequences.

Input is the existing (T, 66) x/y representation. Output is (T, 137):
66 body-relative coordinates, 66 first differences, and five global torso/
center/scale descriptors. The transform is deterministic and shared by
training and real-time inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def engineer_sequence(sequence: np.ndarray) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 66 or values.shape[0] < 2:
        raise ValueError("Expected a sequence shaped (T>=2, 66)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pose sequence contains NaN or infinite values")

    points = values.reshape(values.shape[0], 33, 2)
    shoulders = (points[:, 11] + points[:, 12]) * 0.5
    hips = (points[:, 23] + points[:, 24]) * 0.5
    center = hips
    torso = shoulders - hips
    scale = np.linalg.norm(torso, axis=1)
    body_height = np.ptp(points[:, :, 1], axis=1)
    scale = np.where(scale > 1e-3, scale, np.maximum(body_height, 0.05))
    scale = np.maximum(scale, 0.05)

    relative = (points - center[:, None, :]) / scale[:, None, None]
    relative = relative.reshape(values.shape[0], 66)
    velocity = np.vstack([np.zeros((1, 66), dtype=np.float32), np.diff(relative, axis=0)])

    angle = np.arctan2(torso[:, 0], -torso[:, 1])
    global_features = np.column_stack(
        [
            center[:, 0],
            center[:, 1],
            np.log(scale),
            np.sin(angle),
            np.cos(angle),
        ]
    ).astype(np.float32)

    result = np.concatenate([relative, velocity, global_features], axis=1).astype(np.float32)
    if result.shape != (values.shape[0], 137) or not np.all(np.isfinite(result)):
        raise RuntimeError("Feature engineering produced invalid output")
    return result


def engineer_dataset(input_path: Path, output_path: Path) -> dict:
    with np.load(input_path, allow_pickle=False) as data:
        for key in ("X", "y", "groups"):
            if key not in data.files:
                raise ValueError(f"Dataset is missing required array: {key}")
        raw_x = data["X"]
        y = data["y"]
        groups = data["groups"]
    if raw_x.ndim != 3 or raw_x.shape[2] != 66:
        raise ValueError(f"Expected X shaped (N, T, 66), got {raw_x.shape}")
    features = np.stack([engineer_sequence(seq) for seq in raw_x])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X=features, y=y, groups=groups)
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "raw_shape": list(raw_x.shape),
        "feature_shape": list(features.shape),
        "feature_count": 137,
        "feature_layout": {
            "0:66": "hip-centered and torso-scale-normalized x/y landmarks",
            "66:132": "per-frame first differences of normalized landmarks",
            "132:137": "hip center x/y, log torso scale, torso angle sin/cos",
        },
        "source_videos": int(len(np.unique(groups))),
        "label_counts": {str(int(k)): int(np.sum(y == k)) for k in np.unique(y)},
    }
    output_path.with_suffix(".features.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/pose_sequences.npz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pose_features.npz"))
    args = parser.parse_args()
    print(json.dumps(engineer_dataset(args.input, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
