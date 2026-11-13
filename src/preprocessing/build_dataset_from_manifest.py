"""Extract pose and build a multi-video dataset from a reviewed CSV manifest.

Manifest columns: video_path,label,source_id,dataset. Labels must be exactly
normal or fall. The source_id is retained in output filenames/groups so
windows from one source video cannot leak across data splits.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

from src.preprocessing.build_sequences import build_sequences


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError(f"Invalid empty source identifier: {value!r}")
    return cleaned


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"video_path", "label", "source_id", "dataset"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    seen = set()
    for row in rows:
        row["label"] = row["label"].strip().lower()
        if row["label"] not in {"normal", "fall"}:
            raise ValueError(f"Unsupported label {row['label']!r}; use normal or fall")
        key = (row["dataset"].strip(), row["source_id"].strip())
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate dataset/source_id: {key}")
        seen.add(key)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/pose_sequences.npz"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    outputs = {"normal": [], "fall": []}
    for row in rows:
        video = Path(row["video_path"]).expanduser()
        if not video.is_file():
            raise FileNotFoundError(video)
        stem = safe_name(f"{row['dataset']}_{row['source_id']}")
        output_csv = args.raw_dir / row["label"] / f"{stem}_pose.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "src/preprocessing/extract_pose.py",
            "--video", str(video), "--output", str(output_csv),
            "--max-width", str(args.max_width),
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        outputs[row["label"]].append(output_csv)

    if not outputs["normal"] or not outputs["fall"]:
        raise ValueError("Manifest must include at least one normal and one fall video")
    normal = build_sequences(outputs["normal"], 0, args.sequence_length)
    fall = build_sequences(outputs["fall"], 1, args.sequence_length)
    import json
    import numpy as np

    X = np.asarray(normal[0] + fall[0], dtype=np.float32)
    y = np.asarray(normal[1] + fall[1], dtype=np.int64)
    groups = np.asarray(normal[2] + fall[2], dtype=str)
    reports = normal[3] + fall[3]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y, groups=groups)
    summary = {
        "output": str(args.output),
        "X_shape": list(X.shape),
        "class_sequence_counts": {"normal_0": int(np.sum(y == 0)), "fall_1": int(np.sum(y == 1))},
        "source_video_counts": {
            "normal": len(outputs["normal"]), "fall": len(outputs["fall"])
        },
        "reports": reports,
        "note": "Split by source-video groups, never by overlapping windows.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
