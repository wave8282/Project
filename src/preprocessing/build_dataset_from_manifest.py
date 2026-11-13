"""Extract pose and build a multi-video dataset from a reviewed CSV manifest.

Required columns: video_path,label,source_id,dataset.
Optional columns: start_frame,end_frame (inclusive, original video frame IDs).
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
        start, end = row.get("start_frame", "").strip(), row.get("end_frame", "").strip()
        if bool(start) != bool(end):
            raise ValueError("start_frame and end_frame must either both be set or both be blank")
        if start:
            if int(start) < 0 or int(end) < int(start):
                raise ValueError(f"Invalid inclusive frame range: {start}..{end}")
    return rows


def filter_frame_range(source: Path, target: Path, start: int, end: int) -> int:
    count = 0
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames or reader.fieldnames[0] != "frame_index":
            raise ValueError(f"Unexpected pose CSV header in {source}")
        with target.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                frame_index = int(row["frame_index"])
                if start <= frame_index <= end:
                    writer.writerow(row)
                    count += 1
    if count == 0:
        target.unlink(missing_ok=True)
        raise ValueError(f"No detected pose frames in requested range {start}..{end} for {source}")
    return count


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

        start, end = row.get("start_frame", "").strip(), row.get("end_frame", "").strip()
        selected_csv = output_csv
        if start and end:
            selected_csv = output_csv.with_name(f"{stem}_frames_{start}_{end}_pose.csv")
            filter_frame_range(output_csv, selected_csv, int(start), int(end))
        outputs[row["label"]].append(selected_csv)

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
        "source_video_counts": {"normal": len(outputs["normal"]), "fall": len(outputs["fall"])},
        "reports": reports,
        "note": "Split by source-video groups, never by overlapping windows.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
