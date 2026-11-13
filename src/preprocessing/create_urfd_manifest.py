"""Create a reviewable manifest from prepared URFD cam0 MP4 directories.

The script maps /fall/ files to fall and /normal/ files to normal. Review each
clip and fill start_frame/end_frame for event-only segments before training.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, default=Path("data/external/urfd/videos"))
    parser.add_argument("--output", type=Path, default=Path("data/external/urfd_manifest.csv"))
    args = parser.parse_args()

    rows = []
    for label in ("normal", "fall"):
        folder = args.video_dir / label
        for video in sorted(folder.glob("*.mp4")) if folder.exists() else []:
            rows.append({
                "video_path": str(video.resolve()),
                "label": label,
                "source_id": video.stem,
                "dataset": "URFD",
                "start_frame": "",
                "end_frame": "",
            })
    if not rows or not any(row["label"] == "normal" for row in rows) or not any(row["label"] == "fall" for row in rows):
        raise RuntimeError("Expected at least one video under both normal/ and fall/ directories")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["video_path", "label", "source_id", "dataset", "start_frame", "end_frame"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} review-required video rows to {args.output}")


if __name__ == "__main__":
    main()
