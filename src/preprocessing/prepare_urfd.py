"""Convert downloaded URFD cam0 RGB image-sequence ZIPs to MP4 videos.

The script does not download or redistribute the dataset. Obtain the archives
from the official URFD page, review its non-commercial licence, and pass the
directory containing fall-XX-cam0-rgb.zip and adl-XX-cam0-rgb.zip files.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np


ARCHIVE_RE = re.compile(r"^(fall|adl)-(\d+)-cam0-rgb\.zip$", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def convert_archive(archive_path: Path, output_path: Path, fps: float, overwrite: bool = False) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            [m for m in archive.infolist() if Path(m.filename).suffix.lower() in IMAGE_SUFFIXES],
            key=lambda m: natural_key(m.filename),
        )
        if not members:
            raise ValueError(f"No image frames found in {archive_path}")

        writer = None
        written = 0
        try:
            for member in members:
                encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create video writer for {output_path}")
                elif frame.shape[1::-1] != (width, height):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                written += 1
        finally:
            if writer is not None:
                writer.release()
    if written == 0:
        raise RuntimeError(f"No decodable image frames in {archive_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/urfd/videos"))
    parser.add_argument("--fps", type=float, required=True, help="Use the dataset's documented/synchronized frame rate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    archives = sorted(args.source_dir.glob("*.zip"))
    converted = []
    for archive in archives:
        match = ARCHIVE_RE.match(archive.name)
        if not match:
            continue
        kind, number = match.group(1).lower(), int(match.group(2))
        label = "fall" if kind == "fall" else "normal"
        output = args.output_dir / label / f"urfd_{kind}_{number:02d}_cam0.mp4"
        count = convert_archive(archive, output, args.fps, args.overwrite)
        converted.append({"archive": archive.name, "video": str(output), "label": label, "frames": count})
    if not converted:
        raise RuntimeError("No matching URFD cam0 RGB ZIP archives found")
    for item in converted:
        print(f'{item["label"]}: {item["archive"]} -> {item["video"]} ({item["frames"]} frames)')


if __name__ == "__main__":
    main()
