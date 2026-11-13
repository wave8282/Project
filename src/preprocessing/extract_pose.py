"""Extract frame-indexed MediaPipe Pose x/y landmarks from a video.

The output CSV contains the original frame index plus 33 x/y landmark pairs.
Frames without a detected pose are omitted, but their absence is represented by
gaps in frame_index so downstream sequence generation cannot bridge the gap.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp


def build_header() -> list[str]:
    header = ["frame_index"]
    for landmark_index in range(33):
        header.extend([f"lm_{landmark_index:02d}_x", f"lm_{landmark_index:02d}_y"])
    return header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize wider frames to this width for faster processing; use 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing CSV and metadata file.",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if args.max_width < 0:
        parser.error("--max-width must be 0 or a positive integer")
    if not 0.0 < args.min_detection_confidence <= 1.0:
        parser.error("--min-detection-confidence must be in (0, 1]")
    if not 0.0 < args.min_tracking_confidence <= 1.0:
        parser.error("--min-tracking-confidence must be in (0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".metadata.json")
    if not args.overwrite and (args.output.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Output already exists: {args.output} or {metadata_path}. "
            "Choose a new path or pass --overwrite."
        )

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Video reports invalid dimensions: {width}x{height}")

    processed_frames = 0
    detected_frames = 0
    resized_frames = 0
    frame_index = 0
    mp_pose = mp.solutions.pose

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as pose, args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(build_header())

            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                inference_frame = frame
                if args.max_width and frame.shape[1] > args.max_width:
                    scale = args.max_width / frame.shape[1]
                    target_height = max(1, round(frame.shape[0] * scale))
                    inference_frame = cv2.resize(
                        frame,
                        (args.max_width, target_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    resized_frames += 1

                rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                processed_frames += 1

                if result.pose_landmarks:
                    values = [frame_index]
                    for landmark in result.pose_landmarks.landmark:
                        values.extend([landmark.x, landmark.y])
                    writer.writerow(values)
                    detected_frames += 1

                frame_index += 1
    finally:
        capture.release()

    metadata = {
        "source_video": str(args.video),
        "output_csv": str(args.output),
        "fps": fps,
        "expected_frames": expected_frames,
        "processed_frames": processed_frames,
        "detected_pose_frames": detected_frames,
        "pose_detection_rate": detected_frames / processed_frames if processed_frames else 0.0,
        "width": width,
        "height": height,
        "landmarks_per_frame": 33,
        "coordinates_per_frame": 66,
        "frame_index_preserved": True,
        "max_inference_width": args.max_width or None,
        "resized_frames": resized_frames,
        "model_complexity": args.model_complexity,
        "min_detection_confidence": args.min_detection_confidence,
        "min_tracking_confidence": args.min_tracking_confidence,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
