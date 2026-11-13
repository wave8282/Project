"""Extract MediaPipe Pose x/y landmarks from a video.

The CSV keeps the original frame index so missing detections do not silently
collapse time. Each detected frame contains 33 landmarks (x/y = 66 values).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-complexity", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output.with_suffix(".metadata.json")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {args.video}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    processed_frames = 0
    detected_frames = 0

    header = ["frame_index"]
    for landmark_index in range(33):
        header.extend([f"lm_{landmark_index:02d}_x", f"lm_{landmark_index:02d}_y"])

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        ) as pose, args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        "pose_detection_rate": (
            detected_frames / processed_frames if processed_frames else 0.0
        ),
        "width": width,
        "height": height,
        "landmarks_per_frame": 33,
        "coordinates_per_frame": 66,
        "frame_index_preserved": True,
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
