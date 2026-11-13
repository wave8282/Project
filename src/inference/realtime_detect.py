"""Real-time or file-based fall inference using trained hybrid models.

Requires artifacts produced by src/training/train_hybrid.py on the engineered
137-feature dataset. This tool is an experimental alert prototype, not a
certified medical or safety device.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from src.preprocessing.engineer_features import engineer_sequence


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index (0) or video file path")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/fall_detector"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--smooth-windows", type=int, default=5)
    parser.add_argument("--consecutive", type=int, default=5)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--events", type=Path, default=Path("artifacts/fall_events.jsonl"))
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    classifier_path = args.model_dir / "fall_lstm_classifier.keras"
    autoencoder_path = args.model_dir / "normal_lstm_autoencoder.keras"
    evaluation_path = args.model_dir / "evaluation.json"
    for path in (classifier_path, autoencoder_path, evaluation_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing trained artifact: {path}. Collect independent videos, "
                "build features, and train the model first."
            )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    threshold = float(evaluation["autoencoder_threshold"])
    classifier = tf.keras.models.load_model(classifier_path)
    autoencoder = tf.keras.models.load_model(autoencoder_path)

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    pose_frames = deque(maxlen=args.sequence_length)
    decisions = deque(maxlen=args.smooth_windows)
    consecutive = 0
    last_alert = 0.0
    args.events.parent.mkdir(parents=True, exist_ok=True)
    mp_pose = mp.solutions.pose
    try:
        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                status = "POSE NOT DETECTED"
                if result.pose_landmarks:
                    xy = []
                    for landmark in result.pose_landmarks.landmark:
                        xy.extend([landmark.x, landmark.y])
                    pose_frames.append(np.asarray(xy, dtype=np.float32))
                else:
                    pose_frames.clear()
                    decisions.clear()
                    consecutive = 0

                if len(pose_frames) == args.sequence_length:
                    raw = np.stack(pose_frames)
                    features = engineer_sequence(raw)[None, ...]
                    probability = float(classifier.predict(features, verbose=0)[0, 0])
                    reconstruction = autoencoder.predict(features, verbose=0)
                    error = float(np.mean(np.square(features - reconstruction)))
                    is_fall = probability >= 0.5 and error > threshold
                    decisions.append(bool(is_fall))
                    stable = sum(decisions) >= max(1, int(np.ceil(len(decisions) * 0.6)))
                    consecutive = consecutive + 1 if stable else 0
                    status = (
                        f"FALL CANDIDATE p={probability:.2f} err={error:.4f}"
                        if stable else f"NORMAL/UNCERTAIN p={probability:.2f} err={error:.4f}"
                    )
                    now = time.time()
                    if consecutive >= args.consecutive and now - last_alert >= args.cooldown_seconds:
                        event = {
                            "timestamp_unix": now,
                            "source": str(args.source),
                            "classifier_probability": probability,
                            "reconstruction_error": error,
                            "threshold": threshold,
                            "event": "fall_candidate",
                        }
                        with args.events.open("a", encoding="utf-8") as file:
                            file.write(json.dumps(event, ensure_ascii=False) + "\n")
                        print("ALERT: FALL CANDIDATE " + json.dumps(event, ensure_ascii=False))
                        last_alert = now
                        consecutive = 0

                if args.display:
                    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255) if "FALL" in status else (0, 255, 0), 2)
                    cv2.imshow("Fall detection prototype - not a safety device", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
