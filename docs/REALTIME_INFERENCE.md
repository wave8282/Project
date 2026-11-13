# Real-time inference prototype

This is step 7 of the experimental pipeline. It loads the trained classifier and
normal-only autoencoder produced by the manual training workflow.

Install both dependency sets:

```bash
python -m pip install -r requirements-pose.txt
python -m pip install -r requirements-train.txt
```

Run a local video:

```bash
python src/inference/realtime_detect.py \
  --source "path/to/test_video.mp4" \
  --model-dir "artifacts/fall_detector" \
  --display
```

Run a webcam (camera index 0):

```bash
python src/inference/realtime_detect.py --source 0 --display
```

The detector buffers 30 consecutive pose frames, applies the same 137-feature
body-relative transform as training, and combines classifier probability with
normal-only autoencoder reconstruction error. A short smoothing queue and
consecutive-positive rule reduce single-window alerts. Candidate events are
appended to JSONL at `artifacts/fall_events.jsonl`.

The model files are not committed to Git. They are produced as a downloadable
artifact by the `Train fall detector` GitHub Actions workflow after a valid
multi-video dataset is available.

**Safety limitation:** this is a research prototype, not a medical device or
certified safety alarm. It can miss falls or raise false alarms. Do not use it
as the sole monitoring method or as a replacement for human supervision.
Validate latency, sensitivity, specificity, false alarms per hour, occlusion,
multiple people, camera placement and recovery/long-lie behavior before any
real-world pilot.
