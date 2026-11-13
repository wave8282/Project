# Building the pose sequence dataset

This is the next data-preparation step for the LSTM classifier and LSTM
Autoencoder experiments.

## 1. Extract each source video separately

Run from the repository root in an environment with OpenCV and MediaPipe
installed:

```bash
python src/preprocessing/extract_pose.py --video "test.mp4..mp4" --output "data/raw/normal/test_normal_pose.csv"
python src/preprocessing/extract_pose.py --video "fall.mp4" --output "data/raw/fall/fall_pose.csv"
```

The project owner has identified `test.mp4..mp4` as the normal-behavior source.
The legacy `test.py` reads `fall.mp4`, but its contents and suitability as a
fall-labeled source must be visually confirmed before using it as ground truth.

The extractor writes a CSV with 67 columns: the original frame index followed
by 66 normalized x/y landmark coordinates. It also writes a metadata JSON file.
Keep these outputs separate by source class and do not overwrite the legacy
`fall_data.csv` or `pose_data.csv`.

## 2. Build sequences

After both source CSVs have been generated and their labels checked:

```bash
python src/preprocessing/build_sequences.py \
  --normal "data/raw/normal/test_normal_pose.csv" \
  --fall "data/raw/fall/fall_pose.csv" \
  --sequence-length 30 \
  --output "data/processed/pose_sequences.npz"
```

The resulting NPZ contains:

- `X`: float32 sequences shaped (samples, 30, 66)
- `y`: integer labels (0 = normal, 1 = fall)
- `groups`: source-video identifiers for group-based splitting

A companion summary JSON records the number of windows generated per source.

## 3. Important evaluation rule

Do not use a random split over overlapping windows. Windows from one video are
highly correlated. Use `groups` to split by source video before training.

A meaningful video-level train/test evaluation needs multiple independent
source videos in each class. With only one normal video and one fall video, a
held-out evaluation that contains both classes cannot be formed reliably.
Collect or confirm additional independent source videos before reporting
generalization performance.

## 4. Current limitations

- Legacy 66-column CSV files are deliberately rejected by the new builder:
  they do not preserve frame indices, so missed pose detections can silently
  compress time.
- The builder does not infer labels from filenames or model predictions.
- It does not train models. Training starts only after the generated dataset
  and source labels have been reviewed.
- The 66 x/y coordinates are retained for compatibility with the existing
  model input shape. Finite coordinates outside [0, 1] are preserved because
  MediaPipe landmarks can lie beyond the image boundary. Visibility, z coordinates,
  body-relative normalization, and camera-motion robustness remain future improvements.
