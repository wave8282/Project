# Building the pose sequence dataset

This is the next data-preparation step for the LSTM classifier and LSTM
Autoencoder experiments.

## 1. Extract each source video separately

Use Python 3.10 from the repository root. Install the dedicated extraction
dependencies:

```bash
python -m pip install -r requirements-pose.txt
```

The project owner has confirmed `test.mp4..mp4` as the normal-behavior video
and `fall.mp4` as an actual fall video. The extractor downsizes 4K frames to
a maximum inference width of 1280 pixels by default to reduce processing cost;
normalized x/y landmark coordinates remain compatible with the 66-feature
model input. It processes every frame and preserves original frame indices.

```bash
python src/preprocessing/extract_pose.py --video "test.mp4..mp4" --output "data/raw/normal/test_normal_pose.csv"
python src/preprocessing/extract_pose.py --video "fall.mp4" --output "data/raw/fall/fall_pose.csv"
```

The extractor refuses to overwrite existing outputs unless `--overwrite` is
explicitly supplied. Each CSV is accompanied by a metadata JSON containing
frame counts, pose detection rate, original dimensions, and inference settings.

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
  model input shape. Visibility, z coordinates, body-relative normalization,
  and camera-motion robustness remain future improvements.

## 5. Train the hybrid model after dataset review

Install the model-training dependencies:

```bash
python -m pip install -r requirements-train.txt
```

Run the group-safe LSTM classifier + normal-only LSTM Autoencoder training:

```bash
python src/training/train_hybrid.py \
  --dataset "data/processed/pose_sequences.npz" \
  --output-dir "artifacts/fall_detector" \
  --epochs 50 \
  --threshold-quantile 0.99
```

The training script creates train/validation/test partitions by source video.
The classifier is trained on labeled training videos. The autoencoder is
trained only on normal training videos, and its anomaly threshold is calibrated
using normal validation videos (not the test set). Evaluation reports accuracy,
precision, recall, F1, and confusion matrices for the classifier and hybrid
decision rule. Models and metrics are saved under the output directory.

**Minimum data requirement:** at least three independent normal source videos
and three independent fall source videos are required. If there are fewer,
training stops rather than reporting misleading metrics from a random
window-level split. This is a safeguard, not a claim that three videos are
enough for a production-grade evaluation; more varied videos are strongly
recommended.

The script has not been run against the project's real videos or real pose
dataset yet. Do not report model performance until extraction, label review,
and training have actually completed.
