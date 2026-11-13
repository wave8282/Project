# Pose extraction results

## Source videos

Both source videos were stored in the repository root and processed by the
GitHub Actions workflow on 2026-11-13.

| Metric | Normal source | Fall source |
|---|---:|---:|
| File | `test.mp4..mp4` | `fall.mp4` |
| Original dimensions | 3840 x 2160 | 3840 x 2160 |
| FPS | 59.94 | 59.94 |
| Total frames | 600 | 600 |
| Processed frames | 600 | 600 |
| Detected pose frames | 597 | 577 |
| Pose detection rate | 99.50% | 96.17% |
| Missed pose frames | 3 | 23 |
| 30-frame windows | 509 | 497 |

Inference resized all 600 frames in each video to a maximum width of 1280 pixels.
The extractor preserved original frame indices. CSVs contain 67 columns:
frame_index plus 66 x/y values for 33 landmarks.

## Continuity audit

The sequence builder separates runs at every missing frame and never creates a
30-frame sample across a gap.

- Normal source contiguous detected-frame runs: 328, 8, 22, 239 frames.
- Fall source contiguous detected-frame runs: 4, 29, 2, 10, 2, 1, 2, 1, 526 frames.
- Non-finite coordinate values: 0 in both CSVs.
- Valid 30-frame windows after continuity checks: 509 normal, 497 fall.

## Generated dataset

- File: `data/processed/pose_sequences.npz`
- `X`: shape `(1006, 30, 66)`, float32
- `y`: shape `(1006,)`, int64
- `groups`: source-video identifiers for leakage-safe splitting
- Normal windows (label 0): 509
- Fall windows (label 1): 497

## Important evaluation limitation

This run has only one independent source video per class. Therefore it is
sufficient to validate the extraction and sequence-building pipeline, but it
is not sufficient for an honest source-video-separated train/validation/test
evaluation. The hybrid training script intentionally refuses to train until
at least three independent source videos per class are supplied.

The owner identified the source labels as normal and fall. Automated pose
detection confirms landmark availability, not the semantic correctness of
every frame-level behavior label. Review the extracted skeleton sequences and
add more independent videos with varied people, camera angles, distances,
lighting, and fall types before claiming generalization.
