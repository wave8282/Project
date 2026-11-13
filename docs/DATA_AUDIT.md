# Dataset Audit — Initial Findings

## Scope

This audit uses the repository files on branch `audit/fall-pipeline-review` and the user's confirmation that `test.mp4..mp4` is the normal-behavior video.

The video itself has not yet been decoded in this audit environment. Therefore, its visual content, duration, frame rate, and frame count are not independently verified here.

## Confirmed repository facts

- `fall.mp4`: 11,326,329 bytes.
- `test.mp4`: 13,039,888 bytes.
- `test.mp4..mp4`: 12,954,572 bytes; user identifies this as the normal-behavior video.
- `fall_data.csv` and `pose_data.csv` have the same Git blob SHA: `90ac5f9c7a3ff963c53b4c2dec46d21b988ed329`.
- Both CSV files are 724,982 bytes.
- `test.py` reads `D:/pose_project/fall.mp4` and writes extracted x/y landmark coordinates to `D:/pose_project/fall_data.csv`.
- The inspected repository scripts do not show a pipeline that reads `test.mp4..mp4` and writes `pose_data.csv`.

## CSV structural diagnostics

The identical CSV content was parsed:

| Check | Result |
|---|---:|
| Rows | 573 |
| Values per row | 66 on all rows |
| Non-numeric values | 0 |
| Non-finite values | 0 |
| Values outside [0, 1] | 0 |
| Duplicate rows within file | 0 |
| Minimum coordinate value | 0.1064256 |
| Maximum coordinate value | 0.7682025 |

The 66 values per row are consistent with 33 MediaPipe Pose landmarks represented by x/y coordinates. This is a structural consistency check, not proof of which source video generated the CSV.

## Key interpretation

The CSVs are byte-identical, so they are not independent normal and fall datasets. The current code indicates that `fall_data.csv` is generated from `fall.mp4`. The provenance of `pose_data.csv` is not established.

Do not label `pose_data.csv` as normal data and do not use it as a separate normal class until its source video is confirmed.

## Next actions

1. Preserve both original CSV files unchanged.
2. Extract pose coordinates from the confirmed normal video `test.mp4..mp4` into a new file, e.g. `data/raw/normal/test_normal_pose.csv`.
3. Extract fall pose coordinates from the fall video into a separate new file, e.g. `data/raw/fall/fall_pose.csv`, after confirming the intended fall source.
4. Record source video, extraction script version, frame count, coordinate count, and label in a manifest.
5. Build sequences by video, then split train/validation/test by source video to avoid overlapping-window leakage.

## Reproducibility

Run the diagnostic utility:

```bash
python src/preprocessing/diagnose_csv.py fall_data.csv pose_data.csv
```
