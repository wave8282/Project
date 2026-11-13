# External fall-video datasets and ingestion

This repository stores scripts and manifests, not third-party raw videos. Download
each dataset from its official source, check the current terms, and keep the
downloaded media outside Git unless redistribution is explicitly permitted.

## Recommended sources

### 1. OmniFall (primary scale-up option)

Official project: https://dsch.ai/omnifall/

OmniFall reports 15,000 videos / about 80 hours across staged, synthetic and
in-the-wild domains. The OF-Synthetic subset contains 12,000 five-second clips
and is listed as CC-BY-NC 4.0. OF-In-the-Wild is explicitly test-only and must
not be used for training or threshold calibration. The source project says
that other groups' videos are fetched from their original host because their
licences do not permit OmniFall to redistribute them. Follow the dataset's
own installation, annotation and licence instructions.

Suggested preparation:

```bash
python -m pip install omnifall
omnifall status
omnifall prepare of-syn
```

The package and prepared dataset are separate from this repository's MediaPipe
pipeline. Create a reviewed CSV manifest pointing at local prepared video clips.
Do not assume that every taxonomy label is a binary fall/normal label. In
particular, lying down is not automatically a fall, and the in-the-wild split
must remain held out.

### 2. UR Fall Detection (URFD; smaller RGB baseline)

Official page: https://fenix.ur.edu.pl/~mkepski/ds/uf.html

The official page describes 70 sequences (30 falls and 40 activities of daily
living). RGB image sequences are supplied as per-sequence ZIP files. The stated
licence is CC BY-NC-SA 4.0 and the page describes non-commercial academic use;
commercial use requires contacting the authors. Cite the original Kwolek and
Kepski paper when using it.

Download the cam0 RGB archives manually from the official page, then convert
them to MP4:

```bash
python src/preprocessing/prepare_urfd.py \
  --source-dir "data/external/urfd/downloads" \
  --output-dir "data/external/urfd/videos" \
  --fps <verified_fps>
```

The FPS argument is required intentionally: use the rate documented for the
downloaded sequence or derive it from its synchronization metadata. Do not
silently assume a frame rate. Camera 0 is the common view available for ADL
and fall sequences.

### 3. Additional sources through OmniFall

OmniFall lists UP-Fall (permission-based), Le2i (CC-BY-NC-SA 3.0),
GMDCSA24 (MIT), CAUCAFall (CC BY 4.0), EDF and OCCU (CC BY 4.0), MCFD
(permission-based), and OOPS-derived in-the-wild data (CC BY-NC-SA 4.0).
Review each source's conditions and original citation. Use the source-provided
subject/view splits where available.

## Build a reviewed manifest

Copy `data/external/manifest_template.csv` and add one row per independent
video. Each row must include the local video path, binary label, unique source
ID, and dataset name. Use only clips whose complete content has been reviewed
and whose label is valid for the whole clip.

```bash
python src/preprocessing/build_dataset_from_manifest.py \
  --manifest "data/external/my_manifest.csv" \
  --output "data/processed/pose_sequences.npz"
```

Then engineer body-relative temporal features:

```bash
python src/preprocessing/engineer_features.py \
  --input "data/processed/pose_sequences.npz" \
  --output "data/processed/pose_features.npz"
```

Train with video-level partitions:

```bash
python src/training/train_hybrid.py \
  --dataset "data/processed/pose_features.npz" \
  --output-dir "artifacts/fall_detector" \
  --epochs 50
```

At least three independent videos per class are required by the current
train/validation/test splitter. More subjects, views, environments and fall
types are strongly preferred. Do not use overlapping windows from one source
video as independent train and test examples.

## Labeling policy

- `normal`: a reviewed clip containing only non-fall activities.
- `fall`: a reviewed clip containing the fall event, with its event timing
  documented. If a video contains long normal lead-in or post-fall periods,
  split it into accurately annotated clips or use frame-level annotations.
- `lying`, `sitting`, `bending`, `kneeling`: do not label as fall merely
  because the person is close to the floor.
- Preserve dataset name, source video ID, subject ID and camera/view ID in the
  manifest. Never place alternate camera views of the same event in different
  partitions.

## Reproducibility and ethics

Do not commit third-party videos, identifiable faces, or private recordings to
this public repository. Keep a local data inventory with source URL, download
date, licence, citation, checksum, subject/view identifiers and annotation
reviewer. Obtain consent and institutional approval where required.
