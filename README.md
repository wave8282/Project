# AI 기반 예방형 낙상 감지 시스템

## 프로젝트 소개

MediaPipe Pose로 영상 속 인체 관절 좌표를 추출하고, 시계열 딥러닝 모델을 이용해 정상 행동과 낙상 행동을 구분하는 프로젝트입니다. Autoencoder를 활용한 이상 행동 탐지와 LSTM 기반 낙상 분류를 결합하는 구조를 연구합니다.

> 주의: 본 저장소의 현재 코드와 실험 결과는 구분해서 해석해야 합니다. 실제 낙상 전 위험 예측 성능은 별도의 검증이 필요하며, 코드의 구현만으로 예방 효과가 입증되는 것은 아닙니다.

## 처리 흐름

1. 영상 입력
2. MediaPipe Pose를 이용한 관절 좌표 추출
3. 30프레임 단위 시계열 데이터 구성
4. Autoencoder 기반 재구성 오차 계산 및 이상 여부 판단
5. LSTM 기반 낙상 여부 분류
6. 정상 / 위험 행동 / 낙상 결과 출력

## 사용 기술

- Python
- TensorFlow / Keras
- MediaPipe Pose
- OpenCV
- NumPy / Pandas
- Matplotlib
- scikit-learn

## 개발 환경

- Visual Studio Code
- Google Colab

## 실행 전 확인 사항

- 데이터 파일의 실제 라벨이 정상(0)과 낙상(1)에 맞게 구성되어 있는지 확인해야 합니다.
- 현재 일부 스크립트에는 로컬 Windows 경로가 하드코딩되어 있으므로 실행 환경에 맞게 경로를 설정해야 합니다.
- 시계열 데이터는 서로 겹치는 구간이 많을 수 있으므로, 학습/검증/테스트 분리 시 데이터 누수를 방지해야 합니다.
- 모델 성능을 보고할 때는 정확도뿐 아니라 낙상 재현율, 정밀도, F1-score, 오탐 횟수 및 탐지 지연시간을 함께 확인하는 것이 좋습니다.

## 현재 개선 작업

상세 점검 결과와 수정 우선순위는 [파이프라인 점검 보고서](docs/PIPELINE_AUDIT.md)를 참고하세요.

### Pose extraction and dataset validation

- Install extraction dependencies with `python -m pip install -r requirements-pose.txt`.
- Follow [the dataset build guide](docs/DATASET_BUILD.md) to extract separate normal/fall CSVs and build labeled sequences.
- Run sequence-builder tests with `python -m unittest discover -s tests -v`.
- Do not randomly split overlapping windows; use the saved source-video groups.

- Install model dependencies with `python -m pip install -r requirements-train.txt`.
- Train/evaluate the leakage-safe hybrid model with `python src/training/train_hybrid.py` after reviewing the generated NPZ dataset.
- See [verified pose extraction results](docs/EXTRACTION_RESULTS.md) for actual detection rates, continuity checks, and dataset counts.

### External data, engineered features, training and inference

- Public dataset sources and licence notes: [docs/EXTERNAL_DATASETS.md](docs/EXTERNAL_DATASETS.md).
- Convert downloaded URFD cam0 RGB ZIP archives: `python src/preprocessing/prepare_urfd.py --source-dir data/external/urfd/downloads --fps <verified_fps>`.
- Prepare a reviewed multi-video CSV manifest using [data/external/manifest_template.csv](data/external/manifest_template.csv), then run `python src/preprocessing/build_dataset_from_manifest.py --manifest data/external/my_manifest.csv`.
- Generate body-relative temporal features: `python src/preprocessing/engineer_features.py`.
- Train/evaluate by source video with the GitHub Actions workflow **Train fall detector** or `python src/training/train_hybrid.py`. The workflow skips model fitting until at least three independent source videos per class are present.
- Experimental camera/video inference: [docs/REALTIME_INFERENCE.md](docs/REALTIME_INFERENCE.md).
