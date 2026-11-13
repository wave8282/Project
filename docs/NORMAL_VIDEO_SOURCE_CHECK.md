# Normal Video Source Check

## Source file

- Uploaded filename: `test.mp4..mp4`
- Intended role: normal-behavior source video (as identified by the project owner)
- Container readability: passed with OpenCV 4.13.0
- Video dimensions: 3840 x 2160
- Frame rate: 59.94005994 FPS
- Declared frame count: 600
- Successfully decoded frames: 600
- Duration calculated from frame count / FPS: 10.010 seconds

## Verification boundary

This check confirms that the uploaded video can be opened and all 600 frames can be decoded. It does **not** establish that the recorded activity is normal behavior; that requires visual/behavioral review and/or validated labels.

Pose extraction was not run in this environment because the Python environment does not have the `mediapipe` package installed. Therefore no landmark CSV has been generated from this video in this run.

## Safe next step

Run the repository's `src/preprocessing/extract_pose.py` in an environment with its declared dependencies, writing to a new path such as:

```bash
python src/preprocessing/extract_pose.py --video "test.mp4..mp4" --output "data/raw/normal/test_normal_pose.csv"
```

Keep the original `fall_data.csv` and `pose_data.csv` unchanged. Do not treat the resulting file as a validated normal dataset until its source, labels, and extracted frames have been reviewed.
