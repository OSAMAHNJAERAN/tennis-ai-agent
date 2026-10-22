# Camera Pose Recovery

Recovers camera extrinsics (R, T) and intrinsics for each video by annotating court line corners and running PnP calibration. The tennis court's known geometry (standard line spacing) provides the 3D-2D correspondences needed.

## Key classes

- [`CameraCalibrator`](camera_calibration.py) — PnP calibration from annotated court corners
- [`CourtAnnotator`](courtannotator.py) — interactive GUI for annotating court keypoints in a frame
- [`TennisCourtVisualizer`](viz_camera_on_court_fast.py) — visualize estimated camera poses overlaid on a court diagram

## Usage

### Step 1 — Extract a key frame for annotation

```bash
python camera_pose_recovery/extract_key_frames.py --video <path/to/video.MOV>
```

### Step 2 — Annotate and calibrate

```bash
python camera_pose_recovery/run_camera_calibration.py \
    --video <path/to/video.MOV>
```

Optionally pass a PromptHMR (or other model) result to seed the intrinsics estimate:

```bash
python camera_pose_recovery/run_camera_calibration.py \
    --video <path/to/video.MOV> \
    --phmr_results <path/to/results.pkl>
```

Or run as a batch bash script (edit paths inside first):

```bash
bash camera_pose_recovery/get_camera_calib.sh
```

Calibration results are saved under `calib_root` (set in `environments.yaml`).

## Court coordinate system

The origin is placed at the top-left corner of the tennis court. All multi-view pose analyses use this shared coordinate frame.

| Court layout | Example annotation |
| --- | --- |
| ![Court](../media/court.png) | ![Annotations](../media/annotations.png) |
