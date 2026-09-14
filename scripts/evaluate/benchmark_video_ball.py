"""Compare ball models on sparse, publisher-labeled tennis validation video.

Only explicit CSV rows are scored. Unlabeled video frames are never negatives
and never interpolated into ground truth. Raw top-one localization is reported
separately from the existing tracker's output. Models process every video frame.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml

from src.detection.wasb_ball_detector import WASBBallDetector, WASBEnsembleDetector
from src.detection.yolo11_ball_detector import YOLO11BallDetector
from src.evaluation.point_metrics import evaluate_points
from src.tracking.tracker_factory import build_temporal_ball_tracker


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    # Capture code identity before long inference, so later workspace edits do
    # not silently change the implementation identity reported for this run.
    code_hashes = {name: digest(ROOT / path) for name, path in {
        'evaluator_sha256': 'scripts/evaluate/benchmark_video_ball.py',
        'wasb_adapter_sha256': 'src/detection/wasb_ball_detector.py',
        'yolo_adapter_sha256': 'src/detection/yolo11_ball_detector.py',
        'tracker_sha256': 'src/tracking/temporal_ball_tracker.py',
        'point_metric_sha256': 'src/evaluation/point_metrics.py',
    }.items()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["yolo11", "wasb"], required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/external/racketvision_validation"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase6_analytics/pipeline.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wasb-source", type=Path, default=Path("artifacts/research/WASB-SBDT"))
    parser.add_argument("--wasb-threshold", type=float, default=.5)
    parser.add_argument("--wasb-step", type=int, choices=[1, 3], default=3)
    parser.add_argument("--ensemble-checkpoint", type=Path)
    parser.add_argument("--ensemble-weight", type=float, default=.75)
    parser.add_argument("--cache-labeled-heatmaps", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.ensemble_checkpoint and args.backend != 'wasb':
        raise ValueError('Ensemble requires WASB backend')
    cache_dir = args.output.with_suffix("").with_name(args.output.stem + "_heatmaps")
    if args.cache_labeled_heatmaps:
        if args.backend != 'wasb' or args.wasb_step != 1:
            raise ValueError("Heatmap caching requires WASB step one")
        cache_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('This evaluation runner requires the designated validation manifest')
    for item in manifest["files"]:
        if digest(args.dataset / item["path"]) != item["sha256"]:
            raise ValueError(f"Dataset checksum changed: {item['path']}")
    config = yaml.safe_load(args.config.read_text())
    checkpoint = args.checkpoint or Path(config["ball_detection"]["model"] if args.backend == "yolo11"
                                        else "artifacts/models/ball/wasb_tennis_best.pth.tar")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    if args.backend == "yolo11":
        ball_config = config["ball_detection"]
        model = YOLO11BallDetector(str(checkpoint), imgsz=ball_config["imgsz"],
                                   high_conf=ball_config["high_conf"], low_conf=ball_config["low_conf"],
                                   device=device)
        group_size = 1
    else:
        if args.ensemble_checkpoint:
            model = WASBEnsembleDetector(checkpoint, args.ensemble_checkpoint, args.ensemble_weight,
                                         source=args.wasb_source, device=device, threshold=args.wasb_threshold)
        else:
            model = WASBBallDetector(checkpoint, args.wasb_source, device=device, threshold=args.wasb_threshold)
        group_size = 3
    model_load_s = time.perf_counter() - start
    clips, all_raw, all_tracked = [], [], []
    for match, rally in manifest["selected_clips"]:
        video_path = args.dataset / f"tennis/videos/{match}_{rally}.mp4"
        label_path = args.dataset / f"tennis/all/{match}/csv/{rally}_ball.csv"
        with label_path.open(newline="") as stream:
            labels = list(csv.DictReader(stream))
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or expected_frames <= 0:
            raise ValueError(f"Invalid video: {video_path}")
        candidates, group, times = [], [], []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        loop_start = time.perf_counter()

        def predict(frames):
            if device == "cuda":
                torch.cuda.synchronize()
            before = time.perf_counter()
            result = ([model.extract_candidates(frames[0])] if args.backend == "yolo11"
                      else model.predict_triplet(frames))
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - before)
            return result

        def video_frames():
            while True:
                ret, frame = cap.read()
                if not ret:
                    return
                if frame.shape[:2] != (height, width):
                    raise ValueError("Variable-resolution video requires a new segment")
                yield frame

        if args.backend == 'wasb' and args.wasb_step == 1:
            # This timing includes streaming decode. Report its scope explicitly.
            before = time.perf_counter()
            if args.cache_labeled_heatmaps:
                wanted = {int(label['Frame']) for label in labels}
                heatmap_cache = {}
                for index, (heatmap, shape) in enumerate(model.predict_heatmap_stream(video_frames())):
                    candidates.append(model.decode_heatmaps([heatmap], shape)[0])
                    if index in wanted:
                        heatmap_cache[str(index)] = heatmap
                np.savez_compressed(cache_dir / f'{match}_{rally}.npz', **heatmap_cache)
            else:
                candidates = list(model.predict_stream(video_frames()))
            times.append(time.perf_counter() - before)
        else:
            for frame in video_frames():
                group.append(frame)
                if len(group) == group_size:
                    candidates.extend(predict(group))
                    group = []
            if group:
                candidates.extend(predict(group))
        cap.release()
        if len(candidates) != expected_frames:
            raise ValueError(f'Decoded {len(candidates)} of {expected_frames} declared frames: {video_path}')
        decode_prediction_s = time.perf_counter() - loop_start
        tracker = build_temporal_ball_tracker(config)
        track_start = time.perf_counter()
        tracked = tracker.track_video_candidates(candidates, fps=fps, frame_size=(width, height))
        track_time = time.perf_counter() - track_start
        raw_rows, tracked_rows, seen = [], [], set()
        for label in labels:
            index, visible = int(label["Frame"]), int(label["Visibility"])
            if index in seen or not 0 <= index < len(candidates) or visible not in (0, 1):
                raise ValueError(f"Invalid/duplicate label frame: {label_path}: {index}")
            seen.add(index)
            x, y = float(label["X"]), float(label["Y"])
            target = [x * width / 1920, y * height / 1080] if visible else None
            # WASB is sorted by blob mass; YOLO is explicitly sorted by confidence.
            observations = candidates[index]
            if args.backend == "yolo11":
                observations = sorted(observations, key=lambda candidate: candidate.confidence, reverse=True)
            prediction = [observations[0].x_px, observations[0].y_px] if observations else None
            point = tracked[index]
            common = {"clip": f"{match}_{rally}", "frame": index, "width": width, "height": height,
                      "target_xy": target}
            raw_rows.append({**common, "prediction_xy": prediction})
            tracked_rows.append({**common, "prediction_xy": [point.x_px, point.y_px] if point.x_px is not None else None,
                                 "state": point.state.value})
        all_raw.extend(raw_rows)
        all_tracked.extend(tracked_rows)
        raw_metrics, tracked_metrics = evaluate_points(raw_rows), evaluate_points(tracked_rows)
        report = {"clip": f"{match}_{rally}", "decoded_frames": len(candidates), "fps": fps,
                  "size": [width, height], "annotated_frames": len(labels),
                  "explicit_absent_frames": sum(int(label["Visibility"]) == 0 for label in labels),
                  "raw_top1": raw_metrics, "existing_tracker": tracked_metrics,
                  "prediction_seconds": sum(times), "decode_prediction_seconds": decode_prediction_s,
                  "prediction_fps": len(candidates) / sum(times), "tracker_seconds": track_time,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
                  "raw_labeled_rows": raw_rows, "tracked_labeled_rows": tracked_rows,
                  "predictions": [[{"x": c.x_px, "y": c.y_px, "confidence": c.confidence} for c in row] for row in candidates]}
        clips.append(report)
        print(json.dumps({"clip": report["clip"], "frames": len(candidates), "labels": len(labels),
                          "raw_f1": raw_metrics["f1"], "tracked_f1": tracked_metrics["f1"],
                          "fps": report["prediction_fps"]}), flush=True)
    output = {"schema_version": "1.0", "qualification_evidence": False, "backend": args.backend,
              "dataset_manifest_sha256": digest(args.dataset / "manifest.json"),
              "dataset_revision": manifest["revision"], "split": "PUBLISHER_VALIDATION_SUBSET",
              "source_match_independence": manifest["source_broadcast_independence"],
              "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
              "ensemble_checkpoint": str(args.ensemble_checkpoint) if args.ensemble_checkpoint else None,
              "ensemble_checkpoint_sha256": digest(args.ensemble_checkpoint) if args.ensemble_checkpoint else None,
              "ensemble_second_weight": args.ensemble_weight if args.ensemble_checkpoint else None,
              "wasb_heatmap_threshold": args.wasb_threshold if args.backend == "wasb" else None,
              "wasb_temporal_step": args.wasb_step if args.backend == "wasb" else None,
              "labeled_heatmap_cache": str(cache_dir) if args.cache_labeled_heatmaps else None,
              "config": config, "config_sha256": digest(args.config),
              "code_hash_capture_scope": "BEFORE_INFERENCE", **code_hashes,
              "detector_adapter_sha256": code_hashes['wasb_adapter_sha256' if args.backend == 'wasb' else 'yolo_adapter_sha256'],
              "model_load_seconds": model_load_s, "device": device, "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "timing_scope": ("Streaming decode and overlapping prediction; includes first-use overhead and heatmap cache writes when enabled; excludes model loading/tracking"
                               if args.backend == 'wasb' and args.wasb_step == 1 else
                               "All prediction calls including first-use overhead; excludes decoding/model loading/tracking unless named"),
              "metric_scope": "Single selected ball; wrong location counts FP+FN; only explicit sparse labels; no GT interpolation",
              "wasb_source_commit": subprocess.check_output(["git", "-C", str(args.wasb_source), "rev-parse", "HEAD"], text=True).strip() if args.backend == "wasb" else None,
              "raw_top1": {str(t): evaluate_points(all_raw, t) for t in (2, 4, 8)},
              "existing_tracker": {str(t): evaluate_points(all_tracked, t) for t in (2, 4, 8)}, "clips": clips}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
