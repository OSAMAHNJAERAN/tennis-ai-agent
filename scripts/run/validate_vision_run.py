"""Run Phase 6 with measured process memory, model setup and video decode checks."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="configs/phase6_analytics/wasb_validation.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    process = psutil.Process()
    memory = [process.memory_info().rss]
    done = threading.Event()

    def sample_memory():
        while not done.wait(.1):
            memory.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        from src.pipeline.phase6_pipeline import Phase6Pipeline
        pipeline = Phase6Pipeline(args.config)
        construction_s = time.perf_counter() - started
        result = pipeline.run(args.video, str(args.output))
        complete_s = time.perf_counter() - started
    finally:
        memory.append(process.memory_info().rss)
        done.set()
        monitor.join()
    cap = cv2.VideoCapture(str(args.output / "annotated.mp4"))
    if not cap.isOpened():
        raise ValueError("Output video cannot be decoded")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    shots = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in (0, 50, 100, 150):
            path = args.output / f"frame_{count:05}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Screenshot write failed")
            shots.append(str(path))
        count += 1
    cap.release()
    detections = json.loads((args.output / "detections.json").read_text())
    if count != detections["metadata"]["frames"] or fps != detections["metadata"]["fps"]:
        raise ValueError("Output frame count/FPS disagrees with exported detections")
    report = {
        "schema_version": "1.0", "result": result, "input_video": args.video,
        "config": args.config, "construction_and_import_seconds": construction_s,
        "full_run_seconds_including_construction": complete_s,
        "full_run_fps_including_construction": count / complete_s,
        "process_peak_rss_bytes_sampled_100ms": max(memory), "memory_samples": len(memory),
        "output_decoded_frames": count, "output_fps": fps, "screenshots": shots,
        "validation_note": "Execution, alignment and sampled runtime memory only; not CV accuracy",
    }
    with open(args.video, "rb") as stream:
        report["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "run_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "result"}), flush=True)


if __name__ == "__main__":
    main()
