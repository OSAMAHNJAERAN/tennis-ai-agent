"""Measure a generic racket detector on publisher validation annotations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch
from ultralytics import YOLO

from scripts.evaluate.benchmark_ball_detection import sha256
from src.evaluation.detection_metrics import aggregate_counts, match_detections
from src.utils.video_frame_sequence import VideoFrameSequence
from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.detection.player_detector import PlayerDetector
from src.tracking.court_player_tracker import CourtPlayerTracker
from src.tracking.racket_tracking import RacketTracking
from src.tracking.global_racket_tracking import GlobalRacketTracking


def main():
    evaluator_hash = sha256(Path(__file__))
    adapter_hash = sha256(ROOT / 'src/tracking/racket_tracking.py')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/external/racketvision_validation"))
    parser.add_argument("--model", type=Path, default=Path("yolo11m.pt"))
    parser.add_argument("--racket-model", type=Path, help="Separate racket-only candidate; person detector remains --model")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--mode", choices=['full_frame', 'player_crops'], default='full_frame')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--association', choices=['independent', 'global'], default='independent')
    args = parser.parse_args()
    if args.association == 'global' and args.mode != 'player_crops':
        raise ValueError('Global association requires player crops')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('Racket benchmark requires a validation manifest')
    for item in manifest['files']:
        if sha256(args.dataset / item['path']) != item['sha256']:
            raise ValueError(f"Dataset checksum changed: {item['path']}")
    selected = {tuple(item) for item in manifest["selected_clips"]}
    coco_path = args.dataset / "tennis/info/val_coco.json"
    coco = json.loads(coco_path.read_text())
    racket_id = next(item["id"] for item in coco["categories"] if item["name"] == "tennis_racket")
    annotations = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["category_id"] == racket_id:
            annotations[ann["image_id"]].append(ann["bbox"])
    images = defaultdict(list)
    for item in coco["images"]:
        parts = item["file_name"].split("/")
        # This mixed ball/racket COCO export includes ball-only frames. Missing
        # racket annotations are unlabeled, not verified racket-absent negatives.
        if (parts[0], parts[2]) in selected and annotations[item["id"]]:
            images[(parts[0], parts[2])].append(item)
    if not images:
        raise ValueError('No explicitly annotated racket-positive frames in the selected clips')
    player_detector = PlayerDetector(str(args.model)) if args.mode == 'player_crops' else None
    model = player_detector.model if player_detector else YOLO(str(args.model))
    court_detector = CourtKeypointDetector('models/keypoints_model.pth') if player_detector else None
    if model.names[38] != "tennis racket":
        raise ValueError("Model does not use expected COCO tennis racket class")
    racket_path = args.racket_model or args.model
    if not racket_path.is_file():
        raise ValueError("Racket checkpoint does not exist")
    racket_model = YOLO(str(racket_path)) if args.racket_model else model
    if racket_model.names[38] != "tennis racket":
        raise ValueError("Racket candidate must preserve COCO class 38")
    device = 0 if torch.cuda.is_available() else "cpu"
    reports, counts, seconds, player_seconds = [], [], 0., 0.
    calibration_reports = []
    for (match, rally), labeled_images in images.items():
        with VideoFrameSequence(str(args.dataset / f"tennis/videos/{match}_{rally}.mp4")) as frames:
            players = None
            if player_detector:
                start = time.perf_counter()
                calibration = calibrate_court(court_detector.predict(frames[0]),
                                              TennisCourtGeometry.get_canonical_keypoints())
                # Reset ByteTrack at each independent clip; process all frames for selection.
                for tracker in getattr(model.predictor, 'trackers', []):
                    tracker.reset()
                detections = [player_detector.detect_and_track(frame, persist=True) for frame in frames]
                players = CourtPlayerTracker(calibration).select(detections, frames.metadata.fps)
                player_seconds += time.perf_counter() - start
                calibration_reports.append({'clip': f'{match}_{rally}', **calibration.to_dict()})
            for item in sorted(labeled_images, key=lambda image: int(Path(image['file_name']).stem)):
                index = int(Path(item["file_name"]).stem)
                frame = frames[index]
                h, w = frame.shape[:2]
                targets = []
                for x, y, bw, bh in annotations[item["id"]]:
                    targets.append([x * w / item["width"], y * h / item["height"],
                                    (x + bw) * w / item["width"], (y + bh) * h / item["height"]])
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                if players is None:
                    result = racket_model.predict(frame, imgsz=args.imgsz, classes=[38], conf=.25,
                                           iou=.7, device=device, verbose=False)[0]
                    boxes = result.boxes.xyxy.cpu().numpy().tolist()
                    confidences = result.boxes.conf.cpu().numpy().tolist()
                else:
                    # Score crop detection on sparse labels. Reset racket memory rather
                    # than pretending widely separated annotations are adjacent frames.
                    racket = (GlobalRacketTracking if args.association == 'global' else RacketTracking)(model=racket_model, device=device, confidence=.25, imgsz=args.imgsz)
                    observations = racket.observe(frame, {identity: track[index] for identity, track in players.items()},
                                                  index / frames.metadata.fps)
                    present = [value for value in observations.values() if value['bbox_xyxy'] is not None]
                    boxes = [value['bbox_xyxy'] for value in present]
                    confidences = [value['confidence'] for value in present]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                seconds += time.perf_counter() - start
                count = match_detections(boxes, targets, iou_threshold=.5)
                counts.append(count)
                reports.append({"image": item["file_name"], "targets": targets, "predictions": boxes,
                                "confidences": confidences, "counts": count.to_dict()})
                if players is not None and args.association == 'global':
                    reports[-1]['raw_candidates'] = [{**r, 'sources': sorted(r['sources'])} for r in racket.last_candidates]
                    reports[-1]['observations'] = observations
                    reports[-1]['player_boxes'] = {str(i): [float(v) for v in (track[index].x1, track[index].y1, track[index].x2, track[index].y2)] if track[index] is not None else None for i, track in players.items()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".partial.json").write_text(json.dumps({"complete": False, "per_image": reports}, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Completed {match}_{rally}: {len(labeled_images)} annotated images", flush=True)
    output = {"schema_version": "1.0", "complete": True, "qualification_evidence": False,
              "model": str(args.model), "model_sha256": sha256(args.model), "class_id": 38,
              "racket_model": str(racket_path), "racket_model_sha256": sha256(racket_path),
              "imgsz": args.imgsz, "confidence": .25, "match_iou": .5,
              "mode": args.mode, "association": args.association, "evaluator_sha256": evaluator_hash,
              "global_adapter_sha256": sha256(ROOT / "src/tracking/global_racket_tracking.py") if args.association == "global" else None,
              "racket_adapter_sha256": adapter_hash, "code_hash_capture_scope": "BEFORE_INFERENCE",
              "player_and_court_seconds": player_seconds, "calibration_reports": calibration_reports,
              "crop_scope": ('SPARSE_FRAME_DETECTION_WITH_FULL_VIDEO_PLAYER_SELECTION; RACKET_MEMORY_RESET_PER_LABEL'
                             if args.mode == 'player_crops' else None),
              "dataset_revision": manifest["revision"], "annotation_sha256": sha256(coco_path),
              "split": "SIX_CLIP_PUBLISHER_VALIDATION_SUBSET", "images": len(reports),
              "annotation_scope": "RACKET_POSITIVE_FRAMES_ONLY; ABSENT_RACKET_PERFORMANCE_UNMEASURED",
              "negative_images": sum(not item["targets"] for item in reports),
              "metrics": aggregate_counts(counts).to_dict(), "prediction_seconds": seconds,
              "prediction_fps": len(reports) / seconds,
              "timing_scope": "Prediction only including first-use overhead; excludes decode/model construction",
              "per_image": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "per_image"}), flush=True)


if __name__ == "__main__":
    main()
