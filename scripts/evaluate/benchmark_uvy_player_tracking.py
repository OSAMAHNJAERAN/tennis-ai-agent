"""Diagnostic player-role tracking on frame-aligned fan recordings.

Publisher annotations have visually confirmed omissions/misclassification.
Raw scores are retained as diagnostics and cannot establish qualification.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from scripts.evaluate.audit_uvy_videos import read_gt
from src.court.calibration import calibrate_court
from src.court.camera_registration import CourtCameraRegistration
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.detection.player_detector import PlayerDetector
from src.evaluation.tracking_metrics import evaluate_sequence
from src.tracking.court_player_tracker import CourtPlayerTracker
from src.tracking.player_tracker import PlayerTracker
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def role_frames(selected, count, source_ids=False):
    output = []
    for frame in range(count):
        rows = []
        for role in (1, 2):
            box = selected[role][frame]
            if box is not None:
                identity = box.track_id if source_ids else role
                if identity is None:
                    raise ValueError('Selected observation has no source identity')
                rows.append({'id': int(identity), 'role': role, 'source_id': box.track_id,
                             'box': [float(box.x1), float(box.y1), float(box.x2), float(box.y2)],
                             'confidence': float(box.confidence)})
        output.append(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/external/uvy_tennis_videos'))
    parser.add_argument('--alignment', type=Path, default=Path('outputs/vision_upgrade_audit/uvy_video_alignment/audit.json'))
    parser.add_argument('--timing', type=Path, default=Path('artifacts/validation/vision_upgrade/uvy_tennis_timing_audit.json'))
    parser.add_argument('--court-checkpoint', type=Path, default=ROOT / 'models/keypoints_model.pth')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    alignment, timing = json.loads(args.alignment.read_text()), json.loads(args.timing.read_text())
    if not manifest['complete'] or not alignment['complete'] or alignment['dataset_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Require completed acquisition and corresponding alignment audit')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Acquired dataset changed')
    for sequence, info in manifest['sequences'].items():
        audit = alignment['sequences'][sequence]
        if (audit['decoded_frames'] != info['publisher_frames'] or audit['size'] != info['size']
                or len(audit['sampled_comparisons']) != 5
                or any(row['mae_bgr'] > 3 for row in audit['sampled_comparisons'])
                or timing[sequence]['timestamp_count'] != info['publisher_frames']
                or not timing[sequence]['strictly_increasing']
                or timing[sequence]['interval_variation_seconds'] > 2e-6):
            raise ValueError('Frame alignment or constant-rate timing is unsupported')
    person_weights, court_weights = ROOT / 'yolo11m.pt', args.court_checkpoint
    source = ROOT / 'artifacts/research/TrackEval'
    if not person_weights.is_file() or not court_weights.is_file():
        raise FileNotFoundError('Existing measured checkpoints required')
    args.output.mkdir(parents=True)
    code_files = [Path(__file__), ROOT / 'src/detection/player_detector.py', ROOT / 'src/tracking/player_tracker.py',
                  ROOT / 'src/tracking/court_player_tracker.py', ROOT / 'src/court/court_keypoint_detector.py',
                  ROOT / 'src/court/camera_registration.py', ROOT / 'src/court/calibration.py',
                  ROOT / 'src/evaluation/tracking_metrics.py', ROOT / 'scripts/evaluate/audit_uvy_videos.py']
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'RAW_PUBLISHER_PLAYER_ROLE_DIAGNOSTIC_ON_THREE_FAN_RECORDED_PRO_MATCHES',
              'label_quality': 'FAILS_QUALIFICATION: visible active players omitted at V01 frame605; chair umpire classed player ID9 at V03 frame141; other errors may exist',
              'target_semantics': 'Publisher class1 player only; selected officials/spectators count as raw player-role FP; all-person detections are not scored as role outputs',
              'timing_semantics': 'Use decoded constant-rate FPS verified against source PTS; publisher nominal30 is not substituted',
              'identity_semantics': 'Role IDs are segment-local near/far labels, not biometric identities; underlying detector IDs scored separately',
              'dataset_manifest_sha256': digest(manifest_path), 'alignment_audit_sha256': digest(args.alignment),
              'timing_audit_sha256': digest(args.timing), 'checkpoint_hashes': {str(p): digest(p) for p in (person_weights, court_weights)},
              'code_hashes': {str(p): digest(p) for p in code_files},
              'player_inference': 'Existing PlayerDetector.detect_and_track; YOLO11m native method defaults and ByteTrack; new detector per sequence',
              'sequences': {}}
    report_path = args.output / 'report.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    court = CourtKeypointDetector(str(court_weights), device=device)
    for sequence, info in manifest['sequences'].items():
        started = time.perf_counter()
        detector = PlayerDetector(str(person_weights), device=device)
        with VideoFrameSequence(str(args.dataset / info['video'])) as frames:
            fps, count = frames.metadata.fps, len(frames)
            keypoints = court.predict(frames[0])
            calibration = calibrate_court(keypoints, TennisCourtGeometry.get_canonical_keypoints())
            registration = CourtCameraRegistration(calibration, keypoints)
            detections, calibrations, camera_audit = [], [], []
            for frame in frames:
                boxes = detector.detect_and_track(frame, persist=True)
                detections.append(boxes)
                result = registration.update(frame, boxes)
                calibrations.append(result.calibration)
                camera_audit.append(result.audit)
            legacy = PlayerTracker.choose_players(detections, keypoints)
            upgraded = CourtPlayerTracker(calibration).select(detections, fps=fps, calibrations=calibrations)
            predictions = {'legacy_roles': role_frames(legacy, count), 'upgraded_roles': role_frames(upgraded, count),
                           'upgraded_source_ids': role_frames(upgraded, count, source_ids=True)}
            ground_truth = [[item for item in row if item['class'] == 1]
                            for row in read_gt(args.dataset / 'UVY' / sequence / 'gt/gt.txt', count)]
            metrics = {name: evaluate_sequence(ground_truth, values, source) for name, values in predictions.items()}
            for index in sorted({0, count // 2, count - 1}):
                image = frames[index].copy()
                for role in predictions['upgraded_roles'][index]:
                    x1, y1, x2, y2 = [round(v) for v in role['box']]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(image, f"{sequence} frame{index+1} court valid={calibrations[index].is_valid}",
                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(args.output / f'{sequence}_{index+1:06}.jpg'), image)
        report['sequences'][sequence] = {'frames': count, 'fps': fps, 'seconds': time.perf_counter() - started,
                                         'initial_calibration': calibration.to_dict(), 'keypoints': keypoints.tolist(),
                                         'registered_frames': sum(c.is_valid for c in calibrations),
                                         'person_detection_count': sum(map(len, detections)),
                                         'predictions': predictions, 'metrics': metrics, 'camera_audit': camera_audit}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'sequence': sequence, 'registered_frames': report['sequences'][sequence]['registered_frames'],
                          'frames': count, 'metrics': {name: result['summary'] for name, result in metrics.items()}}), flush=True)
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
