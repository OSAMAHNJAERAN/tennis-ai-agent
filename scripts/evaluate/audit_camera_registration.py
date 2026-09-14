"""Audit court-plane camera registration on a previously validated video run."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.replay_wasb_thresholds import digest
from src.court.calibration import CourtCalibration
from src.court.camera_registration import CourtCameraRegistration
from src.court.court_geometry import TennisCourtGeometry
from src.utils.bbox_utils import BBox
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {name: json.loads((args.run / f'{name}.json').read_text())
              for name in ('court_geometry', 'detections', 'run_validation')}
    video = inputs['detections']['metadata']['video']
    if digest(video) != inputs['run_validation']['input_sha256']:
        raise ValueError('Source video differs from the validated run')
    court = inputs['court_geometry']
    if not court['is_valid']:
        raise ValueError('Cannot audit registration without valid initial calibration')
    initial = CourtCalibration(np.asarray(court['homography_matrix']))
    # Polygon derived from the saved initial fit; this is not independent GT.
    points = cv2.perspectiveTransform(TennisCourtGeometry.get_canonical_keypoints()[None].astype(np.float32),
                                      np.linalg.inv(initial.image_to_court))[0]
    tracker = CourtCameraRegistration(initial, points)
    code_hash = digest(ROOT / 'src/court/camera_registration.py')
    args.output.mkdir(parents=True)
    records, corrections = [], []
    started = time.perf_counter()
    with VideoFrameSequence(video) as frames:
        if len(frames) != len(inputs['detections']['frames']):
            raise ValueError('Detection records do not align with source frame count')
        for index, frame in enumerate(frames):
            detection = inputs['detections']['frames'][index]
            if detection['frame_index'] != index:
                raise ValueError('Detection frame ID mismatch')
            boxes = [BBox(*detection[f'player_{n}']['bbox']) for n in (1, 2) if detection[f'player_{n}']]
            result = tracker.update(frame, boxes)
            records.append(result.audit)
            for box in boxes:
                foot = ((box.x1 + box.x2) / 2, box.y2)
                corrected = result.calibration.project_ground_point(foot)
                original = initial.project_ground_point(foot)
                if corrected is not None:
                    corrections.append(float(np.linalg.norm(np.asarray(corrected) - original)))
            if index in (0, 50, 100, 150, len(frames) - 1):
                display = frame.copy()
                if result.anchor_to_frame_px is not None:
                    transformed = cv2.perspectiveTransform(points[None], result.anchor_to_frame_px)[0]
                    for old, new in zip(points, transformed):
                        cv2.circle(display, tuple(map(round, old)), 5, (40, 40, 255), 1)
                        cv2.circle(display, tuple(map(round, new)), 4, (60, 240, 90), 1)
                label = f"REGISTRATION {result.audit['is_valid']} / SUPPORT {result.audit['inlier_count']}"
                cv2.putText(display, label, (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                if not cv2.imwrite(str(args.output / f'frame_{index:05}.jpg'), display):
                    raise RuntimeError('Audit screenshot write failed')
    elapsed = time.perf_counter() - started
    displacements = [record['max_landmark_displacement_source_px'] for record in records if record['is_valid']]
    report = {'schema_version': '1.0', 'qualification_evidence': False, 'source_run': str(args.run),
              'registration_code_sha256': code_hash, 'input_sha256': digest(video),
              'source_run_hashes': {name: digest(args.run / f'{name}.json') for name in inputs},
              'frames': len(records), 'valid_frames': sum(record['is_valid'] for record in records),
              'first_invalid_frame': next((record['frame_index'] for record in records if not record['is_valid']), None),
              'maximum_landmark_displacement_source_px': max(displacements, default=None),
              'maximum_player_ground_correction_m': max(corrections, default=None),
              'ground_correction_scope': 'CHANGE_RELATIVE_TO_STATIC_FIT; NOT_GROUND_TRUTH_ERROR',
              'seconds_including_decode_and_screenshots': elapsed, 'registration': records}
    (args.output / 'camera_registration_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'registration'}), flush=True)


if __name__ == '__main__':
    main()
