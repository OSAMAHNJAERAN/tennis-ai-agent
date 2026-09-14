"""Known camera-motion challenge using a frozen real frame and explicit scene cut.

Ground truth is the injected image warp, not manually labeled court geometry.
This tests camera-motion compensation and abstention, not full CV accuracy.
"""

import argparse
import json
from pathlib import Path
import sys

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
    parser.add_argument('--cut-video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    detections = json.loads((args.run / 'detections.json').read_text())
    validation = json.loads((args.run / 'run_validation.json').read_text())
    geometry = json.loads((args.run / 'court_geometry.json').read_text())
    video = detections['metadata']['video']
    if not geometry['is_valid'] or digest(video) != validation['input_sha256']:
        raise ValueError('Require a matching source video and valid anchor calibration')
    with VideoFrameSequence(video) as frames:
        anchor = frames[0].copy()
    with VideoFrameSequence(str(args.cut_video)) as frames:
        cut = cv2.resize(frames[0], (anchor.shape[1], anchor.shape[0]))
    initial = CourtCalibration(np.asarray(geometry['homography_matrix']))
    points = cv2.perspectiveTransform(TennisCourtGeometry.get_canonical_keypoints()[None].astype(np.float32),
                                      np.linalg.inv(initial.image_to_court))[0]
    people = [BBox(*detections['frames'][0][f'player_{n}']['bbox']) for n in (1, 2)
              if detections['frames'][0][f'player_{n}']]
    feet = np.asarray([[(box.x1 + box.x2) / 2, box.y2] for box in people], np.float32)
    if not len(feet):
        raise ValueError('Challenge requires observed anchor player boxes')
    expected_ground = [initial.project_ground_point(point) for point in feet]
    tracker = CourtCameraRegistration(initial, points)
    rows, static_errors, corrected_errors, landmark_errors = [], [], [], []
    args.output.mkdir(parents=True)
    for index in range(105):
        # Frames 0–89 pan/zoom a frozen scene; frames 90–99 show another real
        # broadcast; 100–104 return to the anchor but must remain invalid.
        zoom = 1 + .0004 * index
        transform = np.array([[zoom, 0, index * .8], [0, zoom, index * -.25], [0, 0, 1.]], np.float64)
        if index < 90:
            frame = cv2.warpPerspective(anchor, transform, (anchor.shape[1], anchor.shape[0]))
            moved_people = []
            for box in people:
                corners = cv2.perspectiveTransform(np.array([[[box.x1, box.y1], [box.x2, box.y2]]], np.float32), transform)[0]
                moved_people.append(BBox(*corners.ravel()))
        else:
            frame = cut if index < 100 else anchor
            moved_people = []
        result = tracker.update(frame, moved_people)
        row = {'frame': index, 'expected_registration_valid': index < 90, **result.audit}
        if index < 90:
            moved_feet = cv2.perspectiveTransform(feet[None], transform)[0]
            for expected, foot in zip(expected_ground, moved_feet):
                static_errors.append(float(np.linalg.norm(np.asarray(initial.project_ground_point(foot)) - expected)))
                corrected = result.calibration.project_ground_point(foot)
                if corrected is not None:
                    corrected_errors.append(float(np.linalg.norm(np.asarray(corrected) - expected)))
            if result.anchor_to_frame_px is not None:
                expected = cv2.perspectiveTransform(points[None], transform)[0]
                actual = cv2.perspectiveTransform(points[None], result.anchor_to_frame_px)[0]
                landmark_errors.extend(np.linalg.norm(expected - actual, axis=1).tolist())
        rows.append(row)
        if index in (0, 45, 89, 90, 100):
            cv2.putText(frame, f'INJECTED CAMERA CHALLENGE {index} / VALID {result.audit["is_valid"]}',
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
            if not cv2.imwrite(str(args.output / f'frame_{index:05}.jpg'), frame):
                raise RuntimeError('Challenge screenshot write failed')
    report = {'schema_version': '1.0', 'qualification_evidence': False, 'source_run': str(args.run),
              'source_sha256': digest(video), 'cut_video_sha256': digest(args.cut_video),
              'registration_code_sha256': digest(ROOT / 'src/court/camera_registration.py'),
              'ground_truth_scope': 'KNOWN_INJECTED_WARP_OF_ONE_FROZEN_REAL_FRAME; NOT_INDEPENDENT_COURT_LABELS',
              'valid_motion_frames': sum(row['is_valid'] for row in rows[:90]), 'expected_motion_frames': 90,
              'incorrectly_valid_frames_after_cut': sum(row['is_valid'] for row in rows[90:]),
              'max_static_player_position_error_m': max(static_errors),
              'max_registered_player_position_error_m': max(corrected_errors, default=None),
              'mean_registered_player_position_error_m': float(np.mean(corrected_errors)) if corrected_errors else None,
              'max_landmark_warp_error_source_px': max(landmark_errors, default=None), 'frames': rows}
    (args.output / 'camera_perturbation_benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'frames'}), flush=True)


if __name__ == '__main__':
    main()
