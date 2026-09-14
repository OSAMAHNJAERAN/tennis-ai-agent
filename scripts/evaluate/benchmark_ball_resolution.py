"""Controlled source-resolution stress test on identical explicit video labels.

Downsample decoded source frames, not labels or model heatmaps. This measures
controlled information loss, not independently filmed low-resolution matches.
"""

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import torch

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.utils.video_frame_sequence import VideoFrameSequence


def scaled_geometry(width, height, target_height):
    if any(not isinstance(value, int) or value <= 0 for value in (width, height, target_height)):
        raise ValueError('Dimensions must be positive integers')
    if target_height > height:
        raise ValueError('This experiment only permits native size or downsampling')
    return max(1, round(width * target_height / height)), target_height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--heights', type=int, nargs='+', default=[1080, 720, 480])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.heights)) != len(args.heights) or any(value <= 0 for value in args.heights):
        raise ValueError('Use fresh output and distinct positive heights')
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('Resolution benchmark requires validation manifest')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset checksum changed')
    checkpoint = ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar'
    report = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'CONTROLLED_DOWNSAMPLING_OF_SAME_VIDEOS; NOT_INDEPENDENT_SOURCE_RESOLUTION_GENERALIZATION',
              'dataset_manifest_sha256': digest(manifest_path), 'checkpoint_sha256': digest(checkpoint),
              'source_broadcast_independence': manifest['source_broadcast_independence'],
              'code_hashes': {name: digest(ROOT / name) for name in (
                  'scripts/evaluate/benchmark_ball_resolution.py', 'src/detection/wasb_ball_detector.py',
                  'src/tracking/candidate_pixel_motion.py', 'src/tracking/stationary_candidate_filter.py',
                  'src/evaluation/point_metrics.py')},
              'configuration': {'detector_threshold': .2, 'temporal_step': 1, 'stationary_minimum_seconds': .25,
                                'pixel_motion_threshold': 12., 'pixel_lag_seconds': .1, 'pixel_patch_radius': 3,
                                'interpolation': 'OPENCV_INTER_AREA', 'heights': args.heights},
              'timing_scope': 'DECODE_RESIZE_BALL_INFERENCE_FILTERS; EXCLUDES_MODEL_SETUP_OTHER_PIPELINE_COMPONENTS',
              'results': []}
    model = WASBBallDetector(checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu', threshold=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for height in args.heights:
        rows, clips = [], []
        start = time.perf_counter()
        for match, rally in manifest['selected_clips']:
            clip_name = f'{match}_{rally}'
            with (args.dataset / f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as stream:
                labels = list(csv.DictReader(stream))
            with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip_name}.mp4')) as frames:
                fps = frames.metadata.fps
                size = scaled_geometry(frames.metadata.width, frames.metadata.height, height)
                queue = deque()
                def transformed_frames():
                    for frame in frames:
                        transformed = frame if size == (frames.metadata.width, frames.metadata.height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                        queue.append(transformed)
                        yield transformed
                stationary = StationaryCandidateFilter(minimum_seconds=.25)
                pixels = CandidatePixelMotion()
                predictions = []
                for index, candidates in enumerate(model.predict_stream(transformed_frames())):
                    current = queue.popleft()
                    candidates, _ = stationary.filter(candidates, index / fps, fps, size)
                    candidates, _ = pixels.filter(current, candidates, index / fps, minimum_score=12.)
                    selected = candidates[0] if candidates else None
                    predictions.append([selected.x_px, selected.y_px] if selected else None)
                if queue or len(predictions) != len(frames):
                    raise ValueError('Model output and decoded frames are not aligned')
                seen, clip_rows = set(), []
                for label in labels:
                    index, visible = int(label['Frame']), int(label['Visibility'])
                    if index in seen or not 0 <= index < len(frames) or visible not in (0, 1):
                        raise ValueError('Invalid or duplicate label')
                    seen.add(index)
                    target = [float(label['X']) * size[0] / 1920, float(label['Y']) * size[1] / 1080] if visible else None
                    clip_rows.append({'clip': clip_name, 'frame': index, 'width': size[0], 'height': size[1],
                                      'target_xy': target, 'prediction_xy': predictions[index]})
                rows.extend(clip_rows)
                clips.append({'clip': clip_name, 'source_size': [frames.metadata.width, frames.metadata.height],
                              'input_size': list(size), 'fps': fps, 'frames': len(frames), 'labeled_rows': clip_rows})
                print(json.dumps({'height': height, 'clip': clip_name, 'frames': len(frames)}), flush=True)
        summary = summarize(rows)
        report['results'].append({'height': height, 'seconds': time.perf_counter() - start,
                                  'clips': clips, **summary})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'height': height, **{key: value for key, value in summary['pooled'].items()
                                              if key != 'localization_errors_reference_px'}}), flush=True)


if __name__ == '__main__':
    main()
