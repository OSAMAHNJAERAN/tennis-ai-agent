"""Measure continuous tiled ball inference with frozen temporal filters."""

import argparse
from collections import deque
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.detection.tiled_wasb_candidates import predict_tiled_stream
from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('Validation manifest required')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset checksum changed')
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    report = {'schema_version': '1.0', 'complete': False, 'qualification_evidence': False,
              'scope': 'CONTINUOUS_BALL_ONLY_FIVE_VIEW_INFERENCE_ON_REUSED_VALIDATION_CLIPS',
              'dataset_manifest_sha256': digest(manifest_path), 'checkpoint_sha256': digest(checkpoint),
              'configuration': {'crop_fraction': .6, 'heatmap_threshold': .2, 'temporal_step': 1,
                                'merge_radius_reference_px': 4., 'stationary_minimum_seconds': .25,
                                'pixel_motion_threshold': 12., 'pixel_lag_seconds': .1,
                                'pixel_patch_radius': 3, 'ranking': 'PEAK_CONFIDENCE_CROSS_VIEW_UNCALIBRATED'},
              'code_hashes': {name: digest(ROOT / name) for name in (
                  'scripts/evaluate/benchmark_tiled_ball_stream.py', 'src/detection/tiled_wasb_candidates.py',
                  'src/detection/wasb_ball_detector.py', 'src/tracking/stationary_candidate_filter.py',
                  'src/tracking/candidate_pixel_motion.py', 'src/evaluation/point_metrics.py')},
              'timing_scope': 'DECODE_FIVE_VIEW_BALL_INFERENCE_AND_FILTERS_EXCLUDES_MODEL_SETUP_OTHER_COMPONENTS',
              'clips': []}
    model = WASBBallDetector(checkpoint, threshold=.2, device='cuda' if torch.cuda.is_available() else 'cpu')
    rows = {name: [] for name in ('raw', 'stationary', 'pixel_motion')}
    if args.resume_from:
        previous = json.loads(args.resume_from.read_text())
        if previous['complete'] or any(previous[key] != report[key] for key in
                                      ('dataset_manifest_sha256', 'checkpoint_sha256', 'configuration')):
            raise ValueError('Resume requires identical data/model/settings and incomplete report')
        for name, checksum in previous['code_hashes'].items():
            if name != 'scripts/evaluate/benchmark_tiled_ball_stream.py' and checksum != report['code_hashes'][name]:
                raise ValueError('Inference/filter/metric implementation changed since partial run')
        expected = [f'{match}_{rally}' for match, rally in manifest['selected_clips']]
        if [clip['clip'] for clip in previous['clips']] != expected[:len(previous['clips'])]:
            raise ValueError('Partial report must contain a consecutive prefix')
        report['clips'] = previous['clips']
        for clip in report['clips']:
            if any(len(values) != clip['frames'] for values in clip['predictions'].values()):
                raise ValueError('Partial clip prediction count is invalid')
            for name in rows:
                rows[name].extend(clip['labeled_rows'][name])
        report['resume_provenance'] = {'source_report': str(args.resume_from.resolve()),
                                      'source_sha256': digest(args.resume_from),
                                      'inherited_code_hashes': previous['code_hashes'],
                                      'inherited_clips': len(report['clips'])}
        report['timing_scope'] += '; RESUMED_SUM_OF_COMPLETED_CLIP_SECONDS_NOT_UNINTERRUPTED_THROUGHPUT'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for match, rally in manifest['selected_clips'][len(report['clips']):]:
        clip_name = f'{match}_{rally}'
        with (args.dataset / f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as stream:
            labels = list(csv.DictReader(stream))
        clip_started = time.perf_counter()
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip_name}.mp4')) as frames:
            size = (frames.metadata.width, frames.metadata.height)
            fps = frames.metadata.fps
            queue = deque()
            def source():
                for frame in frames:
                    queue.append(frame)
                    yield frame
            stationary = StationaryCandidateFilter(minimum_seconds=.25)
            pixels = CandidatePixelMotion()
            predictions = {name: [] for name in rows}
            for index, candidates in enumerate(predict_tiled_stream(model, source(), size)):
                current = queue.popleft()
                filtered, _ = stationary.filter(candidates, index / fps, fps, size)
                moving, _ = pixels.filter(current, filtered, index / fps, minimum_score=12.)
                for name, values in zip(predictions, (candidates, filtered, moving)):
                    predictions[name].append([values[0].x_px, values[0].y_px] if values else None)
            if queue or any(len(values) != len(frames) for values in predictions.values()):
                raise ValueError('Decoded and predicted frames are misaligned')
            clip_rows = {name: [] for name in rows}
            seen = set()
            for label in labels:
                index, visible = int(label['Frame']), int(label['Visibility'])
                if index in seen or not 0 <= index < len(frames) or visible not in (0, 1):
                    raise ValueError('Invalid or duplicate label')
                seen.add(index)
                target = [float(label['X']) * size[0] / 1920, float(label['Y']) * size[1] / 1080] if visible else None
                for name in rows:
                    clip_rows[name].append({'clip': clip_name, 'frame': index, 'width': size[0], 'height': size[1],
                                            'target_xy': target, 'prediction_xy': predictions[name][index]})
            for name in rows:
                rows[name].extend(clip_rows[name])
            report['clips'].append({'clip': clip_name, 'frames': len(frames), 'fps': fps, 'size': list(size),
                                     'seconds': time.perf_counter() - clip_started,
                                     'labeled_rows': clip_rows, 'predictions': predictions})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'clip': clip_name, 'seconds': report['clips'][-1]['seconds']}), flush=True)
    seconds = sum(clip['seconds'] for clip in report['clips']) if args.resume_from else time.perf_counter() - started
    report.update({'complete': True, 'seconds': seconds,
                   'results': {name: summarize(values) for name, values in rows.items()}})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, result in report['results'].items():
        print(json.dumps({'variant': name, **{key: value for key, value in result['pooled'].items()
                                              if key != 'localization_errors_reference_px'}}), flush=True)


if __name__ == '__main__':
    main()
