"""Prepare real labeled candidate crops; no unlabeled-frame negatives or GT proposals."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.detection.wasb_ball_detector import WASBBallDetector
from src.detection.ball_candidate_verifier import (labeled_frame_candidates, reference_rgb,
                                                  candidate_patch, candidate_features)
from src.utils.video_frame_sequence import VideoFrameSequence
from src.tracking.temporal_ball_tracker import BallObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/ball/wasb_tennis_best.pth.tar'))
    parser.add_argument('--candidate-report', type=Path, help='Reuse an exact model/dataset-matched full-frame candidate report')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    source = json.loads(manifest_path.read_text())
    if source['split'] not in ('TRAINING_ONLY', 'VALIDATION_ONLY'):
        raise ValueError('Require a designated training or validation manifest')
    for item in source['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset file checksum changed')
    provenance = {'schema_version': '1.0', 'qualification_evidence': False, 'split': source['split'],
                  'dataset': str(args.dataset), 'dataset_manifest_sha256': digest(manifest_path),
                  'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest(args.checkpoint),
                  'preparer_sha256': digest(__file__),
                  'verifier_module_sha256': digest(ROOT / 'src/detection/ball_candidate_verifier.py'),
                  'detector_module_sha256': digest(ROOT / 'src/detection/wasb_ball_detector.py'),
                  'configuration': {'heatmap_threshold': .05, 'overlap': 'ALL_REAL_WINDOWS_FOR_LABELED_FRAME',
                                    'max_candidates_per_frame': 32, 'patch_side': 32,
                                    'reference_size': [960, 540], 'past_lag_seconds': .1,
                                    'channels': 'CURRENT_RGB_THEN_PAST_RGB',
                                    'positive_tolerance_reference_px': 4., 'negative_minimum_reference_px': 8.,
                                    'ambiguous_training_targets': 'DISTANCE_IN_(4,8)_IGNORED_IN_TRAINING_LOSS_ONLY'},
                  'selected_clips': source['selected_clips'],
                  'source_broadcast_independence': source['source_broadcast_independence']}
    report_clips = None
    if args.candidate_report:
        candidate_source = json.loads(args.candidate_report.read_text())
        if (candidate_source['dataset_manifest_sha256'] != provenance['dataset_manifest_sha256']
                or candidate_source['checkpoint_sha256'] != provenance['checkpoint_sha256']
                or candidate_source.get('backend') != 'wasb'
                or candidate_source.get('wasb_heatmap_threshold') != .05
                or candidate_source.get('wasb_temporal_step') != 1
                or candidate_source.get('ensemble_checkpoint_sha256') is not None):
            raise ValueError('Saved candidates do not match the fixed preparation protocol')
        report_clips = {clip['clip']: clip for clip in candidate_source['clips']}
        if len(report_clips) != len(candidate_source['clips']) or set(report_clips) != {f'{m}_{r}' for m, r in source['selected_clips']}:
            raise ValueError('Saved candidate clip set differs from dataset')
        provenance['candidate_report'] = str(args.candidate_report)
        provenance['candidate_report_sha256'] = digest(args.candidate_report)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = WASBBallDetector(args.checkpoint, threshold=.05, device=device) if report_clips is None else None
    patches, features, labels, rows, clips = [], [], [], [], []
    started = time.perf_counter()
    for match, rally in source['selected_clips']:
        clip = f'{match}_{rally}'
        label_path = args.dataset / f'tennis/all/{match}/csv/{rally}_ball.csv'
        with label_path.open(newline='') as stream:
            annotations = list(csv.DictReader(stream))
        count_before = len(patches)
        seen = set()
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip}.mp4')) as frames:
            width, height, fps = frames.metadata.width, frames.metadata.height, frames.metadata.fps
            saved = report_clips[clip] if report_clips is not None else None
            saved_labels = {row['frame']: row for row in saved['raw_labeled_rows']} if saved is not None else None
            if saved is not None and (len(saved['predictions']) != len(frames) or saved['size'] != [width, height]
                                      or saved['fps'] != fps or len(saved_labels) != len(annotations)):
                raise ValueError('Saved candidate geometry, timing or labels differ from source video')
            for item in annotations:
                index, visible = int(item['Frame']), int(item['Visibility'])
                if index in seen or not 0 <= index < len(frames) or visible not in (0, 1):
                    raise ValueError('Invalid or duplicate explicit label')
                seen.add(index)
                target = [float(item['X']) * width / 1920, float(item['Y']) * height / 1080] if visible else None
                if target is not None and not (0 <= target[0] < width and 0 <= target[1] < height):
                    raise ValueError('Visible label is not a finite point inside the frame')
                if saved is None:
                    candidates = labeled_frame_candidates(detector, frames, index)
                else:
                    if index not in saved_labels or saved_labels[index]['target_xy'] != target:
                        raise ValueError('Saved candidate report labels differ from explicit CSV')
                    candidates = [BallObservation(value['x'], value['y'], value['confidence'])
                                  for value in saved['predictions'][index]]
                candidates = candidates[:32]
                current = reference_rgb(frames[index])
                past_index = max(0, index - math.ceil(.1 * fps - 1e-9))
                past = reference_rgb(frames[past_index])
                row_candidates = []
                for rank, candidate in enumerate(candidates):
                    error = (math.hypot((candidate.x_px - target[0]) * 512 / width,
                                        (candidate.y_px - target[1]) * 288 / height) if target else math.inf)
                    label = 1 if error <= 4 else 0 if error >= 8 else -1
                    number = len(patches)
                    patches.append(candidate_patch(current, past, candidate, (width, height)))
                    features.append(candidate_features(candidate, rank))
                    labels.append(label)
                    row_candidates.append({'sample_index': number, 'x': candidate.x_px, 'y': candidate.y_px,
                                           'confidence': candidate.confidence, 'rank': rank,
                                           'training_target': label})
                rows.append({'clip': clip, 'frame': index, 'past_frame': past_index,
                             'width': width, 'height': height, 'target_xy': target, 'candidates': row_candidates})
        clips.append({'clip': clip, 'explicit_labels': len(annotations), 'candidate_count': len(patches) - count_before})
        print(json.dumps(clips[-1]), flush=True)
    if not patches:
        raise ValueError('No candidates available for verifier preparation')
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'candidates.npz', patches=np.stack(patches), features=np.stack(features),
                        targets=np.asarray(labels, dtype=np.int8))
    provenance.update({'rows': rows, 'clips': clips, 'candidate_count': len(patches),
                       'candidate_target_counts': {str(value): labels.count(value) for value in (-1, 0, 1)},
                       'cache_sha256': digest(args.output / 'candidates.npz'),
                       'preparation_seconds_excluding_model_load': time.perf_counter() - started})
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
