"""Compare frozen full-frame and crop proposals on explicit sparse labels only."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.detection.wasb_ball_detector import WASBBallDetector
from src.detection.tiled_wasb_candidates import sparse_tile_candidates, merge_by_confidence
from src.evaluation.point_metrics import evaluate_points
from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    baseline = json.loads(args.baseline.read_text())
    checkpoint = ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar'
    if (manifest['split'] != 'VALIDATION_ONLY' or digest(manifest_path) != baseline['dataset_manifest_sha256']
            or baseline['checkpoint_sha256'] != digest(checkpoint) or baseline['wasb_heatmap_threshold'] != .2
            or baseline['wasb_temporal_step'] != 1 or baseline.get('ensemble_checkpoint_sha256') is not None):
        raise ValueError('Baseline does not match the frozen experiment')
    if [clip['clip'] for clip in baseline['clips']] != [f'{m}_{r}' for m, r in manifest['selected_clips']]:
        raise ValueError('Baseline clip order differs from manifest')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset file checksum changed')
    report = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'SPARSE_LABEL_FRAME_CROP_PROPOSAL_EXPERIMENT; NO_CONTINUOUS_TRACKING_OR_FILTER_REPLAY',
              'baseline_sha256': digest(args.baseline), 'dataset_manifest_sha256': digest(manifest_path),
              'checkpoint_sha256': digest(checkpoint),
              'code_hashes': {path: digest(ROOT / path) for path in (
                  'scripts/evaluate/benchmark_tiled_ball_candidates.py', 'src/detection/tiled_wasb_candidates.py',
                  'src/detection/wasb_ball_detector.py', 'src/evaluation/point_metrics.py')},
              'configuration': {'crop_fraction': .6, 'tile_count': 4, 'heatmap_threshold': .2,
                                'merge_radius_reference_px': 4., 'ranking': 'PEAK_CONFIDENCE_CROSS_CROP_UNCALIBRATED'},
              'oracle_scope': 'GROUND_TRUTH_ASSISTED_CANDIDATE_COVERAGE_IS_NOT_DEPLOYABLE_RECALL', 'clips': []}
    model = WASBBallDetector(checkpoint, threshold=.2, device='cuda' if torch.cuda.is_available() else 'cpu')
    all_rows = {key: [] for key in ('baseline_mass_top1', 'baseline_peak_top1', 'tile_peak_top1', 'combined_peak_top1')}
    count_visible, baseline_hits, tile_hits, combined_hits = 0, 0, 0, 0
    started = time.perf_counter()
    for clip in baseline['clips']:
        clip_rows, seen = [], set()
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip["clip"]}.mp4')) as frames:
            width, height = frames.metadata.width, frames.metadata.height
            if [width, height] != clip['size'] or frames.metadata.fps != clip['fps'] or len(frames) != clip['decoded_frames']:
                raise ValueError('Baseline source frame geometry/timing mismatch')
            for row in clip['raw_labeled_rows']:
                index = row['frame']
                if index in seen or not 0 <= index < len(frames):
                    raise ValueError('Duplicate or out-of-range label')
                seen.add(index)
                full = [BallObservation(item['x'], item['y'], item['confidence']) for item in clip['predictions'][index]]
                tiled = sparse_tile_candidates(model, frames, index)
                tiles = [item['observation'] for item in tiled]
                fused = merge_by_confidence(full + tiles, (width, height))
                tile_ranked = merge_by_confidence(tiles, (width, height))
                choices = {'baseline_mass_top1': full[0] if full else None,
                           'baseline_peak_top1': max(full, key=lambda item: item.confidence, default=None),
                           'tile_peak_top1': tile_ranked[0] if tile_ranked else None,
                           'combined_peak_top1': fused[0] if fused else None}
                for name, candidate in choices.items():
                    all_rows[name].append({**row, 'prediction_xy': [candidate.x_px, candidate.y_px] if candidate else None})
                def correct(values):
                    if row['target_xy'] is None:
                        return False
                    x, y = row['target_xy']
                    return any(math.hypot((item.x_px - x) * 512 / width, (item.y_px - y) * 288 / height) <= 4 for item in values)
                if row['target_xy'] is not None:
                    count_visible += 1
                    baseline_hits += correct(full)
                    tile_hits += correct(tiles)
                    combined_hits += correct(full + tiles)
                clip_rows.append({**row, 'full_candidate_count': len(full),
                                  'full_has_correct_candidate': correct(full), 'tiles_have_correct_candidate': correct(tiles),
                                  'tile_candidates': [{'x': item['observation'].x_px, 'y': item['observation'].y_px,
                                                       'confidence': item['observation'].confidence,
                                                       'tile_index': item['tile_index'], 'tile_box': item['tile_box'],
                                                       'tile_mass_rank': item['tile_mass_rank']} for item in tiled]})
        report['clips'].append({'clip': clip['clip'], 'labeled_rows': clip_rows})
        print(json.dumps({'clip': clip['clip'], 'labels': len(clip_rows), 'tiles_with_correct_ball': sum(row['tiles_have_correct_candidate'] for row in clip_rows)}), flush=True)
    report.update({'metrics': {name: evaluate_points(rows) for name, rows in all_rows.items()},
                   'selected_labeled_rows': all_rows, 'oracle_candidate_coverage': {
                       'visible_labels': count_visible, 'full_frame_hits': baseline_hits, 'tile_hits': tile_hits,
                       'union_hits_before_merging': combined_hits,
                       'union_recall_ceiling': combined_hits / count_visible if count_visible else None},
                   'seconds_including_sparse_decode_and_crop_inference': time.perf_counter() - started})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    for name, metrics in report['metrics'].items():
        print(json.dumps({'candidate': name, **{key: value for key, value in metrics.items() if key != 'localization_errors_reference_px'}}))
    print(json.dumps(report['oracle_candidate_coverage']))


if __name__ == '__main__':
    main()
