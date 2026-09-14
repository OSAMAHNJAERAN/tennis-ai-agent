"""Score a complete Phase 6 run against explicit publisher ball labels."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.replay_wasb_thresholds import digest
from src.evaluation.point_metrics import evaluate_points
from src.utils.video_io import get_video_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.dataset / 'manifest.json').read_text())
    selected = {f'{match}_{rally}': (match, rally) for match, rally in manifest['selected_clips']}
    if manifest['split'] != 'VALIDATION_ONLY' or args.clip not in selected:
        raise ValueError('Choose a clip in the explicit validation manifest')
    match, rally = selected[args.clip]
    video = args.dataset / f'tennis/videos/{args.clip}.mp4'
    label = args.dataset / f'tennis/all/{match}/csv/{rally}_ball.csv'
    for path in (video, label):
        expected = next(item['sha256'] for item in manifest['files']
                        if item['path'] == path.relative_to(args.dataset).as_posix())
        if digest(path) != expected:
            raise ValueError('Dataset evidence changed')
    validation = json.loads((args.run / 'run_validation.json').read_text())
    if validation['input_sha256'] != digest(video):
        raise ValueError('Run video differs from the annotated dataset clip')
    detections = json.loads((args.run / 'detections.json').read_text())
    metadata = get_video_metadata(str(video))
    if len(detections['frames']) != metadata.frame_count or detections['metadata']['fps'] != metadata.fps:
        raise ValueError('Run frame ordering/FPS differs from the dataset clip')
    rows, seen = [], set()
    with label.open(newline='') as stream:
        for item in csv.DictReader(stream):
            index, visible = int(item['Frame']), int(item['Visibility'])
            if index in seen or not 0 <= index < metadata.frame_count or visible not in (0, 1):
                raise ValueError('Invalid or duplicate explicit label')
            seen.add(index)
            observation = detections['frames'][index]
            if observation['frame_index'] != index:
                raise ValueError('Detection frame IDs are not aligned')
            rows.append({'clip': args.clip, 'frame': index, 'width': metadata.width, 'height': metadata.height,
                         'target_xy': [float(item['X']) * metadata.width / 1920,
                                       float(item['Y']) * metadata.height / 1080] if visible else None,
                         'prediction_xy': observation['ball']['position_px']})
    metrics = {str(t): evaluate_points(rows, t) for t in (2, 4, 8)}
    report = {'schema_version': '1.0', 'qualification_evidence': False,
              'run': str(args.run), 'dataset': str(args.dataset), 'clip': args.clip,
              'detections_sha256': digest(args.run / 'detections.json'), 'labels_sha256': digest(label),
              'dataset_manifest_sha256': digest(args.dataset / 'manifest.json'), 'evaluator_sha256': digest(__file__),
              'metrics': metrics, 'labeled_rows': rows,
              'measurement_scope': 'EXPLICIT_SPARSE_PUBLISHER_LABELS; ACTUAL_COMPLETE_PIPELINE_EXPORT'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics['4'].items() if k != 'localization_errors_reference_px'}), flush=True)


if __name__ == '__main__':
    main()
