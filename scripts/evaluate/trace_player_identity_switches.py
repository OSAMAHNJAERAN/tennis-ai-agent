"""Record switch locations from the pinned official CLEAR assignment calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import load_trackeval_metrics, prepare_sequence


def trace(ground_truth, predictions, trackeval_path):
    classes, provenance = load_trackeval_metrics(trackeval_path)
    data = prepare_sequence(ground_truth, predictions)
    module = sys.modules[classes['CLEAR'].__module__]
    original = module.linear_sum_assignment
    assignments = []

    def capture(cost):
        rows, columns = original(cost)
        valid = -cost[rows, columns] > np.finfo(float).eps
        assignments.append((rows[valid], columns[valid]))
        return rows, columns

    module.linear_sum_assignment = capture
    try:
        result = classes['CLEAR']().eval_sequence(data)
    finally:
        module.linear_sum_assignment = original
    active_frames = [i for i, (gt, pred) in enumerate(zip(ground_truth, predictions, strict=True)) if gt and pred]
    if len(active_frames) != len(assignments):
        raise ValueError('Official assignment trace differs from frame coverage')
    last, switches, matched = {}, [], []
    for index, (rows, columns) in zip(active_frames, assignments, strict=True):
        for row, column in zip(rows, columns, strict=True):
            gt, prediction = ground_truth[index][int(row)], predictions[index][int(column)]
            record = {'frame': index, 'gt_id': gt['id'], 'tracker_id': prediction['id'],
                      'source_id': prediction.get('source_id', prediction['id'])}
            matched.append(record)
            previous = last.get(gt['id'])
            if previous and previous['tracker_id'] != prediction['id']:
                switches.append({'previous_match': previous, 'current_match': record,
                                 'elapsed_frames': index-previous['frame']})
            last[gt['id']] = record
    if len(switches) != int(result['IDSW']) or len(matched) != int(result['CLR_TP']):
        raise ValueError('Trace does not reproduce official counts')
    return switches, {k: int(result[k]) for k in ('IDSW', 'CLR_TP', 'CLR_FP', 'CLR_FN')}, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT/'outputs/vision_upgrade_audit'
    dataset = ROOT/'data/external/uvy_tennis_videos'
    result = {'complete': False, 'qualification_evidence': False, 'script_sha256': digest(__file__),
              'scope': 'PINNED_CLEAR_ASSIGNMENT_TRACE; ORIGINAL_PUBLISHER_LABELS', 'models': {}}
    for label, folder in [('original', 'uvy_duplicate_handoff_pilot01'), ('trained', 'uvy_trained_racket_players_pilot01')]:
        path = base/folder/'report.json'
        report = json.loads(path.read_text())
        if not report['complete'] or digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
            raise ValueError('Incomplete or changed dataset')
        result['models'][label] = {'report_sha256': digest(path), 'sequences': {}}
        for sequence, info in report['sequences'].items():
            selected_path = path.parent/info['selected_file']
            if digest(selected_path) != info['selected_sha256']:
                raise ValueError('Selected observations changed')
            predictions = json.loads(selected_path.read_text())
            gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(predictions))]
            switches, counts, provenance = trace(gt, predictions, ROOT/'artifacts/research/TrackEval')
            if any(value != info['metrics']['summary'][key] for key, value in counts.items()):
                raise ValueError('Original report counts do not reproduce')
            result['models'][label]['sequences'][sequence] = {'switches': switches, 'verified_counts': counts}
            result['trackeval_provenance'] = provenance
    result['complete'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['models']))


if __name__ == '__main__':
    main()
