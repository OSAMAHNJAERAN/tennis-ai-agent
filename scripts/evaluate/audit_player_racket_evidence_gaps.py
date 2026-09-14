"""Label-assisted diagnosis of crop coverage and player evidence fragmentation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import box_iou_matrix


def matches(people, truth):
    ious = box_iou_matrix([r['box'] for r in people], [r['box'] for r in truth])
    valid = ious >= .5
    rows, columns = linear_sum_assignment(valid*(min(len(people), len(truth))+1+ious), maximize=True)
    return [(int(a), int(b)) for a, b in zip(rows, columns) if valid[a, b]]


def coverage(box, crop):
    x1, y1, x2, y2 = box
    a, b, c, d = crop
    return max(0, min(x2, c)-max(x1, a))*max(0, min(y2, d)-max(y1, b))/((x2-x1)*(y2-y1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT/'outputs/vision_upgrade_audit'
    person_path = base/'uvy_person_tracked640/report.json'
    selected_path = base/'uvy_racket_supported_players_pilot02_nearest/report.json'
    racket_path = base/'racketvision_racket_global_assignment640.json'
    person, selected, racket = [json.loads(p.read_text()) for p in (person_path, selected_path, racket_path)]
    if not all(r['complete'] for r in (person, selected, racket)):
        raise ValueError('Incomplete inputs')
    dataset = ROOT/'data/external/uvy_tennis_videos'
    if digest(dataset/'manifest.json') != person['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    for item in json.loads((dataset/'manifest.json').read_text())['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset file changed')
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'LABEL_ASSISTED_ERROR_DECOMPOSITION; NOT_DEPLOYABLE_SELECTION',
              'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in (person_path, selected_path, racket_path)},
              'script_sha256': digest(__file__), 'uvy': {}}
    for sequence, info in person['sequences'].items():
        path = person_path.parent/info['predictions_file']
        if digest(path) != info['predictions_sha256']:
            raise ValueError('Person cache changed')
        people = json.loads(path.read_text())
        truth = [[r for r in frame if r['class'] == 1] for frame in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(people))]
        statistics = selected['sequences'][sequence]['track_statistics']
        tracks = defaultdict(lambda: {'gt_frames': 0, 'proposal_matches': 0, 'supported_matches': 0,
                                       'source_matches': Counter(), 'source_heights': defaultdict(list)})
        for rows, gt in zip(people, truth, strict=True):
            for target in gt:
                tracks[target['id']]['gt_frames'] += 1
            for pi, gi in matches(rows, gt):
                target, proposal = gt[gi], rows[pi]
                track = tracks[target['id']]
                identity = str(proposal['id'])
                track['proposal_matches'] += 1
                track['source_matches'][identity] += 1
                track['source_heights'][identity].append(proposal['box'][3]-proposal['box'][1])
                if statistics[identity]['eligible']:
                    track['supported_matches'] += 1
        for track in tracks.values():
            track['source_tracks'] = [
                {'source_id': int(identity), 'matched_frames': n,
                 'median_person_height_px': float(np.median(track['source_heights'][identity])),
                 **statistics[identity]}
                for identity, n in track.pop('source_matches').most_common()]
            track.pop('source_heights')
        report['uvy'][sequence] = {'gt_tracks': dict(tracks),
            'gt_boxes': sum(t['gt_frames'] for t in tracks.values()),
            'matched_person_proposals': sum(t['proposal_matches'] for t in tracks.values()),
            'matched_proposals_on_supported_source_ids': sum(t['supported_matches'] for t in tracks.values())}
    annotation_path = ROOT/'data/external/racketvision_validation/tennis/info/val_coco.json'
    if digest(annotation_path) != racket['annotation_sha256']:
        raise ValueError('Racket labels changed')
    annotations = json.loads(annotation_path.read_text())
    sizes = {r['file_name']: (r['width'], r['height']) for r in annotations['images']}
    crop_rows = []
    for item in racket['per_image']:
        width, height = sizes[item['image']]
        crops = []
        for identity, box in item['player_boxes'].items():
            if box is None:
                continue
            x1, y1, x2, y2 = box
            margin = .75*(y2-y1)
            crops.append((identity, [max(0, int(x1-margin)), max(0, int(y1-.6*margin)),
                                     min(width, int(x2+margin)), min(height, int(y2+.2*margin))]))
        for index, box in enumerate(item['targets']):
            values = [(identity, coverage(box, crop)) for identity, crop in crops]
            best = max(values, key=lambda r: r[1], default=(None, 0))
            raw_iou = box_iou_matrix([r['bbox_xyxy'] for r in item['raw_candidates']], [box])
            crop_rows.append({'image': item['image'], 'target_index': index, 'box': box,
                              'best_crop_source': best[0], 'best_crop_coverage': best[1],
                              'raw_proposal_iou50_available': bool(np.any(raw_iou >= .5)),
                              'width_px': box[2]-box[0], 'height_px': box[3]-box[1]})
    report['racket_crop_coverage'] = {'targets': len(crop_rows),
        'at_least_90_percent_inside_one_crop': sum(r['best_crop_coverage'] >= .9 for r in crop_rows),
        'fully_inside_one_crop': sum(r['best_crop_coverage'] >= 1-1e-9 for r in crop_rows),
        'raw_missing_despite_90_percent_coverage': sum(r['best_crop_coverage'] >= .9 and not r['raw_proposal_iou50_available'] for r in crop_rows),
        'per_target': crop_rows,
        'limitations': 'Sparse broadcast racket-positive labels, not UVY racket truth. Coverage of any crop does not establish ownership. Per-target proposal availability is not one-to-one recall.'}
    report['complete'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'uvy': {s: {k: v for k, v in r.items() if k != 'gt_tracks'} for s, r in report['uvy'].items()},
                      'racket_crop_coverage': {k: v for k, v in report['racket_crop_coverage'].items() if k != 'per_target'}}))


if __name__ == '__main__':
    main()
