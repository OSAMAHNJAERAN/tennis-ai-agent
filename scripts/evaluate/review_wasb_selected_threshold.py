"""Independent scoring and provenance review of the frozen operating-point test."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def error(point, target, width, height):
    return math.hypot((point[0]-target[0])*512/width, (point[1]-target[1])*288/height)


def outcome(point, target, width, height):
    if target is None:
        return 'correct_absence' if point is None else 'absent_false_detection'
    if point is None:
        return 'visible_abstention'
    return 'correct_ball' if error(point, target, width, height) <= 4 else 'wrong_location'


def score(rows, name):
    counts = Counter(outcome(r[name], r['target_xy'], r['width'], r['height']) for r in rows)
    tp, wrong, missing, afp, tn = [counts[k] for k in ('correct_ball', 'wrong_location', 'visible_abstention', 'absent_false_detection', 'correct_absence')]
    fp, fn = wrong + afp, wrong + missing
    return dict(true_positives=tp, false_positives=fp, false_negatives=fn, true_negatives=tn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                absent_false_detections=afp, wrong_visible_localizations=wrong, visible_abstentions=missing)


def main():
    out = ROOT / 'outputs/vision_upgrade_audit/wasb_selected_threshold01'
    target = out / 'review.json'
    if target.exists():
        raise FileExistsError(target)
    report = read(out / 'report.json')
    assert report['complete'] and len(report['clips']) == 18
    provenance = report['provenance']
    assert digest(ROOT / 'docs/experiments/WASB_SELECTED_THRESHOLD_PROTOCOL.md') == provenance['protocol_sha256']
    for path, checksum in provenance['code_hashes'].items():
        assert digest(ROOT / path) == checksum
    training_path = ROOT / 'outputs/vision_upgrade_audit/wasb_training_threshold01/report.json'
    assert digest(training_path) == provenance['selection_sha256']
    selection = read(training_path)
    original = next(m for m in selection['models'] if m['name'] == 'original')
    def selection_key(item):
        t, metrics = item
        m = metrics['pooled']
        return m['f1'], m['precision'], m['recall'], -abs(float(t)-.2), -float(t)
    selected_threshold = float(max(original['scores'].items(), key=selection_key)[0])
    assert selected_threshold == .35 == report['configuration']['selected_threshold']
    assert report['configuration']['control_threshold'] == .2
    assert digest(ROOT / original['checkpoint']) == provenance['checkpoint_sha256'] == original['checkpoint_sha256']
    audit_path = ROOT / 'outputs/vision_upgrade_audit/ball_spaced_error_audit01/report.json'
    assert digest(audit_path) == provenance['audit_sha256']
    audit = read(audit_path)
    old = {(r['clip'], r['frame']): r for r in audit['rows']}
    paired, seen, changes = [], set(), []
    coverage = Counter()
    for clip in report['clips']:
        assert clip['existing_inference_reference_exact']
        folder = ROOT / 'data/external' / clip['dataset']
        assert digest(folder / 'manifest.json') == provenance['dataset_manifests'][clip['dataset']]
        match, rally = clip['clip'].rsplit('_', 1)
        label_path = folder / f'tennis/all/{match}/csv/{rally}_ball.csv'
        assert digest(label_path) == clip['label_sha256']
        assert digest(folder / f"tennis/videos/{clip['clip']}.mp4") == clip['video_sha256']
        with label_path.open(newline='') as handle:
            source = list(csv.DictReader(handle))
        labels = {int(r['Frame']): r for r in source}
        assert len(source) == len(labels) == len(clip['rows'])
        assert set(labels) == {r['frame'] for r in clip['rows']}
        width, height, stride = clip['width'], clip['height'], clip['stride']
        assert stride == max(1, int(math.floor(clip['fps']/30+.5)))
        for row in clip['rows']:
            key = clip['clip'], row['frame']
            assert key not in seen
            seen.add(key)
            label = labels[row['frame']]
            expected = [float(label['X'])*width/1920, float(label['Y'])*height/1080] if int(label['Visibility']) else None
            assert expected == row['target_xy'] == old[key]['target_xy']
            windows = []
            for slot in (2, 1, 0):
                start = row['frame'] - slot*stride
                if start >= 0 and start + 2*stride < clip['frames']:
                    windows.append([[start, start+stride, start+2*stride], slot])
            assert row['windows'] == windows and windows
            pair = dict(clip=clip['clip'], frame=row['frame'], dataset=clip['dataset'], width=width, height=height,
                        target_xy=expected, video_sha256=clip['video_sha256'])
            for name in ('control', 'selected'):
                result = row[name]
                for p in result['candidates']:
                    assert all(math.isfinite(p[k]) for k in ('x', 'y', 'confidence'))
                    assert 0 <= p['x'] < width and 0 <= p['y'] < height and 0 <= p['view'] <= 4
                best = max(result['candidates'], key=lambda p:p['confidence'], default=None)
                assert result['prediction_xy'] == ([best['x'], best['y']] if best else None)
                pair[name] = result['prediction_xy']
                if expected is not None:
                    coverage[name] += any(error([p['x'],p['y']], expected, width, height) <= 4 for p in result['candidates'])
            reference = old[key]['stages']['raw']
            if pair['control'] is None or reference is None:
                assert pair['control'] == reference
            else:
                assert max(abs(a-b) for a,b in zip(pair['control'], reference, strict=True)) <= .0001
            pair['before_outcome'] = outcome(pair['control'], expected, width, height)
            pair['after_outcome'] = outcome(pair['selected'], expected, width, height)
            good = {'correct_ball', 'correct_absence'}
            before, after = pair['before_outcome'] in good, pair['after_outcome'] in good
            if before != after:
                pair['kind'] = ('gained_' if after else 'lost_') + ('absence' if expected is None else 'visible')
                pair['changed_visible_boundary'] = expected is not None and pair['control'] is not None and pair['selected'] is not None
                changes.append(pair.copy())
            paired.append(pair)
    assert seen == set(old) and len(paired) == 900
    metrics = {name: score(paired, name) for name in ('control', 'selected')}
    per_clip = {c: {n:score([r for r in paired if r['clip']==c],n) for n in metrics} for c in sorted({r['clip'] for r in paired})}
    for name, measured in metrics.items():
        assert all(value == report['scores'][name]['pooled'][k] for k,value in measured.items())
        for c, values in per_clip.items():
            assert all(value == report['scores'][name]['per_clip'][c][k] for k,value in values[name].items())
    a,b = metrics['control'], metrics['selected']
    advance = b['f1'] > a['f1'] and b['precision'] >= a['precision'] and b['recall'] >= a['recall']
    assert advance == report['advance_to_continuous']
    selected, used = [], Counter()
    for r in sorted(changes,key=lambda r:(r['clip'],r['frame'])):
        if used[r['kind']] < 2 or r['changed_visible_boundary']:
            selected.append(r.copy())
            used[r['kind']] += 1
    review = dict(complete=True, qualification_evidence=False, report_sha256=digest(out/'report.json'),
                  reviewer_sha256=digest(Path(__file__)), independent_scores_exact=True, labels_exact=True,
                  all_900_control_predictions_match=True, metrics=metrics, per_clip=per_clip,
                  correct_proposal_coverage=dict(coverage), visible_labels=sum(r['target_xy'] is not None for r in paired),
                  paired_changes=dict(Counter(r['kind'] for r in changes)), changes=changes, selected_review=selected,
                  advance_to_continuous=advance, next_gate='CONTINUOUS_PAIRED_COMPARISON' if advance else 'REJECT_THRESHOLD_PROMOTION',
                  limitation='Sparse raw outputs on reused development labels; no final-pipeline or independent qualification.')
    target.write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('changes','selected_review','per_clip')},indent=2))


if __name__ == '__main__':
    main()
