"""Independently verify the head-only candidate's frozen external comparison."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.review_wasb_selected_threshold import digest, read, error, outcome, score


def main():
    out = ROOT/'outputs/vision_upgrade_audit/wasb_head_external01'
    target = out/'review.json'
    if target.exists():
        raise FileExistsError(target)
    report = read(out/'report.json')
    assert report['complete'] and len(report['clips']) == 18
    provenance = report['provenance']
    sources = {
        'training_manifest_sha256':'artifacts/training/vision_upgrade/wasb_head_pilot04/manifest.json',
        'internal_review_sha256':'outputs/vision_upgrade_audit/wasb_head_pilot04/review.json',
        'baseline_sha256':'outputs/vision_upgrade_audit/wasb_selected_threshold01/report.json',
        'baseline_review_sha256':'outputs/vision_upgrade_audit/wasb_selected_threshold01/review.json',
        'protocol_sha256':'docs/experiments/WASB_HEAD_EXTERNAL_PROTOCOL.md',
    }
    for key, path in sources.items():
        assert digest(ROOT/path) == provenance[key]
    for path, checksum in provenance['code_hashes'].items():
        assert digest(ROOT/path) == checksum
    training = read(ROOT/sources['training_manifest_sha256'])
    internal = read(ROOT/sources['internal_review_sha256'])
    assert internal['complete'] and internal['manifest_sha256'] == provenance['training_manifest_sha256']
    assert all(m['selected_threshold'] == '0.35' for m in internal['models'])
    for key in ('initial', 'trained'):
        assert digest(ROOT/training[key+'_checkpoint']) == provenance[key+'_checkpoint_sha256']
    assert report['configuration'] == dict(original_threshold=.35, adapted_threshold=.35, crop_fraction=.6,
                                           target_temporal_fps=30, merge_radius_reference_px=4)
    baseline = read(ROOT/sources['baseline_sha256'])
    old = {c['clip']:c for c in baseline['clips']}
    assert [c['clip'] for c in report['clips']] == [c['clip'] for c in baseline['clips']]
    train_ids = {c.rsplit('_', 1)[0] for c in training['train_clips']+training['selection_clips']}
    assert train_ids.isdisjoint({c.rsplit('_', 1)[0] for c in old})
    paired, changes, seen = [], [], set()
    coverage = Counter()
    for c in report['clips']:
        assert c['internal_inference_reference_exact']
        folder = ROOT/'data/external'/c['dataset']
        assert digest(folder/'manifest.json') == provenance['dataset_manifests'][c['dataset']]
        match, rally = c['clip'].rsplit('_', 1)
        label_path = folder/f'tennis/all/{match}/csv/{rally}_ball.csv'
        assert digest(label_path) == c['label_sha256']
        assert digest(folder/f"tennis/videos/{c['clip']}.mp4") == c['video_sha256']
        for key in ('dataset', 'width', 'height', 'fps', 'stride', 'frames', 'video_sha256', 'label_sha256'):
            assert c[key] == old[c['clip']][key]
        with label_path.open(newline='') as handle:
            source = list(csv.DictReader(handle))
        labels = {int(r['Frame']):r for r in source}
        assert len(source) == len(labels) == len(c['rows'])
        assert set(labels) == {r['frame'] for r in c['rows']}
        old_rows = {r['frame']:r for r in old[c['clip']]['rows']}
        assert set(old_rows) == set(labels)
        width, height, stride = c['width'], c['height'], c['stride']
        assert stride == max(1, int(math.floor(c['fps']/30+.5)))
        for r in c['rows']:
            key = c['clip'], r['frame']
            assert key not in seen
            seen.add(key)
            label = labels[r['frame']]
            expected = [float(label['X'])*width/1920, float(label['Y'])*height/1080] if int(label['Visibility']) else None
            assert expected == r['target_xy'] == old_rows[r['frame']]['target_xy']
            assert r['original'] == old_rows[r['frame']]['selected']
            assert r['control'] == old_rows[r['frame']]['control']
            windows = []
            for slot in (2, 1, 0):
                start = r['frame']-slot*stride
                if start >= 0 and start+2*stride < c['frames']:
                    windows.append([[start, start+stride, start+2*stride], slot])
            assert windows and r['windows'] == windows == old_rows[r['frame']]['windows']
            pair = dict(clip=c['clip'], frame=r['frame'], width=width, height=height, fps=c['fps'], target_xy=expected,
                        dataset=c['dataset'], video_sha256=c['video_sha256'])
            for name in ('original', 'adapted', 'control'):
                result = r[name]
                for p in result['candidates']:
                    assert all(math.isfinite(p[k]) for k in ('x', 'y', 'confidence'))
                    assert 0 <= p['x'] < width and 0 <= p['y'] < height and 0 <= p['view'] <= 4
                best = max(result['candidates'], key=lambda p:p['confidence'], default=None)
                assert result['prediction_xy'] == ([best['x'], best['y']] if best else None)
                pair[name] = result['prediction_xy']
                if expected is not None:
                    coverage[name] += any(error([p['x'], p['y']], expected, width, height) <= 4 for p in result['candidates'])
            before = outcome(pair['original'], expected, width, height)
            after = outcome(pair['adapted'], expected, width, height)
            if before != after:
                changes.append(dict(**pair, before_xy=pair['original'], after_xy=pair['adapted'],
                                    kind=before+' -> '+after))
            paired.append(pair)
    assert len(paired) == 900
    metrics = {n:score(paired, n) for n in ('original', 'adapted', 'control')}
    per_clip = {c:{n:score([r for r in paired if r['clip'] == c], n) for n in metrics} for c in sorted(old)}
    per_fps = {str(fps):{n:score([r for r in paired if r['fps'] == fps], n) for n in metrics}
               for fps in sorted({r['fps'] for r in paired})}
    for name, values in metrics.items():
        assert all(v == report['scores'][name]['pooled'][k] for k,v in values.items())
        for c, values_by_name in per_clip.items():
            assert all(v == report['scores'][name]['per_clip'][c][k] for k,v in values_by_name[name].items())
    passing = {n:sum(v[n]['precision'] is not None and v[n]['recall'] is not None and
                     v[n]['precision'] >= .95 and v[n]['recall'] >= .95 for v in per_clip.values()) for n in metrics}
    a, b = metrics['original'], metrics['adapted']
    advance = b['f1'] > a['f1'] and b['precision'] >= a['precision'] and b['recall'] >= a['recall'] and passing['adapted'] >= passing['original']
    assert advance == report['advance_to_continuous'] and passing == report['individually_passing_clips']
    selected, used = [], Counter()
    for r in sorted(changes, key=lambda r:(r['clip'], r['frame'])):
        if used[r['kind']] < 2:
            selected.append(r)
            used[r['kind']] += 1
    result = dict(complete=True, qualification_evidence=False, report_sha256=digest(out/'report.json'),
                  reviewer_sha256=digest(Path(__file__)), independent_scores_exact=True,
                  all_900_original_predictions_and_candidates_exact=True, labels_and_windows_exact=True,
                  metrics=metrics, per_clip=per_clip, per_fps=per_fps, correct_proposal_coverage=dict(coverage),
                  individually_passing_clips=passing, paired_changes=dict(Counter(r['kind'] for r in changes)),
                  changes=changes, selected_review=selected, advance_to_continuous=advance,
                  next_gate='CONTINUOUS_FILTERED_COMPARISON' if advance else 'REJECT_HEAD_PAIR_PROMOTION')
    target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_clip', 'per_fps', 'changes', 'selected_review')}, indent=2))


if __name__ == '__main__':
    main()
