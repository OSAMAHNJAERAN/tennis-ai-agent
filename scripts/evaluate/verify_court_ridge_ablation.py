"""Independently replay saved ablation metrics without importing its scorer."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summaries = {}
    for mode in ('no_white', 'no_sides', 'ridge_only'):
        output = ROOT/f'outputs/vision_upgrade_audit/court_ridge_ablation_{mode}01'
        path = output/'report.json'
        report = json.loads(path.read_text())
        assert report['complete'] and report['mode'] == mode
        assert len(report['cases']) == 64
        for source, checksum in {**report['sources'], **report['code_hashes']}.items():
            assert digest(ROOT/source) == checksum
        assert digest(ROOT/'docs/experiments/COURT_RIDGE_ABLATION_PROTOCOL.md') == report['protocol_sha256']
        pooled, paired = {}, []
        for row in report['cases']:
            if row['initial_valid']:
                assert digest(output/row['image']) == row['image_sha256']
                assert row['original_replay_maximum_difference_reference_px'] < 1e-5
            if 'scores' not in row:
                continue
            for stage, score in row['scores'].items():
                counts = dict(tp=0, fp=0, fn=0)
                errors = []
                source = row['original']['scores']['projected']['landmarks']
                assert len(score['landmarks']) == len(source) == 14
                for identity, point in enumerate(score['landmarks']):
                    assert point['identity'] == identity
                    assert point['label'] == source[identity]['label']
                    x, y = point['label']
                    eligible = 0 <= x < 1280 and 0 <= y < 720
                    prediction = point['prediction']
                    distance = None if prediction is None else math.hypot(prediction[0]-x, prediction[1]-y)
                    assert eligible == point['eligible']
                    assert (distance is None and point['distance_native_px'] is None) or math.isclose(distance, point['distance_native_px'], abs_tol=1e-10)
                    correct = eligible and distance is not None and distance <= 7
                    assert point['correct'] == correct
                    if eligible:
                        counts['tp'] += int(correct)
                        counts['fn'] += int(not correct)
                        counts['fp'] += int(prediction is not None and not correct)
                        if distance is not None:
                            errors.append(distance)
                assert all(score[k] == counts[k] for k in counts)
                aggregate = pooled.setdefault((row['group'], stage), dict(tp=0, fp=0, fn=0, errors=[]))
                for key in counts:
                    aggregate[key] += counts[key]
                aggregate['errors'].extend(errors)
            entry = dict(index=row['index'], id=row['id'], group=row['group'])
            after = row['scores']['ablation']['landmarks']
            for reference in ('projected', 'original'):
                before = row['scores'][reference]['landmarks']
                entry[reference] = dict(
                    gained=[i for i, (a, b) in enumerate(zip(before, after)) if b['correct'] and not a['correct']],
                    lost=[i for i, (a, b) in enumerate(zip(before, after)) if a['correct'] and not b['correct']])
            paired.append(entry)
        for (group, stage), aggregate in pooled.items():
            expected = report['pooled'][group][stage]
            for key in ('tp', 'fp', 'fn'):
                assert aggregate[key] == expected[key]
            errors = aggregate['errors']
            assert len(errors) == expected['available']
            for key, actual in [
                ('mean_error_px', statistics.mean(errors)), ('median_error_px', statistics.median(errors)),
                ('precision', aggregate['tp']/(aggregate['tp']+aggregate['fp'])),
                ('recall', aggregate['tp']/(aggregate['tp']+aggregate['fn']))]:
                assert math.isclose(expected[key], actual, abs_tol=1e-10)
        changes = {}
        for group in ('selection', 'external_source_check'):
            changes[group] = {reference: {kind: sum(len(row[reference][kind]) for row in paired if row['group'] == group)
                                         for kind in ('gained', 'lost')} for reference in ('projected', 'original')}
        review = dict(complete=True, report_sha256=digest(path), mode=mode,
                      metrics_independently_replayed=True, labels_unchanged=True,
                      qualification_evidence=False, paired_counts=changes, per_image=paired,
                      changed_indices=[row['index'] for row in report['cases'] if row.get('changed_from_previous')],
                      accepted_by_group={group: sum(row['accepted'] for row in report['cases'] if row['group'] == group)
                                         for group in ('selection', 'external_source_check', 'pilot')},
                      visual_review_complete=False)
        target = output/'review.json'
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(review, indent=2), encoding='utf-8')
        summaries[mode] = dict(pooled=report['pooled'], paired_counts=changes,
                               accepted=review['accepted_by_group'], changed=review['changed_indices'])
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
