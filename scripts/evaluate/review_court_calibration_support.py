"""Independently verify fixed calibration ablation and record completed visual review."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(total):
    errors = total.pop('errors')
    tp, fp, fn = (total[k] for k in ('tp', 'fp', 'fn'))
    return dict(**total, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn), available_predictions=len(errors),
                mean_native_error_available_px=statistics.mean(errors) if errors else None,
                median_native_error_available_px=statistics.median(errors) if errors else None)


def main():
    out = ROOT/'outputs/vision_upgrade_audit/court_calibration_support01'
    target = out/'review.json'
    if target.exists():
        raise FileExistsError(target)
    report = json.loads((out/'report.json').read_text())
    assert report['complete'] and len(report['cases']) == 183
    for key in ('sources', 'code_hashes'):
        for name, checksum in report[key].items():
            assert digest(ROOT/name) == checksum
    assert digest(ROOT/'docs/experiments/COURT_CALIBRATION_SUPPORT_PROTOCOL.md') == report['protocol_sha256']
    manifest = json.loads((ROOT/'data/external/court_heatmap_pilot/manifest.json').read_text())
    labels = {row['id']: row['kps'] for row in manifest['samples']}
    # Independent canonical construction in unchanged detector identity order.
    canonical = np.array([[0,0],[10.97,0],[0,23.77],[10.97,23.77],
                          [1.37,0],[1.37,23.77],[9.60,0],[9.60,23.77],
                          [1.37,5.485],[9.60,5.485],[1.37,18.285],[9.60,18.285],
                          [5.485,5.485],[5.485,18.285]])
    # Source geometry is stored as float32 before the benchmark's float64 cast.
    canonical = canonical.astype(np.float32).astype(float)
    pooled, groups, decisions, new = {}, {}, {}, []
    max_projection_difference = max_refinement_difference = 0.
    observed = {
        28: 'Madrid: refinement aligns near baseline and sidelines; all fourteen labels match.',
        29: 'Madrid: near baseline and sidelines improve; all fourteen labels match.',
        166: 'Amateur Wimbledon view: entire projected court lies over spectators above the real court; false acceptance. Refinement rejects and preserves the bad initialization.'}
    for index, row in enumerate(report['cases']):
        assert index == row['index']
        before, after = row['default'], row['alternate']
        assert before['inlier_mask'] == after['inlier_mask']
        assert len(row['available_ids']) == len(set(row['available_ids']))
        for result in (before, after):
            assert result['inlier_count'] == sum(result['inlier_mask'])
            if result['inlier_mask']:
                assert result['correspondence_count'] == len(row['available_ids'])
        assert row['newly_accepted'] == (after['is_valid'] and not before['is_valid'])
        assert not before['is_valid'] or after['is_valid']
        if before['is_valid']:
            assert before == after
            assert row['predictions']['default'] == row['predictions']['alternate']
        for stage in ('default', 'alternate'):
            result = row[stage]
            count = decisions.setdefault(row['group'], dict(cases=0, default_accepted=0, alternate_accepted=0))
            if stage == 'default':
                count['cases'] += 1
            count[stage+'_accepted'] += int(result['is_valid'])
            points = row['predictions'][stage]
            if result['is_valid']:
                homogeneous = np.column_stack((canonical, np.ones(14))) @ np.linalg.inv(result['homography_matrix']).T
                projected = homogeneous[:,:2]/homogeneous[:,2:]
                error = float(np.abs(projected-np.array(points)).max())
                max_projection_difference = max(max_projection_difference, error)
                assert error < 1e-8
            else:
                assert points == [None]*14
        if row['newly_accepted']:
            assert index in observed and digest(out/row['image']) == row['image_sha256']
            evidence = row['refinement']
            initial = np.array(row['predictions']['alternate'])
            if evidence['accepted']:
                scale = 960/row['native_size'][0]
                iteration = evidence['iterations'][-1]
                homogeneous = np.column_stack((initial*scale/960, np.ones(14))) @ np.array(iteration['normalized_image_transform']).T
                expected = homogeneous[:,:2]/homogeneous[:,2:]*960/scale
            else:
                expected = initial
            error = float(np.abs(expected-np.array(row['predictions']['refined'])).max())
            max_refinement_difference = max(max_refinement_difference, error)
            assert error < 1e-8
            new.append(dict(index=index, id=row['id'], model=row['model'],
                            correspondence_count=after['correspondence_count'], inlier_count=after['inlier_count'],
                            inlier_error_native_px=after['inlier_reprojection_error_px'],
                            refinement_accepted=evidence['accepted'], refinement_reason=evidence['reason'],
                            support_before=evidence['before']['mean_visible_line_support'],
                            support_after=evidence.get('proposed_after',evidence['before'])['mean_visible_line_support'],
                            image=row['image'], image_sha256=row['image_sha256'],
                            visually_inspected=True, observation=observed[index]))
        if 'scores' not in row:
            continue
        for stage, scored in row['scores'].items():
            counts = dict(tp=0, fp=0, fn=0, errors=[])
            for identity, item in enumerate(scored['landmarks']):
                assert item['identity'] == identity and item['label'] == labels[row['id']][identity]
                point = row['predictions'][stage][identity]
                assert point == item['prediction']
                x,y = item['label']
                eligible = 0 <= x < 1280 and 0 <= y < 720
                distance = None if point is None else math.hypot(point[0]-x,point[1]-y)
                assert (distance is None and item['distance_native_px'] is None) or math.isclose(distance,item['distance_native_px'],abs_tol=1e-10)
                correct = eligible and distance is not None and distance <= 7
                assert item['eligible'] == eligible and item['correct'] == correct
                if eligible:
                    counts['tp'] += int(correct)
                    counts['fp'] += int(point is not None and not correct)
                    counts['fn'] += int(not correct)
                    if distance is not None:
                        counts['errors'].append(distance)
            assert all(scored[k] == counts[k] for k in ('tp','fp','fn'))
            # Refined scores exist only on newly accepted labeled cases.
            for container, key in ((pooled,stage),(groups,row['group']+'/'+stage)):
                total = container.setdefault(key,dict(tp=0,fp=0,fn=0,errors=[],images=0))
                total['images'] += 1
                for name in ('tp','fp','fn'):
                    total[name] += counts[name]
                total['errors'].extend(counts['errors'])
    assert {row['index'] for row in new} == set(observed)
    review = dict(complete=True, qualification_evidence=False,
                  report_sha256=digest(out/'report.json'), reviewer_sha256=digest(Path(__file__)),
                  labels_unchanged=True, metrics_independently_replayed=True,
                  maximum_projection_difference_native_px=max_projection_difference,
                  maximum_refinement_difference_native_px=max_refinement_difference,
                  decisions=decisions, pooled={k:summarize(v) for k,v in pooled.items()},
                  groups={k:summarize(v) for k,v in groups.items()}, newly_accepted=new,
                  decision='REJECT_GLOBAL_RATIO_REDUCTION: one visually clear false acceptance among three new fits; no runtime mutation.')
    target.write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(review,indent=2))


if __name__ == '__main__':
    main()
