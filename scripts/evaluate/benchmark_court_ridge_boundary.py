"""Combine two frozen court improvements using verified saved line evidence."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.refine_court_line_geometry import refine as previous_refine, LINES
from scripts.evaluate.refine_court_boundary_geometry import refine
from scripts.evaluate.benchmark_court_line_refinement import digest
from scripts.evaluate.benchmark_court_refinement_labels import score


def main():
    parent = ROOT/'outputs/vision_upgrade_audit/court_ridge_ablation_no_white01/report.json'
    source = json.loads(parent.read_text())
    assert source['complete'] and len(source['cases']) == 64
    for name, checksum in {**source['sources'], **source['code_hashes']}.items():
        assert digest(ROOT/name) == checksum
    boundary_source = ROOT/'outputs/vision_upgrade_audit/court_boundary_refinement01/report.json'
    boundary_report = json.loads(boundary_source.read_text())
    for name, checksum in boundary_report['code_hashes'].items():
        assert digest(ROOT/name) == checksum
    out = ROOT/'outputs/vision_upgrade_audit/court_ridge_boundary01'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = dict(complete=False, qualification_evidence=False,
                  scope='COMBINATION_SELECTED_ON_REUSED_DEVELOPMENT_DATA',
                  sources={str(p.relative_to(ROOT)): digest(p) for p in (parent, boundary_source)},
                  protocol_sha256=digest(ROOT/'docs/experiments/COURT_RIDGE_BOUNDARY_PROTOCOL.md'),
                  code_hashes={p: digest(ROOT/p) for p in [
                      'scripts/evaluate/benchmark_court_ridge_boundary.py',
                      'scripts/evaluate/refine_court_boundary_geometry.py',
                      'scripts/evaluate/refine_court_line_geometry.py',
                      'scripts/evaluate/benchmark_court_refinement_labels.py']}, cases=[])
    for old, boundary in zip(source['cases'], boundary_report['cases'], strict=True):
        assert (old['index'], old['id'], old['group']) == (boundary['index'], boundary['id'], boundary['group'])
        row = dict(index=old['index'], id=old['id'], group=old['group'],
                   initial_valid=old['initial_valid'], previous_accepted=old['accepted'])
        scores = None
        if 'scores' in old:
            scores = dict(projected=old['scores']['projected'], original=old['scores']['original'],
                          no_white=old['scores']['ablation'], boundary_only=boundary['scores']['boundary'])
        if not old['initial_valid']:
            row.update(accepted=False, reason='INITIAL_GEOMETRIC_CALIBRATION_INVALID', changed=False)
            if scores is not None:
                scores['combined'] = old['scores']['projected']
                row['scores'] = scores
            report['cases'].append(row)
            continue
        points = np.array(old['original_points'])
        segments = old['segments']
        expected = np.array(old['ablation_refined_points'])
        replay, evidence = previous_refine(points, segments)
        difference = float(np.max(np.abs(replay-expected)))
        assert difference < 1e-5 and evidence['accepted'] == old['accepted']
        actual, evidence = refine(points, segments)
        changed = evidence['accepted'] != old['accepted'] or np.max(np.abs(actual-expected)) > .1
        row.update(accepted=evidence['accepted'], reason=evidence['reason'], evidence=evidence,
                   original_points=points.tolist(), no_white_points=expected.tolist(),
                   combined_points=actual.tolist(), changed=bool(changed),
                   no_white_replay_maximum_difference_reference_px=difference)
        if scores is not None:
            labels = [p['label'] for p in scores['projected']['landmarks']]
            scores['combined'] = score((actual/.75).tolist(), labels)
            row['scores'] = scores
        if changed:
            if old['group'] == 'pilot':
                origin = old['original']
                path = Path(origin['video'])
                assert digest(path) == origin['video_sha256']
                cap = cv2.VideoCapture(str(path))
                cap.set(cv2.CAP_PROP_POS_FRAMES, origin['frame'])
                ok, image = cap.read()
                cap.release()
                assert ok
            else:
                path = ROOT/'data/external/court_heatmap_pilot/images'/f"{old['id']}.png"
                assert digest(path) == old['original']['input_sha256']
                image = cv2.imread(str(path))
            image = cv2.resize(image, (640, 360))
            panels = []
            for name, candidate in [('No whiteness filter', expected), ('With boundary constraints', actual)]:
                panel = image.copy()
                for a, b in LINES:
                    cv2.line(panel, tuple(np.rint(candidate[a]*2/3).astype(int)), tuple(np.rint(candidate[b]*2/3).astype(int)), (30,30,230), 1)
                if scores is not None:
                    for label in labels:
                        cv2.circle(panel, tuple(np.rint(np.array(label)*.5).astype(int)), 4, (30,255,30), 1)
                panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
                cv2.putText(panel, f"{old['index']} {old['id']}: {name}", (5,14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                cv2.putText(panel, 'Green unchanged label / red prediction', (5,27), cv2.FONT_HERSHEY_SIMPLEX, .35, (255,255,255), 1)
                panels.append(panel)
            target = out/f"case_{old['index']:02}.jpg"
            assert cv2.imwrite(str(target), np.concatenate(panels), [cv2.IMWRITE_JPEG_QUALITY, 80])
            row.update(image=target.name, image_sha256=digest(target), visually_inspected=False)
        report['cases'].append(row)
        (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(index=row['index'], accepted=row['accepted'], changed=row['changed'],
                              tp={k:v['tp'] for k,v in row.get('scores',{}).items()})), flush=True)
    pooled = {}
    for group in ('selection', 'external_source_check'):
        pooled[group] = {}
        for stage in ('projected', 'original', 'no_white', 'boundary_only', 'combined'):
            records = [row['scores'][stage] for row in report['cases'] if row['group'] == group]
            counts = {k:sum(x[k] for x in records) for k in ('tp','fp','fn')}
            errors = [p['distance_native_px'] for x in records for p in x['landmarks'] if p['eligible'] and p['distance_native_px'] is not None]
            pooled[group][stage] = dict(**counts, precision=counts['tp']/(counts['tp']+counts['fp']),
                                       recall=counts['tp']/(counts['tp']+counts['fn']), available=len(errors),
                                       mean_error_px=float(np.mean(errors)), median_error_px=float(np.median(errors)))
    report.update(complete=True, pooled=pooled)
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(pooled), flush=True)


if __name__ == '__main__':
    main()
