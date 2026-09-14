"""Verify the frozen 125-image court check and select review images explicitly."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'outputs/vision_upgrade_audit/court_remaining_pilot01'
    report_path = out/'report.json'
    report = json.loads(report_path.read_text())
    assert report['complete'] and len(report['images']) == 125
    dataset = ROOT/'data/external/court_heatmap_pilot'
    manifest_path = dataset/'manifest.json'
    assert digest(manifest_path) == report['manifest_sha256']
    manifest = json.loads(manifest_path.read_text())
    selected = [row for row in manifest['samples'] if row['partition'] == 'train']
    assert [row['id'] for row in selected] == [row['id'] for row in report['images']]
    for name, checksum in report['code_hashes'].items():
        assert digest(ROOT/name) == checksum
    assert digest(ROOT/'models/keypoints_model.pth') == report['checkpoint_sha256']
    assert digest(ROOT/'docs/experiments/COURT_REMAINING_PILOT_PROTOCOL.md') == report['protocol_sha256']
    pooled, paired, excluded = {}, [], []
    maximum_boundary_residual = 0.
    maximum_rejected_roundtrip = 0.
    line_endpoints = [(0,1),(2,3),(0,2),(1,3)]
    for source, row in zip(selected, report['images'], strict=True):
        assert digest(dataset/'images'/f"{row['id']}.png") == row['input_sha256']
        assert digest(out/row['image']) == row['image_sha256']
        if row['initial_calibration']['is_valid']:
            initial = np.array([x['prediction'] for x in row['scores']['projected']['landmarks']])*.75
            for iteration in row['refinement']['iterations']:
                homogeneous = np.column_stack((initial/960, np.ones(14))) @ np.array(iteration['normalized_image_transform']).T
                assert np.all(homogeneous[:,2] > .1)
                moved = homogeneous[:,:2]/homogeneous[:,2:]*960
                for identity in iteration['locked_boundary_lines']:
                    a,b = line_endpoints[identity]
                    delta = initial[b]-initial[a]
                    normal = np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
                    maximum_boundary_residual = max(maximum_boundary_residual, float(np.abs((moved[[a,b]]-initial[a])@normal).max()))
            if not row['refinement']['accepted']:
                actual = np.array([x['prediction'] for x in row['scores']['refined']['landmarks']])*.75
                maximum_rejected_roundtrip = max(maximum_rejected_roundtrip, float(np.max(np.abs(actual-initial))))
        for stage, score in row['scores'].items():
            counts = dict(tp=0, fp=0, fn=0)
            errors = []
            assert len(score['landmarks']) == 14
            for identity, item in enumerate(score['landmarks']):
                assert item['identity'] == identity and item['label'] == source['kps'][identity]
                x, y = item['label']
                eligible = 0 <= x < 1280 and 0 <= y < 720
                point = item['prediction']
                distance = None if point is None else math.hypot(point[0]-x, point[1]-y)
                assert (distance is None and item['distance_native_px'] is None) or math.isclose(distance, item['distance_native_px'], abs_tol=1e-10)
                correct = eligible and distance is not None and distance <= 7
                assert item['eligible'] == eligible and item['correct'] == correct
                if eligible:
                    counts['tp'] += int(correct)
                    counts['fp'] += int(point is not None and not correct)
                    counts['fn'] += int(not correct)
                    if distance is not None:
                        errors.append(distance)
                elif stage == 'raw':
                    excluded.append(dict(id=row['id'], identity=identity, label=item['label']))
            assert all(score[k] == counts[k] for k in counts)
            total = pooled.setdefault(stage, dict(tp=0, fp=0, fn=0, errors=[]))
            for key in counts:
                total[key] += counts[key]
            total['errors'].extend(errors)
        entry = dict(id=row['id'], accepted=row['refinement']['accepted'],
                     initial_valid=row['initial_calibration']['is_valid'])
        after = row['scores']['refined']['landmarks']
        for reference in ('projected', 'original'):
            before = row['scores'][reference]['landmarks']
            entry[reference] = dict(
                gained=[i for i, (a,b) in enumerate(zip(before,after)) if b['correct'] and not a['correct']],
                lost=[i for i, (a,b) in enumerate(zip(before,after)) if a['correct'] and not b['correct']])
        paired.append(entry)
    for stage, total in pooled.items():
        expected = report['pooled'][stage]
        assert all(total[k] == expected[k] for k in ('tp','fp','fn'))
        assert len(total['errors']) == expected['available_errors']
        for key, value in [
            ('precision', total['tp']/(total['tp']+total['fp'])),
            ('recall', total['tp']/(total['tp']+total['fn'])),
            ('mean_native_error_available_px', statistics.mean(total['errors'])),
            ('median_native_error_available_px', statistics.median(total['errors']))]:
            assert math.isclose(expected[key], value, abs_tol=1e-10)
    assert maximum_boundary_residual < 1e-8
    assert maximum_rejected_roundtrip < 1e-10
    required = [row for row in paired if row['projected']['lost'] or len(row['original']['lost']) > len(row['original']['gained'])]
    # Deterministic representative gains: six largest net improvements over original.
    gains = sorted([row for row in paired if len(row['original']['gained']) > len(row['original']['lost'])],
                   key=lambda row: (-(len(row['original']['gained'])-len(row['original']['lost'])), row['id']))[:6]
    visual_rows = required + [row for row in gains if row not in required]
    boards = []
    for start in range(0, len(visual_rows), 4):
        group = visual_rows[start:start+4]
        board = Image.new('RGB', (1280,1568), (20,20,20))
        for i, selection in enumerate(group):
            row = next(x for x in report['images'] if x['id'] == selection['id'])
            image = Image.open(out/row['image'])
            assert image.size == (640,1568)
            # Original refinement and combined refinement, following raw/projection.
            board.paste(image.crop((0,784,640,1568)), ((i%2)*640,(i//2)*784))
        path = out/f'review_board_{start//4+1:02}.jpg'
        board.resize((960,1176)).save(path, quality=67)
        boards.append(dict(path=path.name, sha256=digest(path), ids=[row['id'] for row in group], visually_inspected=False))
    review = dict(complete=True, report_sha256=digest(report_path), qualification_evidence=False,
                  labels_unchanged=True, metrics_independently_replayed=True,
                  maximum_boundary_residual_reference_px=maximum_boundary_residual,
                  maximum_rejected_roundtrip_reference_px=maximum_rejected_roundtrip,
                  excluded_labels=excluded, per_image=paired,
                  paired_counts={reference:{kind:sum(len(row[reference][kind]) for row in paired) for kind in ('gained','lost')} for reference in ('projected','original')},
                  required_regression_review_ids=[row['id'] for row in required],
                  representative_gain_rule='Six largest net gains over original refinement; ties sorted by image ID',
                  representative_gain_ids=[row['id'] for row in gains], visual_boards=boards)
    target = out/'review.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(review,indent=2), encoding='utf-8')
    print(json.dumps(dict(pooled=report['pooled'],paired_counts=review['paired_counts'],
                          invalid=sum(not row['initial_valid'] for row in paired),
                          accepted=sum(row['accepted'] for row in paired),excluded=excluded,
                          required=review['required_regression_review_ids'],gains=review['representative_gain_ids'],boards=boards),indent=2))


if __name__ == '__main__':
    main()
