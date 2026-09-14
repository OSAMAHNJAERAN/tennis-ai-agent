"""Trace missing boundary evidence through frozen ridge and matching gates."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.probe_court_line_segments import propose
from scripts.evaluate.refine_court_line_geometry import LINES, match_lines
from scripts.evaluate.benchmark_court_line_refinement import digest


def compatibility(points, segment, identity):
    a, b = LINES[identity]
    start, end = points[[a, b]]
    first, last = np.array(segment).reshape(2, 2)
    length = float(np.linalg.norm(end-start))
    size = float(np.linalg.norm(last-first))
    if min(length, size) < 1e-10:
        return dict(compatible=False, degenerate=True)
    direction = (end-start)/length
    tangent = (last-first)/size
    angle = float(np.degrees(np.arccos(np.clip(abs(tangent@direction), 0, 1))))
    along = sorted([(first-start)@direction, (last-start)@direction])
    overlap = float(max(0, min(length, along[1])-max(0, along[0])))
    normal = np.array([-tangent[1], tangent[0]])
    distance = np.abs((np.stack((start, end))-first)@normal)
    return dict(length=size, angle=angle, overlap=overlap,
                max_endpoint_distance=float(distance.max()),
                compatible=bool(length >= 30 and size >= 30 and angle <= 6
                                and overlap >= min(length, size)*.5
                                and overlap >= 30 and distance.max() <= 20))


def main():
    out = ROOT/'outputs/vision_upgrade_audit/court_missing_boundary_diagnosis01'
    if out.exists():
        raise FileExistsError(out)
    source = ROOT/'outputs/vision_upgrade_audit/court_refinement_selection01/report.json'
    report_source = json.loads(source.read_text())
    assert report_source['complete']
    for name, checksum in report_source['code_hashes'].items():
        assert digest(ROOT/name) == checksum
    out.mkdir(parents=True)
    report = dict(complete=False,
                  scope='FOUR_REUSED_DIAGNOSIS_CASES; NO_THRESHOLD_CHANGES',
                  source_sha256=digest(source),
                  code_hashes={p: digest(ROOT/p) for p in [
                      'scripts/evaluate/diagnose_missing_court_boundaries.py',
                      'scripts/evaluate/probe_court_line_segments.py',
                      'scripts/evaluate/refine_court_line_geometry.py']}, cases=[])
    for identity, line_id in [('5QObSWGBQB8_1200', 0), ('ktiDOhLZIVs_3500', 1),
                              ('6CRu9DY7KII_800', 0), ('6CRu9DY7KII_850', 0)]:
        row = next(x for x in report_source['images'] if x['id'] == identity)
        path = ROOT/'data/external/court_heatmap_pilot/images'/f'{identity}.png'
        assert digest(path) == row['input_sha256']
        image = cv2.resize(cv2.imread(str(path)), (960, 540))
        points = np.array([p['prediction'] for p in row['scores']['projected']['landmarks']])*.75
        raw = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
        raw_rows = [dict(segment=s.tolist(), **compatibility(points, s, line_id)) for s in raw[:, 0]] if raw is not None else []
        proposals = propose(image)
        diagnosed = [dict(proposal=p, compatibility=compatibility(points, p['segment'], line_id)) for p in proposals]
        compatible = [x for x in diagnosed if x['compatibility']['compatible']]
        retained = [x for x in compatible if x['proposal']['support_fraction'] >= .5]
        selected = [p['segment'] for p in proposals if p['support_fraction'] >= .5]
        initial_matches = match_lines(points, np.array(selected).reshape(-1, 2, 2))
        near = [x for x in diagnosed if x['compatibility']['angle'] <= 6
                and x['compatibility']['max_endpoint_distance'] <= 20
                and x['compatibility']['overlap'] > 0]
        entry = dict(id=identity, boundary=line_id, input_sha256=digest(path),
                     initial_points=points.tolist(), raw_lsd_count=len(raw_rows),
                     raw_compatible=[x for x in raw_rows if x['compatible']],
                     compatible_proposals=compatible, retained_compatible_count=len(retained),
                     initial_matches=initial_matches,
                     final_matches=row['refinement']['iterations'][-1]['matches'], near_candidates=near)
        panels = []
        for title, show in [('All near-line ridge proposals', near),
                            ('Retained near-line proposals', [x for x in near if x['proposal']['support_fraction'] >= .5])]:
            panel = image.copy()
            a, b = LINES[line_id]
            cv2.line(panel, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (0, 0, 255), 1)
            for i, item in enumerate(show):
                segment = np.rint(item['proposal']['segment']).astype(int)
                cv2.line(panel, tuple(segment[:2]), tuple(segment[2:]), (255, 255, 0), 2)
                cv2.putText(panel, f"{i}:{item['proposal']['support_fraction']:.2f}", tuple((segment[:2]+segment[2:])//2), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 255, 255), 1)
            cv2.rectangle(panel, (0, 0), (960, 30), (20, 20, 20), -1)
            cv2.putText(panel, f'{identity} boundary {line_id}: {title}', (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            panels.append(panel)
        target = out/f'{identity}.jpg'
        assert cv2.imwrite(str(target), np.concatenate(panels), [cv2.IMWRITE_JPEG_QUALITY, 70])
        entry.update(image=target.name, image_sha256=digest(target))
        report['cases'].append(entry)
        print(json.dumps(dict(id=identity, raw_compatible=len(entry['raw_compatible']),
                              compatible=len(compatible), retained=len(retained),
                              near=[dict(support=x['proposal']['support_fraction'],
                                         bright=x['proposal']['bright_fraction'], white=x['proposal']['white_fraction'],
                                         sides=x['proposal']['similar_sides_fraction'], geometry=x['compatibility']) for x in near])), flush=True)
    report['complete'] = True
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
