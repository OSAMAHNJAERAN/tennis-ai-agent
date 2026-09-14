"""Verify saved controlled-cut evidence without rerunning registration."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.court.court_geometry import TennisCourtGeometry


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('outputs/vision_upgrade_audit/returning_view_registration01')
    report = json.loads((out / 'report.json').read_text())
    assert report['complete'] and not report['qualification_evidence']
    for path, expected in {**report['source_hashes'], **report['code_hashes']}.items():
        assert digest(path) == expected, path
    assert digest('docs/experiments/COURT_RETURNING_VIEW_PROTOCOL.md') == report['protocol_sha256']
    for path, expected in report['image_hashes'].items():
        assert digest(out / path) == expected, path
    cases = {case['name']: case for case in report['cases']}
    for name, case in cases.items():
        entries = case['entries']
        assert len(entries) == case['frames']
        for key in ('original', 'recovery'):
            assert sum(e[key]['is_valid'] for e in entries) == case[key + '_valid']
        restored = [e['frame'] for e in entries if e['recovery']['recovered_this_frame']]
        assert restored == case['recovered_frames']
        assert not [e for e in entries if e['source'] != 'match148' and e['recovery']['is_valid']]
        for index, entry in enumerate(entries):
            state = entry['recovery']
            assert state['frame_index'] == index
            if not state['is_valid']:
                assert state['image_to_court'] is None and state['anchor_to_frame_px'] is None
            if index < 60 or name == 'healthy':
                assert entry['original']['image_to_court'] == state['image_to_court']
            if index in restored:
                count = state['required_confirmation_frames']
                streak = entries[index-count+1:index+1]
                assert len(streak) == count
                assert all(e['source'] == 'match148' and e['recovery']['recovery_attempted'] for e in streak)
                assert [e['recovery']['confirmation_count'] for e in streak] == list(range(1, count+1))
        expected = [96] if name in ('unrelated_cut_return', 'blank_cut_return') else []
        assert restored == expected
    canonical = TennisCourtGeometry.get_canonical_keypoints()
    errors = []
    for name in ('unrelated_cut_return', 'blank_cut_return'):
        for entry in cases[name]['entries'][90:]:
            state = entry['recovery']
            if not state['is_valid']:
                continue
            reference = np.array(cases['healthy']['entries'][entry['source_frame']]['recovery']['image_to_court'])
            image = np.column_stack([canonical, np.ones(len(canonical))]) @ np.linalg.inv(reference).T
            image /= image[:, 2:3]
            ground = image @ np.array(state['image_to_court']).T
            ground = ground[:, :2] / ground[:, 2:3]
            errors.append(float(np.linalg.norm(ground-canonical, axis=1).max()))
    assert abs(max(errors)-report['maximum_returned_vs_healthy_landmark_ground_difference_m']) < 1e-5
    cap = cv2.VideoCapture(str(out / 'cut_return_comparison.mp4'))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 60
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (400, 1280, 3)
        count += 1
    cap.release()
    assert count == 150
    review = dict(complete=True, independently_verified_cases=len(cases), decoded_video_frames=count,
                  report_sha256=digest(out/'report.json'), video_sha256=digest(out/'cut_return_comparison.mp4'),
                  reviewer_sha256=digest(__file__), maximum_agreement_error_m=max(errors),
                  visual_review={'inspected': ['frame_060.jpg', 'frame_095.jpg', 'frame_119.jpg'],
                                 'notes': 'Different court withheld; original-view confirmation pending; restored landmarks align visually with original court lines.',
                                 'scope': 'Visual sanity check only; not independent physical court accuracy'},
                  production_qualified=False)
    (out/'completion_review.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review))


if __name__ == '__main__':
    main()
