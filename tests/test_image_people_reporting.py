from src.api.reporting import readable_report


def report_result(evidence):
    return {'summary': {'id': 'match_report_person_evidence', 'label': 'Saved clip',
                        'analyzedAt': '2026-10-01', 'frames': 3, 'fps': 30,
                        'durationSeconds': .1}, 'imagePeople': evidence}


def test_report_counts_observations_without_claiming_participants_or_identity():
    result = report_result({'frames': [{'people': [{}, {}]}, {'people': []}, {'people': None}]})
    html = readable_report(result)
    assert '2 observed boxes across 1 of 3 frames' in html
    assert 'Person evidence is available for 2 frames' in html
    assert 'spectators, officials and false detections' in html
    assert 'persistent person identity are not independently verified' in html
    assert 'Availability is not detection accuracy' in html


def test_legacy_report_does_not_invent_raw_person_absence():
    html = readable_report(report_result(None))
    assert 'Image person detections were not exported' in html
    assert '0 observed boxes' not in html


def test_report_distinguishes_reviewable_empty_frames_from_unavailable_evidence():
    html = readable_report(report_result({'frames': [{'people': []}] * 3}))
    assert '0 observed boxes across 0 of 3 frames' in html
    assert 'Person evidence is available for 3 frames' in html
    assert 'an empty observation list means no valid person boxes were exported' in html
