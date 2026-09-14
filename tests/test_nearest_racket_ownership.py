from src.tracking.nearest_racket_ownership import NearestRacketOwnership, nearest_person_sources
from src.utils.bbox_utils import BBox


def candidate(sources):
    return {'bbox_xyxy': [45, 40, 55, 50], 'confidence': .9, 'sources': set(sources)}


def test_foreground_crop_cannot_claim_distant_players_racket():
    people = {1: BBox(0, 100, 150, 400), 2: BBox(30, 20, 45, 60)}
    result = NearestRacketOwnership(model=object()).assign_candidates(people, [candidate([1, 2])], 0)
    assert result[1]['state'] == 'MISSING'
    assert result[2]['bbox_xyxy'] == candidate([2])['bbox_xyxy']


def test_nearest_non_source_prevents_unsupported_ownership_without_transfer():
    people = {1: BBox(0, 100, 150, 400), 2: BBox(30, 20, 45, 60)}
    result = NearestRacketOwnership(model=object()).assign_candidates(people, [candidate([1])], 0)
    assert all(v['state'] == 'MISSING' for v in result.values())


def test_overlapping_people_remain_ambiguous_and_input_is_preserved():
    people = {1: BBox(0, 0, 60, 60), 2: BBox(40, 30, 90, 100)}
    raw = candidate([1, 2])
    result = nearest_person_sources(people, [raw])
    assert result[0]['sources'] == {1, 2}
    result[0]['sources'].remove(1)
    assert raw['sources'] == {1, 2}


def test_missing_people_and_invalid_candidates_never_create_tracks():
    tracker = NearestRacketOwnership(model=object())
    assert tracker.assign_candidates({}, [candidate([1])], 0) == {}
    assert nearest_person_sources({1: None}, [candidate([1])]) == []
    bad = {**candidate([1]), 'bbox_xyxy': [0, 0, 0, 1]}
    result = tracker.assign_candidates({1: BBox(0, 0, 60, 60)}, [bad], .1)
    assert result[1]['position_px'] is None


def test_order_invariance_and_missing_timestamp_do_not_fill_boxes():
    people = {1: BBox(0, 100, 150, 400), 2: BBox(30, 20, 45, 60)}
    a, b = NearestRacketOwnership(model=object()), NearestRacketOwnership(model=object())
    assert a.assign_candidates(people, [candidate([1, 2])], 0) == b.assign_candidates(dict(reversed(list(people.items()))), [candidate([2, 1])], 0)
    assert all(v['position_px'] is None for v in a.assign_candidates(people, [], .2).values())
