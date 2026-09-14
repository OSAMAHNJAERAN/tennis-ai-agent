import pytest

from src.tracking.global_racket_tracking import GlobalRacketTracking
from src.utils.bbox_utils import BBox


def proposal(box, score, sources):
    return {'bbox_xyxy': box, 'confidence': score, 'sources': set(sources)}


def test_shared_best_detection_does_not_hide_an_alternative_or_depend_on_player_order():
    players = {1: BBox(0, 0, 20, 100), 2: BBox(80, 0, 100, 100)}
    candidates = [proposal([20, 20, 40, 50], .9, [1, 2]),
                  proposal([21, 20, 41, 50], .8, [2]),
                  proposal([100, 20, 120, 50], .6, [2])]
    first = GlobalRacketTracking(model=object()).assign_candidates(players, candidates, 0)
    second = GlobalRacketTracking(model=object()).assign_candidates(dict(reversed(list(players.items()))), list(reversed(candidates)), 0)
    assert first == second
    assert first[1]['bbox_xyxy'] == [20, 20, 40, 50]
    assert first[2]['bbox_xyxy'] == [100, 20, 120, 50]
    assert first[1]['contact_event'] is first[2]['contact_event'] is None


def test_sources_limit_assignment_and_a_duplicate_cannot_fill_two_players():
    tracker = GlobalRacketTracking(model=object())
    players = {1: BBox(0, 0, 20, 100), 2: BBox(80, 0, 100, 100)}
    result = tracker.assign_candidates(players, [proposal([20, 20, 40, 50], .9, [1]),
                                                  proposal([20, 20, 40, 50], .8, [1])], 0)
    assert result[1]['state'] == 'DETECTED'
    assert result[2]['state'] == 'MISSING'
    assert result[2]['position_px'] is None


def test_temporal_gate_missing_evidence_expiry_and_pixel_velocity():
    tracker = GlobalRacketTracking(model=object())
    players = {1: BBox(0, 0, 20, 100)}
    tracker.assign_candidates(players, [proposal([20, 20, 40, 50], .9, [1])], 0)
    moved = tracker.assign_candidates(players, [proposal([23, 20, 43, 50], .9, [1])], .1)
    assert moved[1]['velocity_px_per_s'] == pytest.approx([30, 0])
    far = proposal([700, 20, 720, 50], .9, [1])
    assert tracker.assign_candidates(players, [far], .15)[1]['state'] == 'MISSING'
    assert tracker.assign_candidates(players, [], .2)[1]['position_px'] is None
    reacquired = tracker.assign_candidates(players, [proposal([20, 20, 40, 50], .9, [1])], .5)
    assert reacquired[1]['velocity_px_per_s'] is None
    with pytest.raises(ValueError, match='increase'):
        tracker.assign_candidates(players, [], .5)


def test_empty_players_and_invalid_detector_evidence_do_not_create_observations():
    tracker = GlobalRacketTracking(model=object())
    assert tracker.assign_candidates({}, [], 0) == {}
    players = {1: BBox(0, 0, 20, 100)}
    bad = [proposal([0, 0, 0, 0], .9, [1]), proposal([0, 0, 10, 10], float('nan'), [1]),
           proposal([0, 0, 10, 10], .1, [1])]
    assert tracker.assign_candidates(players, bad, 0)[1]['state'] == 'MISSING'
