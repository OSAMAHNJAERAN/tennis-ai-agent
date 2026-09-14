import pytest

from src.tracking.anchored_ball_recovery import recover_anchored_gaps
from src.tracking.temporal_ball_tracker import BallObservation


def test_recovery_exports_real_candidate_instead_of_linear_guide():
    left, right = BallObservation(10, 20, .9), BallObservation(30, 20, .9)
    observed = BallObservation(22, 21, .1)
    strong = [left, None, right]
    result, audit = recover_anchored_gaps(strong, [[], [observed], []], 30, (512, 288))
    assert result == [left, observed, right]
    assert result[1] is observed
    assert strong[1] is None
    assert audit[0]['lookahead_frames'] == 1


@pytest.mark.parametrize('weak', [[], [BallObservation(200, 200, .1)]])
def test_no_plausible_observation_means_no_filled_position(weak):
    result, audit = recover_anchored_gaps([BallObservation(10, 20, .9), None, BallObservation(30, 20, .9)],
                                         [[], weak, []], 30, (512, 288))
    assert result[1] is None
    assert audit == []


@pytest.mark.parametrize('fps,confidence,end_x', [(5, .9, 30), (30, .3, 30), (30, .9, 10)])
def test_long_gaps_weak_anchors_and_stationary_context_do_not_recover(fps, confidence, end_x):
    result, audit = recover_anchored_gaps([BallObservation(10, 20, confidence), None,
                                          BallObservation(end_x, 20, confidence)],
                                         [[], [BallObservation(20, 20, .1)], []], fps, (512, 288))
    assert result[1] is None
    assert not audit


def test_mismatched_sequences_are_rejected():
    with pytest.raises(ValueError):
        recover_anchored_gaps([None], [], 30, (512, 288))
