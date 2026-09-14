import pytest

from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.tracking.temporal_ball_tracker import BallObservation


def test_persistent_distractor_removed_without_inventing_ball_or_reordering_moving_candidates():
    gate = StationaryCandidateFilter()
    for frame in range(30):
        logo = BallObservation(60, 220, .9)
        ball = BallObservation(150 + frame * 4, 100, .8)
        kept, removed = gate.filter([logo, ball], frame / 30, 30, (512, 288))
    assert kept == [ball]
    assert removed == [logo]
    kept, _ = gate.filter([logo], 1., 30, (512, 288))
    assert kept == []


def test_sparse_recurrences_do_not_count_as_continuous_stationary_evidence():
    gate = StationaryCandidateFilter()
    for frame in range(90):
        candidates = [BallObservation(60, 220, .9)] if frame % 20 == 0 else []
        kept, removed = gate.filter(candidates, frame / 30, 30, (512, 288))
        assert not removed
        assert kept == candidates


def test_long_gap_resets_history_and_repeated_timestamp_is_rejected():
    gate = StationaryCandidateFilter()
    ball = BallObservation(20, 30, .9)
    for frame in range(30):
        gate.filter([ball], frame / 30, 30, (512, 288))
    assert gate.filter([ball], 5., 30, (512, 288))[0] == [ball]
    with pytest.raises(ValueError, match='increasing timestamps'):
        gate.filter([ball], 5., 30, (512, 288))
