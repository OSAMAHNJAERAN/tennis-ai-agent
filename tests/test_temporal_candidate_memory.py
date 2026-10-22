"""A large indexed candidate source must not be copied into a full frame list."""
from collections.abc import Sequence
import weakref

import pytest

from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallTracker


class ObservedFrames(Sequence):
    def __init__(self, count):
        self.count = count
        self.live = self.peak = self.reads = 0

    def __len__(self):
        return self.count

    def _released(self):
        self.live -= 1

    def __getitem__(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        self.reads += 1
        candidate = BallObservation(20 + index * .01, 30., .9)
        self.live += 1
        self.peak = max(self.peak, self.live)
        weakref.finalize(candidate, self._released)
        return [candidate]


@pytest.mark.parametrize('count', [32, 5000])
def test_tracker_preserves_observations_without_retaining_whole_candidate_source(count):
    frames = ObservedFrames(count)
    trajectory = TemporalBallTracker(enable_multi_candidate_association=False).track_video_candidates(
        frames, fps=60, frame_size=(1920, 1080))
    assert len(trajectory) == count
    assert all(point.frame_index == i and point.state is BallState.DETECTED and point.confidence == .9
               and point.x_px == 20 + i * .01 and point.y_px == 30.
               for i, point in enumerate(trajectory))
    # The result retains selected scalars, not its source candidate objects.
    # This independently observes object lifetimes, including neighbor reads.
    assert frames.peak < 32
    assert frames.live == 0
    assert frames.reads >= count


def test_bounds_filter_keeps_order_and_existing_missing_transition_semantics():
    frames = [[BallObservation(-1, 10, .99), BallObservation(20, 30, .2), BallObservation(21, 30, .9)],
              [BallObservation(float('nan'), 20, .9), BallObservation(30, 40, .3)],
              [BallObservation(1920, 10, .99), BallObservation(40, 50, .8)],
              [BallObservation(20, float('inf'), .99)]]
    tracker = TemporalBallTracker(enable_multi_candidate_association=False, max_prediction_gap=0,
                                  max_interpolation_gap=0, enable_short_gap_reacquisition=False)
    trajectory = tracker.track_video_candidates(frames, fps=60, frame_size=(1920, 1080))
    assert [(point.x_px, point.y_px, point.confidence, point.state) for point in trajectory] == [
        (21., 30., .9, BallState.DETECTED), (30., 40., .3, BallState.DETECTED),
        (40., 50., .8, BallState.DETECTED), (None, None, None, BallState.MISSING)]
    assert len(frames[0]) == 3 and frames[0][0].x_px == -1
