import numpy as np
import pytest

from scripts.evaluate.benchmark_ball_temporal_spacing import infer_candidates
from scripts.evaluate.benchmark_wasb_selected_threshold import infer_pair, same_point
from src.tracking.temporal_ball_tracker import BallObservation


class Detector:
    threshold = .2

    def __init__(self):
        self.calls = 0

    def predict_heatmaps(self, frames):
        self.calls += 1
        return np.array([np.full((2, 2), f[0, 0, 0] / 10) for f in frames]), frames[0].shape[:2]

    def decode_heatmaps(self, maps, shape):
        # Threshold changes localization as well as candidate presence.
        return [[BallObservation(5 + self.threshold, 6, float(maps[0].max()))]] if maps[0].max() > self.threshold else [[]]


@pytest.mark.parametrize('length,target,stride', [(7, 3, 1), (9, 4, 2), (7, 0, 2), (7, 6, 2)])
def test_paired_shared_forward_passes_match_independent_inference(length, target, stride):
    frames = [np.full((40, 64, 3), i, np.uint8) for i in range(length)]
    detector = Detector()
    pair, windows = infer_pair(detector, frames, target, stride, .35)
    assert detector.calls == 5 * len(windows)
    assert detector.threshold == .2
    for name, threshold in [('control', .2), ('selected', .35)]:
        reference = Detector()
        reference.threshold = threshold
        merged, candidates, independent_windows = infer_candidates(reference, frames, target, stride)
        assert candidates == pair[name]['candidates']
        assert independent_windows == windows
        assert pair[name]['prediction_xy'] == ([merged[0].x_px, merged[0].y_px] if merged else None)


def test_pair_restores_threshold_on_decode_error():
    detector = Detector()
    def fail(*args):
        raise RuntimeError('Decode failed')
    detector.decode_heatmaps = fail
    with pytest.raises(RuntimeError, match='Decode failed'):
        infer_pair(detector, [np.zeros((40, 64, 3), np.uint8)] * 3, 1, 1, .35)
    assert detector.threshold == .2


def test_replay_requires_matching_absence_and_subpixel_tolerance():
    assert same_point(None, None)
    assert not same_point(None, [0, 0])
    assert not same_point([0, 0], None)
    assert same_point([1, 2], [1.00001, 2])
    assert not same_point([1, 2], [1.001, 2])
