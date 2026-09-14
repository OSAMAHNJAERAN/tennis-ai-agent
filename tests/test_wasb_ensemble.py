from copy import deepcopy

import numpy as np
import pytest

from scripts.evaluate.replay_wasb_ensemble import aligned_rows
from src.detection.wasb_ball_detector import WASBBallDetector, WASBEnsembleDetector


def test_ensemble_stream_matches_average_of_aligned_single_model_streams(monkeypatch):
    def heatmaps(self, frames):
        # Different temporal slots must average using the same frame/window weights.
        return np.array([np.full((2, 2), value * self.scale + slot)
                         for slot, value in enumerate(frames)]), (2, 2)

    monkeypatch.setattr(WASBBallDetector, 'predict_heatmaps', heatmaps)
    first, second = [WASBBallDetector.__new__(WASBBallDetector) for _ in range(2)]
    first.scale, second.scale = 1., 3.
    ensemble = WASBEnsembleDetector.__new__(WASBEnsembleDetector)
    ensemble.scale, ensemble.second, ensemble.second_weight = 1., second, .75
    for length in (1, 2, 3, 7):
        a = list(first.predict_heatmap_stream(range(length)))
        b = list(second.predict_heatmap_stream(range(length)))
        combined = list(ensemble.predict_heatmap_stream(range(length)))
        assert len(combined) == length
        for (left, _), (right, _), (actual, _) in zip(a, b, combined):
            np.testing.assert_allclose(actual, .25 * left + .75 * right)


@pytest.mark.parametrize('field,value', [('frame', 12), ('target_xy', None), ('width', 640)])
def test_cached_ensemble_rejects_different_examples(field, value):
    first = {'clips': [{'clip': 'match_1', 'raw_labeled_rows': [
        {'frame': 10, 'target_xy': [30, 40], 'prediction_xy': None, 'width': 1920, 'height': 1080}]}]}
    second = deepcopy(first)
    second['clips'][0]['raw_labeled_rows'][0][field] = value
    with pytest.raises(ValueError, match='identity or ground truth'):
        aligned_rows(first, second)


def test_cached_ensemble_allows_different_predictions():
    first = {'clips': [{'clip': 'match_1', 'raw_labeled_rows': [
        {'frame': 10, 'target_xy': [30, 40], 'prediction_xy': None}]}]}
    second = deepcopy(first)
    second['clips'][0]['raw_labeled_rows'][0]['prediction_xy'] = [31, 41]
    assert aligned_rows(first, second) == [('match_1', first['clips'][0]['raw_labeled_rows'])]
