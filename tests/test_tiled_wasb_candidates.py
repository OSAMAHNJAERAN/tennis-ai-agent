import numpy as np
import pytest

from src.detection.tiled_wasb_candidates import overlapping_tiles, sparse_tile_candidates, merge_by_confidence
from src.tracking.temporal_ball_tracker import BallObservation
from src.detection.tiled_wasb_candidates import predict_tiled_stream
from src.detection.wasb_ball_detector import WASBBallDetector


def test_tiles_cover_all_edges_with_overlap():
    assert overlapping_tiles(100, 60) == [(0, 0, 60, 36), (40, 0, 100, 36), (0, 24, 60, 60), (40, 24, 100, 60)]
    with pytest.raises(ValueError):
        overlapping_tiles(100, 60, .4)


def test_tiled_temporal_windows_recover_source_coordinates_and_correct_output_slot():
    class Model:
        def predict_heatmaps(self, frames):
            return np.array([np.full((1, 1), frame[0, 0, 0], dtype=float) for frame in frames]), frames[0].shape[:2]

        def decode_heatmaps(self, heatmaps, shape):
            return [[BallObservation(2, 3, float(value[0, 0]))] for value in heatmaps]
    frames = [np.full((60, 100, 3), index, dtype=np.uint8) for index in range(5)]
    result = sparse_tile_candidates(Model(), frames, 2)
    assert [(item['observation'].x_px, item['observation'].y_px) for item in result] == [(2, 3), (42, 3), (2, 27), (42, 27)]
    assert all(item['observation'].confidence == 2 for item in result)


def test_confidence_merge_never_averages_or_invents_a_position():
    weak, strong, far = BallObservation(10, 10, .2), BallObservation(12, 10, .9), BallObservation(100, 100, .5)
    assert merge_by_confidence([weak, strong, far], (512, 288)) == [strong, far]
    assert merge_by_confidence([], (512, 288)) == []


@pytest.mark.parametrize('count', [0, 1, 2, 3, 7])
def test_continuous_views_match_sparse_real_windows_at_boundaries(count):
    class Model:
        predict_stream = WASBBallDetector.predict_stream
        predict_heatmap_stream = WASBBallDetector.predict_heatmap_stream

        def predict_heatmaps(self, frames):
            # Context-dependent value catches wrong output slots/window averaging.
            context = sum(float(frame[0, 0, 0]) for frame in frames)
            return np.array([np.full((1, 1), context + i) for i in range(len(frames))]), frames[0].shape[:2]

        def decode_heatmaps(self, heatmaps, shape):
            return [[BallObservation(2, 3, float(value[0, 0]))] for value in heatmaps]

        def predict_triplet(self, frames):
            maps, shape = self.predict_heatmaps(frames)
            return self.decode_heatmaps(maps, shape)

    frames = [np.full((60, 100, 3), index, dtype=np.uint8) for index in range(count)]
    model = Model()
    full = list(model.predict_stream(iter(frames)))
    actual = list(predict_tiled_stream(model, iter(frames), (100, 60)))
    assert len(actual) == count
    for index, values in enumerate(actual):
        sparse = [item['observation'] for item in sparse_tile_candidates(model, frames, index)]
        assert values == merge_by_confidence(full[index] + sparse, (100, 60))
