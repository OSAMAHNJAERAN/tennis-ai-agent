import numpy as np
import pytest

from src.detection.ball_candidate_verifier import labeled_frame_candidates, candidate_patch, candidate_features
from src.tracking.temporal_ball_tracker import BallObservation
from scripts.train.train_ball_candidate_verifier import score_rows, training_partition, TrainingCandidates


class FrameDetector:
    def predict_heatmaps(self, frames):
        start = frames[0]
        return [np.array([[start * 10 + offset]], dtype=np.float32) for offset in range(3)], (1, 1)

    def decode_heatmaps(self, heatmaps, shape):
        return [[float(heatmap[0, 0])] for heatmap in heatmaps]

    def predict_triplet(self, frames):
        return [[frame] for frame in frames]


def test_sparse_supervision_uses_only_real_overlapping_windows_and_correct_output_slot():
    detector = FrameDetector()
    assert labeled_frame_candidates(detector, list(range(5)), 0) == [0.]
    assert labeled_frame_candidates(detector, list(range(5)), 2) == [11.]
    assert labeled_frame_candidates(detector, list(range(5)), 4) == [22.]
    assert labeled_frame_candidates(detector, [0, 1], 1) == [1]
    with pytest.raises(ValueError):
        labeled_frame_candidates(detector, [0], 1)


def test_candidate_crop_is_centered_at_observation_and_keeps_current_past_channels():
    current = np.zeros((540, 960, 3), dtype=np.uint8)
    past = current.copy()
    current[200, 100] = [255, 0, 0]
    past[200, 100] = [0, 255, 0]
    candidate = BallObservation(200, 400, .2)
    patch = candidate_patch(current, past, candidate, (1920, 1080))
    assert patch.shape == (6, 32, 32)
    assert patch[:, 16, 16].tolist() == [255, 0, 0, 0, 255, 0]
    assert candidate_features(candidate, 1).tolist() == pytest.approx([.2, .5])


def test_border_candidates_keep_fixed_crop_dimensions():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    for x, y in ((0, 0), (1919, 1079)):
        assert candidate_patch(frame, frame, BallObservation(x, y, .9), (1920, 1080)).shape == (6, 32, 32)


def test_training_partition_refuses_validation_and_keeps_selection_clips_disjoint():
    with pytest.raises(ValueError, match='refuses'):
        training_partition({'split': 'VALIDATION_ONLY'})
    training, selection = training_partition({'split': 'TRAINING_ONLY', 'rows': [],
                                             'selected_clips': [[str(i), '000'] for i in range(20)]})
    assert len(training) == 16 and len(selection) == 4
    assert not set(training) & set(selection)


def test_candidate_selection_scores_real_observations_and_explicit_absence():
    row = {'clip': 'a', 'frame': 0, 'width': 512, 'height': 288, 'target_xy': [10, 20],
           'candidates': [{'sample_index': 0, 'x': 100, 'y': 100}, {'sample_index': 1, 'x': 11, 'y': 21}]}
    metrics, predictions = score_rows([row, {**row, 'frame': 1, 'target_xy': None}], np.array([.1, .9]), .5)
    assert metrics['true_positives'] == 1 and metrics['absent_false_detections'] == 1
    assert predictions[0]['prediction_xy'] == [11, 21]
    metrics, _ = score_rows([{**row, 'candidates': []}], np.array([]), .5)
    assert metrics['visible_abstentions'] == 1


def test_motion_blur_is_shared_by_current_and_past_without_relabeling():
    patches = np.zeros((1, 6, 32, 32), dtype=np.uint8)
    patches[0, :, 16, 16] = 255
    features = np.array([[.2, 1.]], dtype=np.float32)
    dataset = TrainingCandidates(patches, features, np.array([1]), np.array([0]), motion_blur_probability=1.)
    image, metadata, label = dataset[0]
    assert np.array_equal(image[:3].numpy(), image[3:].numpy())
    assert image.shape == (6, 32, 32)
    assert metadata.tolist() == pytest.approx([.2, 1.])
    assert label == 1
    assert patches[0, 0, 16, 16] == 255
