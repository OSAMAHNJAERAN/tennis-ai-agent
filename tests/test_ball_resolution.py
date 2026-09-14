import pytest

from scripts.evaluate.benchmark_ball_resolution import scaled_geometry
from src.evaluation.point_metrics import evaluate_points
from scripts.evaluate.compare_ball_resolution import compare


@pytest.mark.parametrize('height,expected_width', [(1080, 1920), (720, 1280), (480, 853)])
def test_resolution_geometry_and_matching_tolerance_scale_together(height, expected_width):
    width, result_height = scaled_geometry(1920, 1080, height)
    assert (width, result_height) == (expected_width, height)
    row = {'width': width, 'height': height, 'target_xy': [width / 2, height / 2],
           'prediction_xy': [width / 2 + 3 * width / 512, height / 2]}
    assert evaluate_points([row])['true_positives'] == 1
    row['prediction_xy'][0] = width / 2 + 5 * width / 512
    assert evaluate_points([row])['false_negatives'] == 1


def test_upscale_is_not_counted_as_a_new_source_resolution():
    with pytest.raises(ValueError, match='downsampling'):
        scaled_geometry(1920, 1080, 2160)


def test_paired_comparison_preserves_normalized_targets_and_counts_real_outcome_changes():
    original = {'clip': 'a', 'frame': 1, 'width': 1920, 'height': 1080,
                'target_xy': [960, 540], 'prediction_xy': [960, 540]}
    smaller = {**original, 'width': 1280, 'height': 720, 'target_xy': [640, 360], 'prediction_xy': None}
    result = compare([original], [smaller])
    assert result['lost_correct_visible_frames'] == 1
    assert result['new_correct_visible_frames'] == 0
    smaller['target_xy'] = [650, 360]
    with pytest.raises(ValueError, match='scaled'):
        compare([original], [smaller])
