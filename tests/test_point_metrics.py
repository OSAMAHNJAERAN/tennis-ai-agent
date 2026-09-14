import pytest

from src.evaluation.point_metrics import evaluate_points


def row(target, prediction, size=(1920, 1080)):
    return dict(target_xy=target, prediction_xy=prediction, width=size[0], height=size[1])


def test_wrong_point_counts_as_false_positive_and_false_negative():
    result = evaluate_points([row((100, 100), (500, 500))])
    assert (result["false_positives"], result["false_negatives"], result["true_positives"]) == (1, 1, 0)
    assert result["annotated_frames"] == 1


def test_scale_invariance_and_absent_ball_negatives():
    rows = [row((100, 100), (110, 100)), row(None, None), row(None, (50, 50)), row((10, 10), None)]
    result = evaluate_points(rows)
    assert result["precision"] == .5
    assert result["recall"] == .5
    assert result["true_negatives"] == 1
    assert result["annotated_frames"] == 4
    small = row((50, 50), (55, 50), (960, 540))
    assert evaluate_points([small])["localization_errors_reference_px"] == pytest.approx(
        result["localization_errors_reference_px"])


def test_empty_metrics_are_unknown():
    result = evaluate_points([])
    assert result["precision"] is None
    assert result["recall"] is None


def test_bad_point_rejected():
    with pytest.raises(ValueError):
        evaluate_points([row((float("nan"), 2), (3, 4))])


def test_visibility_error_categories_do_not_confuse_wrong_location_with_absence():
    result = evaluate_points([row(None, None), row(None, (2, 3)), row((2, 3), None),
                              row((2, 3), (500, 500)), row((2, 3), (2, 3))])
    assert result['absent_frames'] == 2
    assert result['visible_frames'] == 3
    assert result['absent_false_detections'] == 1
    assert result['wrong_visible_localizations'] == 1
    assert result['visible_abstentions'] == 1
    assert result['absent_specificity'] == .5
    assert result['false_positives'] == 2
    assert result['false_negatives'] == 2
