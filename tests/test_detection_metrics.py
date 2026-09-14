import pytest

from src.evaluation.detection_metrics import aggregate_counts, match_detections


def test_wrong_location_is_both_false_positive_and_false_negative():
    result = match_detections([[20, 20, 30, 30]], [[0, 0, 10, 10]]).to_dict()
    assert (result["true_positives"], result["false_positives"], result["false_negatives"]) == (0, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0
    assert result["mean_matched_iou"] is None


def test_duplicate_detections_are_not_double_counted():
    result = match_detections([[0, 0, 10, 10]] * 2, [[0, 0, 10, 10]]).to_dict()
    assert result["precision"] == .5
    assert result["recall"] == 1
    assert result["f1"] == pytest.approx(2 / 3)


def test_matching_maximizes_cardinality_in_ambiguous_scene():
    # Greedily assigning the first prediction to GT0 would strand pred1.
    result = match_detections([[1, 0, 11, 10], [0, 0, 8, 10]],
                              [[0, 0, 10, 10], [4, 0, 14, 10]], iou_threshold=.5)
    assert result.true_positives == 2


def test_empty_and_negative_images_preserve_undefined_denominators():
    assert match_detections([], []).to_dict()["precision"] is None
    negative = match_detections([[0, 0, 10, 10]], [])
    positive = match_detections([], [[0, 0, 10, 10]])
    result = aggregate_counts([negative, positive]).to_dict()
    assert (result["false_positives"], result["false_negatives"]) == (1, 1)
    assert result["precision"] == result["recall"] == 0


@pytest.mark.parametrize("boxes", [[[0, 0, 0, 1]], [[0, 1, 1, 0]], [[0, 0, float('nan'), 1]], [[0, 1]]])
def test_invalid_boxes_fail_loudly(boxes):
    with pytest.raises(ValueError):
        match_detections(boxes, [])


@pytest.mark.parametrize("threshold", [0, -1, 2, float('nan')])
def test_invalid_iou_threshold_fails(threshold):
    with pytest.raises(ValueError):
        match_detections([], [], iou_threshold=threshold)
