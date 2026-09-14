import pytest

from scripts.evaluate.compare_tiled_ball_stream import verify_sparse_control


def row(prediction):
    return {'clip': 'clip1', 'frame': 1, 'width': 512, 'height': 288,
            'target_xy': [10., 10.], 'prediction_xy': prediction}


def test_raw_control_checks_coordinates_even_when_outcome_matches():
    with pytest.raises(ValueError, match='coordinates differ'):
        verify_sparse_control([row([10., 10.])], [row([11., 10.])])
    result = verify_sparse_control([row([10., 10.])], [row([10.0001, 10.])])
    assert result['matching_labels'] == 1
    assert result['maximum_coordinate_difference_px'] == pytest.approx(.0001)


def test_raw_control_rejects_changed_missing_state_and_label_duplication():
    with pytest.raises(ValueError, match='outcomes differ'):
        verify_sparse_control([row(None)], [row([10., 10.])])
    with pytest.raises(ValueError, match='Duplicate'):
        verify_sparse_control([row(None), row(None)], [row(None)])


def test_raw_control_requires_same_dimensions_and_labels():
    old = row([10., 10.])
    resized = {**old, 'width': 1024, 'height': 576,
               'target_xy': [20., 20.], 'prediction_xy': [20., 20.]}
    with pytest.raises(ValueError, match='dimensions differ'):
        verify_sparse_control([old], [resized])
    with pytest.raises(ValueError, match='different explicit labels'):
        verify_sparse_control([old], [{**old, 'frame': 2}])
