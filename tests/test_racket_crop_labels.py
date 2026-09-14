import pytest

from scripts.data.prepare_racket_crop_pilot import crop_labels


def test_crop_transform_clips_intersections_and_excludes_outside_boxes():
    result = crop_labels([[110, 60, 130, 100], [90, 90, 120, 130], [0, 0, 10, 10]], [100, 50, 200, 150])
    assert len(result) == 2
    assert result[0] == pytest.approx([.2, .3, .2, .4])
    assert result[1] == pytest.approx([.1, .6, .2, .4])
