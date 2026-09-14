import pytest

from scripts.data.prepare_racket_scale_pilot import anchor_crop


@pytest.mark.parametrize('box', [[0,0,8,20], [620,330,640,360], [290,160,350,180]])
@pytest.mark.parametrize('factor', [1.5,4.])
@pytest.mark.parametrize('offset', [-.1,.1])
def test_scale_crop_preserves_anchor_at_edges(box,factor,offset):
    left,top,right,bottom = anchor_crop(box,640,360,factor,offset,-offset)
    assert 0 <= left <= box[0] < box[2] <= right <= 640
    assert 0 <= top <= box[1] < box[3] <= bottom <= 360
    assert right-left == bottom-top
