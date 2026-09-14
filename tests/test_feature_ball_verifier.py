import numpy as np
import pytest
import torch
from src.detection.feature_ball_verifier import context_patch, FeatureBallVerifier


def test_center_and_scaling():
    rgb = np.zeros((540,960,3),np.uint8)
    rgb[200,300] = [21,42,63]
    a = context_patch(rgb,[600,400],[1920,1080])
    assert a.shape == (3,64,64)
    assert a[:,32,32].tolist() == [21,42,63]
    assert np.array_equal(a,context_patch(rgb,[300,200],[960,540]))


def test_reflected_border():
    rgb = np.zeros((540,960,3),np.uint8)
    rgb[1,1] = 255
    a = context_patch(rgb,[0,0],[960,540])
    assert a[:,31,31].tolist() == a[:,33,33].tolist() == [255]*3


@pytest.mark.parametrize('point',[[float('nan'),1],[-1,1],[960,1]])
def test_invalid_point(point):
    with pytest.raises(ValueError):
        context_patch(np.zeros((540,960,3),np.uint8),point,[960,540])


def test_one_finite_score_per_candidate():
    model = FeatureBallVerifier().eval()
    with torch.inference_mode():
        scores = model(torch.zeros(3,6144),torch.tensor([[.2,1.],[.8,.5],[.5,.3]]))
    assert scores.shape == (3,) and torch.isfinite(scores).all()
