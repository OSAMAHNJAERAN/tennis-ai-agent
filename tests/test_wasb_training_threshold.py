import numpy as np
import pytest

from scripts.evaluate.select_wasb_training_threshold import infer_thresholds, selection_key, THRESHOLDS
from src.tracking.temporal_ball_tracker import BallObservation


class Detector:
    threshold = .2

    def __init__(self):
        self.calls = 0
        self.decoded = []

    def predict_heatmaps(self, frames):
        self.calls += 1
        # Each output slot has its own source-frame value. Only slot-aligned
        # maps may contribute to the requested target.
        return np.array([np.full((2,2),frame[0,0,0]/10) for frame in frames]),frames[0].shape[:2]

    def decode_heatmaps(self, maps, shape):
        self.decoded.append((self.threshold,maps[0].copy()))
        return [[BallObservation(5.,5.,float(maps[0].max()))]] if maps[0].max() > self.threshold else [[]]


def test_shared_heatmaps_are_redecoded_at_each_threshold_with_target_alignment():
    detector = Detector()
    frames = [np.full((40,64,3),i,np.uint8) for i in range(7)]
    results, windows = infer_thresholds(detector,frames,3,1)
    assert windows == [([1,2,3],2),([2,3,4],1),([3,4,5],0)]
    assert detector.calls == 15  # Three windows in each of five views, shared across thresholds.
    assert len(detector.decoded) == 25
    assert all(np.allclose(hm,.3) for _,hm in detector.decoded)
    assert {t for t,_ in detector.decoded} == set(THRESHOLDS)
    assert results['0.2']['prediction_xy'] is not None
    assert results['0.35']['prediction_xy'] is None
    assert detector.threshold == .2


def test_decode_error_restores_detector_threshold():
    detector = Detector()
    def broken(*args):
        raise RuntimeError('decode failed')
    detector.decode_heatmaps = broken
    with pytest.raises(RuntimeError,match='decode failed'):
        infer_thresholds(detector,[np.zeros((40,64,3),np.uint8)]*3,1,1)
    assert detector.threshold == .2


def test_threshold_selection_uses_f1_before_precision_then_prefers_existing_threshold():
    def score(f1,precision,recall):
        return {'pooled':dict(f1=f1,precision=precision,recall=recall)}
    candidates = [('0.7',score(.89,1,.8)),('0.2',score(.9,.91,.89)),('0.1',score(.9,.91,.89))]
    assert max(candidates,key=selection_key)[0] == '0.2'
    candidates.append(('0.35',score(.9,.92,.88)))
    assert max(candidates,key=selection_key)[0] == '0.35'
