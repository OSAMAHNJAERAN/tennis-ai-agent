import numpy as np
import pytest
from scripts.evaluate.temporal_spacing_stream import predict_spaced_tiled_stream
from src.tracking.temporal_ball_tracker import BallObservation


@pytest.mark.parametrize('length', [0,1,2,3,5,6,7,10,11])
@pytest.mark.parametrize('stride', [1,2,3])
def test_continuous_phase_output_preserves_every_native_index(length,stride):
    class Detector:
        def predict_stream(self,frames):
            for frame in frames:
                yield [BallObservation(0.,0.,float(frame[0,0,0]))]
    frames=[np.full((10,10,3),i,np.uint8) for i in range(length)]
    results=list(predict_spaced_tiled_stream(Detector(),iter(frames),(10,10),stride))
    assert len(results)==length
    assert [r[0].confidence for r in results]==list(range(length))


def test_phases_use_spaced_triplets_and_target_aligned_mean():
    class Detector:
        def predict_stream(self,frames):
            values=[int(f[0,0,0]) for f in frames]
            for i,value in enumerate(values):
                windows=[values[start:start+3] for start in range(max(0,i-2),min(i,len(values)-3)+1)]
                assert windows and all(w[1]-w[0]==2 and w[2]-w[1]==2 for w in windows)
                score=sum(sum(w) for w in windows)/len(windows)
                yield [BallObservation(0.,0.,score)]
    frames=[np.full((10,10,3),i,np.uint8) for i in range(10)]
    results=list(predict_spaced_tiled_stream(Detector(),iter(frames),(10,10),2))
    # Original index 4 is aligned to slots 2/1/0 of [0,2,4], [2,4,6], [4,6,8].
    assert results[4][0].confidence==12.
