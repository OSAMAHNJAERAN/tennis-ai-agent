import numpy as np
import pytest
from scripts.evaluate.benchmark_ball_temporal_spacing import aligned_windows, infer_candidates


def test_every_output_slot_identifies_exact_native_target():
    for length in range(6,15):
        for stride in (1,2):
            for target in range(length):
                for indices,slot in aligned_windows(length,target,stride):
                    assert indices[slot]==target
                    assert indices==list(range(indices[0],indices[0]+3*stride,stride))
                    assert min(indices)>=0 and max(indices)<length


def test_edges_use_only_available_real_triplets():
    assert aligned_windows(10,0,2)==[([0,2,4],0)]
    assert aligned_windows(10,9,2)==[([5,7,9],2)]
    assert aligned_windows(10,1,2)==[([1,3,5],0)]
    with pytest.raises(ValueError): aligned_windows(3,1,2)


def test_nonconsecutive_native_frames_do_not_shift_output_slot():
    class Detector:
        def predict_heatmaps(self,frames):
            return np.array([np.full((2,2),f[0,0,0]) for f in frames]),frames[0].shape[:2]
        def decode_heatmaps(self,maps,shape):
            # Each input frame encodes its native ID. Every view's mean must
            # contain the target ID, never the adjacent frame or output index.
            assert np.array_equal(maps[0],np.full((2,2),5))
            return [[]]
    frames=[np.full((10,10,3),i,np.uint8) for i in range(11)]
    selected,candidates,windows=infer_candidates(Detector(),frames,5,2)
    assert not selected and not candidates
    assert windows==[([1,3,5],2),([3,5,7],1),([5,7,9],0)]
