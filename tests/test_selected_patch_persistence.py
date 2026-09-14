import copy

import numpy as np
import pytest

from src.tracking.selected_patch_persistence import SelectedPatchPersistence, validate_spatial_candidate_config
from src.tracking.temporal_ball_tracker import BallObservation


def test_stream_gate_preserves_startup_and_rejects_repeated_texture():
    frame=np.random.default_rng(7).integers(30,150,(64,64,3),dtype=np.uint8)
    point=BallObservation(32,32,.9)
    gate=SelectedPatchPersistence(reference_size=(64,64))
    assert gate.filter(frame,point,0)[0] is point
    selected,evidence=gate.filter(frame,point,.1)
    assert selected is None and evidence['rejected']
    assert evidence['similarity'] == pytest.approx(1.)
    assert evidence['local_residual'] == pytest.approx(0.)
    assert gate.filter(frame,point,1.)[0] is point


def test_local_change_is_retained_and_unknown_points_still_advance_history():
    frame=np.random.default_rng(17).integers(20,100,(64,64,3),dtype=np.uint8)
    point=BallObservation(32,32,.9)
    gate=SelectedPatchPersistence(reference_size=(64,64))
    assert gate.filter(frame,None,0)[0] is None
    changed=frame.copy()
    changed[32,31:34] += 100
    assert gate.filter(changed,point,.1)[0] is point
    with pytest.raises(ValueError):
        gate.filter(changed,point,.1)


def test_unmeasured_tracking_or_authority_combinations_are_rejected():
    config={'ball_detection':{'backend':'wasb','temporal_step':1,'spatial_crops':{'enabled':True}},
            'temporal_tracking':{'strategy':'model_top1'},'event_detection':{'authoritative_enabled':False}}
    validate_spatial_candidate_config(config)
    for section,key,value in [('ball_detection','backend','yolo11'),('ball_detection','temporal_step',3),
                               ('temporal_tracking','strategy','kalman'),('event_detection','authoritative_enabled',True)]:
        invalid=copy.deepcopy(config)
        invalid[section][key]=value
        with pytest.raises(ValueError):
            validate_spatial_candidate_config(invalid)
    validate_spatial_candidate_config({})
