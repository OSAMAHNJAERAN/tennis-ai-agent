"""Sampling preserves explicit provenance and never creates accepted labels."""

import numpy as np

from scripts.data.prepare_ball_review_expansion import allocate, closest_index, evenly_spaced, hypothesized_box, select_positive


def test_native_pts_context_is_nearest_actual_frame_without_manufacturing_time():
    assert closest_index([0,.04,.08,.12],.10)==3
    assert closest_index([0,.04,.08,.12],-.1)==0
    assert closest_index([0,.04,.08,.12],.8)==3


def test_frame_sampling_spreads_coverage_and_deduplicates_overlapping_strata():
    pool=[{"clip":"m", "frame_index":i*5,"publisher_center_xy_px":[100,100+i*10],
           "labeled_image_motion_proxy_px_per_second":float(i)} for i in range(50)]
    chosen=select_positive(pool,20)
    assert len(chosen)==20
    assert len({r['frame_index'] for r in chosen})==20
    assert {r['sampling_stratum'] for r in chosen}=={'upper_image_far_proxy','lower_image_near_proxy','fast_image_motion_blur_proxy','temporally_spread_residual'}
    assert evenly_spaced(list(range(10)),3)==[0,4,9]


def test_group_capacity_shortfall_redistributes_without_changing_groups():
    assert allocate(['one','two','three'],8,{'one':1,'two':6,'three':6})=={'one':1,'two':4,'three':3}


def test_component_rectangle_is_explicitly_a_proposal_and_fallback_is_hypothesis():
    frame=np.zeros((100,100,3),dtype=np.uint8)
    frame[47:53,47:53]=[20,220,200]
    box,method=hypothesized_box(frame,[50,50])
    assert box==[47,47,53,53]
    assert 'PROPOSAL_NOT_GROUND_TRUTH' in method
    _,method=hypothesized_box(np.zeros_like(frame),[50,50])
    assert 'HYPOTHESIS_NOT_GROUND_TRUTH' in method
