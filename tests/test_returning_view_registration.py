import cv2
import numpy as np
import pytest

from src.court.calibration import CourtCalibration
from src.court.camera_registration import CourtCameraRegistration
from src.court.returning_view_registration import ReturningViewRegistration


def fixture(seed=7):
    gray=np.random.default_rng(seed).integers(20,220,(360,640),np.uint8)
    image=cv2.cvtColor(cv2.GaussianBlur(gray,(3,3),.5),cv2.COLOR_GRAY2BGR)
    points=np.array([[40,40],[600,40],[600,320],[40,320]],np.float32)
    return image,points


def test_original_view_recovers_after_confirmation_without_filling_gap():
    image,points=fixture()
    initial=CourtCalibration(np.diag([.02,.02,1.]))
    tracker=ReturningViewRegistration(initial,points,fps=25,retry_seconds=.04)
    baseline=CourtCameraRegistration(initial,points)
    shift=np.array([[1.,0,14],[0,1.,-6],[0,0,1.]])
    moved=cv2.warpPerspective(image,shift,(640,360))
    frames=[image,image,np.zeros_like(image),moved,moved,moved,moved]
    results=[tracker.update(f) for f in frames]
    old=[baseline.update(f) for f in frames]
    assert [r.calibration.is_valid for r in results]==[True,True,False,False,False,True,True]
    assert not any(r.calibration.is_valid for r in old[2:])
    for i in (2,3,4):
        assert results[i].anchor_to_frame_px is None and results[i].audit['image_to_court'] is None
    assert results[5].calibration.project_ground_point((214,194))==pytest.approx((4,4),abs=.03)
    assert results[5].audit['recovered_this_frame'] and results[5].audit['registration_segment_id']==1
    assert not results[6].audit['recovered_this_frame']
    assert [r.audit['frame_index'] for r in results]==list(range(len(frames)))


def test_failed_confirmation_resets_streak_and_unrelated_scene_stays_missing():
    image,points=fixture()
    other,_=fixture(99)
    tracker=ReturningViewRegistration(CourtCalibration(np.eye(3)),points,fps=25,retry_seconds=.04)
    frames=[image,np.zeros_like(image),image,image,other,other,image,image,image]
    results=[tracker.update(f) for f in frames]
    assert not any(r.calibration.is_valid for r in results[1:8])
    assert results[-1].calibration.is_valid
    assert results[4].audit['confirmation_count']==0


def test_healthy_registration_matches_existing_path_exactly():
    image,points=fixture()
    initial=CourtCalibration(np.eye(3))
    a=CourtCameraRegistration(initial,points)
    b=ReturningViewRegistration(initial,points,fps=60)
    for i in range(5):
        shifted=cv2.warpAffine(image,np.array([[1.,0,i],[0,1.,i/2]]),(640,360))
        cv2.setRNGSeed(7);expected=a.update(shifted)
        cv2.setRNGSeed(7);actual=b.update(shifted)
        assert np.array_equal(expected.calibration.image_to_court,actual.calibration.image_to_court)
        assert actual.audit['recovery_count']==0


def test_retry_is_bounded_and_invalid_anchor_cannot_recover():
    image,points=fixture()
    tracker=ReturningViewRegistration(CourtCalibration(np.eye(3)),points,fps=60)
    tracker.update(image);tracker.update(np.zeros_like(image))
    results=[tracker.update(np.zeros_like(image)) for _ in range(60)]
    assert sum(r.audit['recovery_attempted'] for r in results)==2
    assert tracker.confirmation_frames==6
    invalid=ReturningViewRegistration(CourtCalibration(None,rejection_reason='INVALID'),points,fps=25)
    assert not any(invalid.update(image).calibration.is_valid for _ in range(5))


@pytest.mark.parametrize('fps',[0,-1,float('nan'),float('inf'),True])
def test_invalid_timing_rejected(fps):
    with pytest.raises(ValueError):
        ReturningViewRegistration(CourtCalibration(np.eye(3)),fixture()[1],fps=fps)
