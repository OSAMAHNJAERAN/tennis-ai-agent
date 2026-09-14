import numpy as np
import pytest

from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.temporal_ball_tracker import BallObservation


def test_fixed_graphic_with_jittered_candidate_centers_has_no_pixel_motion():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    frame[200:204, 100:104] = 255
    extractor = CandidatePixelMotion()
    assert extractor.measure(frame, [BallObservation(102, 202, .9)], 0) == [None]
    scores = extractor.measure(frame, [BallObservation(104, 201, .9)], .1)
    assert scores == [0.]


def test_moving_ball_produces_evidence_at_its_observed_position():
    first = np.zeros((540, 960, 3), dtype=np.uint8)
    second = first.copy()
    first[200:203, 100:103] = 255
    second[200:203, 110:113] = 255
    extractor = CandidatePixelMotion()
    extractor.measure(first, [], 0)
    score = extractor.measure(second, [BallObservation(111, 201, .1)], .1)[0]
    assert score > 200


def test_no_future_frame_is_required_and_missing_history_remains_unknown():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    extractor = CandidatePixelMotion()
    assert extractor.measure(frame, [BallObservation(100, 200, .9)], 0) == [None]
    assert extractor.measure(frame, [BallObservation(100, 200, .9)], .033) == [None]


def test_wrong_frame_geometry_and_time_reversal_are_rejected():
    extractor = CandidatePixelMotion()
    extractor.measure(np.zeros((540, 960, 3), dtype=np.uint8), [], 0)
    with pytest.raises(ValueError):
        extractor.measure(np.zeros((540, 960, 3), dtype=np.uint8), [], 0)
    with pytest.raises(ValueError):
        extractor.measure(np.zeros((270, 480, 3), dtype=np.uint8), [], .1)


def test_wider_patch_covers_a_ball_within_accepted_localization_error():
    first = np.zeros((540, 960, 3), dtype=np.uint8)
    current = first.copy()
    current[199:202, 106:109] = 255
    scores = []
    for radius in (3, 8):
        extractor = CandidatePixelMotion(patch_radius=radius)
        extractor.measure(first, [], 0)
        scores.append(extractor.measure(current, [BallObservation(100, 200, .9)], .1)[0])
    assert scores == [0., 255.]


def test_filter_preserves_unknown_history_and_exports_the_rejection_evidence():
    image = np.zeros((540, 960, 3), dtype=np.uint8)
    candidate = BallObservation(100, 200, .9)
    extractor = CandidatePixelMotion()
    kept, evidence = extractor.filter(image, [candidate], 0)
    assert kept == [candidate]
    assert evidence[0]['history_available'] is False
    kept, evidence = extractor.filter(image, [candidate], .1)
    assert kept == []
    assert evidence[0]['pixel_motion_score'] == 0
    assert evidence[0]['accepted'] is False
