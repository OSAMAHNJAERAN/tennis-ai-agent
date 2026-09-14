import cv2
import numpy as np
import pytest

from scripts.evaluate.refine_court_line_geometry import refine, LINES
from src.court.court_geometry import TennisCourtGeometry


def court():
    canonical = TennisCourtGeometry.get_canonical_keypoints().astype(float)
    transform = cv2.getPerspectiveTransform(canonical[:4].astype(np.float32),
        np.array([[280,130],[680,130],[110,480],[850,480]],np.float32))
    return cv2.perspectiveTransform(canonical[None],transform)[0]


@pytest.mark.parametrize('shift', [(7,5), (-6,4), (3,-7)])
def test_known_projective_court_recovers_from_displaced_initialization(shift):
    truth = court()
    segments = [truth[[a,b]] for a,b in LINES]
    initial = truth + np.asarray(shift)
    actual, report = refine(initial, segments)
    assert report['accepted']
    assert np.linalg.norm(actual-truth,axis=1).max() < .1
    # A common projective fit preserves all canonical intersections.
    fit,_ = cv2.findHomography(truth,actual,0)
    assert np.linalg.norm(cv2.perspectiveTransform(truth[None],fit)[0]-actual,axis=1).max() < 1e-4


@pytest.mark.parametrize('mode', ['empty','far','one_family'])
def test_insufficient_or_remote_image_evidence_does_not_move_court(mode):
    points = court()
    segments = [] if mode == 'empty' else [points[[a,b]] for a,b in LINES]
    if mode == 'far': segments = np.asarray(segments)+100
    if mode == 'one_family': segments = [points[[a,b]] for a,b in LINES[:2]]
    actual, report = refine(points,segments)
    assert not report['accepted']
    np.testing.assert_array_equal(actual, points)


def test_already_aligned_court_is_unchanged_without_improvement():
    points = court()
    actual,report = refine(points,[points[[a,b]] for a,b in LINES])
    assert not report['accepted']
    np.testing.assert_array_equal(actual,points)


def test_nonfinite_points_fail():
    points=court();points[0,0]=np.nan
    with pytest.raises(ValueError):refine(points,[])
