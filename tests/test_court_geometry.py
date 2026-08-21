import pytest
import numpy as np
from src.court.court_geometry import TennisCourtGeometry

def test_court_dimensions():
    assert TennisCourtGeometry.COURT_WIDTH_DOUBLES == 10.97
    assert TennisCourtGeometry.COURT_LENGTH == 23.77
    assert TennisCourtGeometry.NET_HEIGHT_CENTER == 0.914
    assert TennisCourtGeometry.COURT_WIDTH_SINGLES == 8.23
    assert TennisCourtGeometry.SERVICE_LINE_DISTANCE == 6.40

def test_canonical_keypoints_shape(court_geometry):
    kpts = court_geometry.get_canonical_keypoints()
    assert kpts.shape == (14, 2)

def test_canonical_keypoints_within_court(court_geometry):
    kpts = court_geometry.get_canonical_keypoints()
    assert np.all(kpts[:, 0] >= 0.0)
    assert np.all(kpts[:, 0] <= TennisCourtGeometry.COURT_WIDTH_DOUBLES)
    assert np.all(kpts[:, 1] >= 0.0)
    assert np.all(kpts[:, 1] <= TennisCourtGeometry.COURT_LENGTH)

def test_court_polygon(court_geometry):
    poly = court_geometry.get_court_polygon()
    assert len(poly) == 4
    singles_poly = court_geometry.get_singles_court_polygon()
    assert len(singles_poly) == 4
