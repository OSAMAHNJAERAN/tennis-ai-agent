import pytest
from src.utils.bbox_utils import BBox, get_foot_position, get_center, point_distance, bbox_distance, get_bbox_area

def test_foot_position():
    bbox = BBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    foot_pos = get_foot_position(bbox)
    assert foot_pos == (20.0, 40.0)

def test_center():
    bbox = BBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)
    center_pos = get_center(bbox)
    assert center_pos == (20.0, 30.0)

def test_point_distance():
    pt1 = (0.0, 0.0)
    pt2 = (3.0, 4.0)
    dist = point_distance(pt1, pt2)
    assert dist == 5.0

def test_bbox_distance():
    b1 = BBox(x1=0.0, y1=0.0, x2=2.0, y2=2.0)  # center (1, 1)
    b2 = BBox(x1=4.0, y1=3.0, x2=6.0, y2=5.0)  # center (5, 4)
    dist = bbox_distance(b1, b2)
    assert dist == 5.0

def test_area():
    bbox = BBox(x1=10.0, y1=20.0, x2=30.0, y2=40.0)  # width=20, height=20
    area = get_bbox_area(bbox)
    assert area == 400.0
