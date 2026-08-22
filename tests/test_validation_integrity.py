import os
import json
import numpy as np
import pytest

from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point
from scripts.evaluate.evaluate_independent_events import match_independent_events

def test_benchmark_independence():
    """Verify that independent ground truth has correct provenance and no model output dependencies."""
    gt_file = "data/benchmarks/tennis_events_independent/ground_truth.json"
    assert os.path.exists(gt_file), "Independent ground truth file must exist"
    
    with open(gt_file, 'r') as f:
        data = json.load(f)
        
    assert data.get('provenance') == "MANUAL_FROM_RAW_VIDEO", "Provenance must be MANUAL_FROM_RAW_VIDEO"
    assert "events" in data and len(data["events"]) >= 6
    
    # Check each event has frame_best, frame_min, frame_max
    for ev in data["events"]:
        assert "event_type" in ev
        assert "frame_best" in ev
        assert "frame_min" in ev and "frame_max" in ev
        assert ev["frame_min"] <= ev["frame_best"] <= ev["frame_max"]

def test_court_geometry_canonical_alignment():
    """Verify that all 14 canonical court keypoints align with ITF standard court dimensions."""
    can_kps = TennisCourtGeometry.get_canonical_keypoints()
    assert can_kps.shape == (14, 2)
    
    # Check boundaries
    assert np.all(can_kps[:, 0] >= 0.0)
    assert np.all(can_kps[:, 0] <= TennisCourtGeometry.COURT_WIDTH_DOUBLES)
    assert np.all(can_kps[:, 1] >= 0.0)
    assert np.all(can_kps[:, 1] <= TennisCourtGeometry.COURT_LENGTH)
    
    # Check specific corners
    np.testing.assert_allclose(can_kps[0], [0.0, 0.0])       # Top-left doubles
    np.testing.assert_allclose(can_kps[1], [10.97, 0.0])     # Top-right doubles
    np.testing.assert_allclose(can_kps[2], [0.0, 23.77])     # Bottom-left doubles
    np.testing.assert_allclose(can_kps[3], [10.97, 23.77])    # Bottom-right doubles
    np.testing.assert_allclose(can_kps[4], [1.37, 0.0])      # Top-left singles
    np.testing.assert_allclose(can_kps[5], [1.37, 23.77])    # Bottom-left singles

def test_independent_event_matcher():
    """Verify that the independent event matcher evaluates true positives, timing, and errors accurately."""
    gt_events = [
        {"event_type": "BOUNCE", "frame_best": 81, "frame_min": 80, "frame_max": 82, "ball_position_px": [716.6, 734.7], "court_position_m": [3.11, 20.56], "player_id": None},
        {"event_type": "PLAYER_1_HIT", "frame_best": 84, "frame_min": 83, "frame_max": 85, "ball_position_px": [698.9, 727.7], "court_position_m": [2.93, 20.33], "player_id": 1}
    ]
    
    # Exact prediction match
    pred_exact = [
        {"event_type": "BOUNCE", "frame": 81, "ball_position_px": [716.6, 734.7], "court_position_m": [3.11, 20.56], "player_id": None},
        {"event_type": "PLAYER_1_HIT", "frame": 84, "ball_position_px": [698.9, 727.7], "court_position_m": [2.93, 20.33], "player_id": 1}
    ]
    res = match_independent_events(pred_exact, gt_events, tolerance_frames=1, target_category="ALL")
    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["f1"] == 1.0
    assert res["mean_timing_frames"] == 0.0
    
    # 1-frame offset match (within tolerance 2)
    pred_offset = [
        {"event_type": "BOUNCE", "frame": 82, "ball_position_px": [718.0, 730.0], "court_position_m": [3.15, 20.50], "player_id": None},
        {"event_type": "PLAYER_1_HIT", "frame": 85, "ball_position_px": [690.0, 725.0], "court_position_m": [2.90, 20.20], "player_id": 1}
    ]
    res_off = match_independent_events(pred_offset, gt_events, tolerance_frames=2, target_category="ALL")
    assert res_off["tp"] == 2
    assert res_off["f1"] == 1.0
    assert res_off["mean_timing_frames"] == 1.0
