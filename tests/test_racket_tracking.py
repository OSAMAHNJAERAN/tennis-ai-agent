import pytest

from src.tracking.racket_tracking import RacketTracking


def candidate(x):
    return {"bbox_xyxy": [x, 10, x + 20, 30], "confidence": .9}


def test_racket_missing_evidence_does_not_hold_an_observed_position():
    tracker = RacketTracking(model=object())
    first = tracker.associate(1, [candidate(0)], 0, 200)
    missing = tracker.associate(1, [], .033, 200)
    assert first["state"] == "DETECTED"
    assert missing["state"] == "MISSING"
    assert missing["position_px"] is None
    assert missing["contact_event"] is None


def test_racket_pixel_motion_and_memory_expiry():
    tracker = RacketTracking(model=object())
    tracker.associate(1, [candidate(0)], 0, 200)
    moved = tracker.associate(1, [candidate(3)], .1, 200)
    assert moved["velocity_px_per_s"] == pytest.approx([30, 0])
    assert moved["motion_direction_deg"] == 0
    reacquired = tracker.associate(1, [candidate(500)], 1, 200)
    assert reacquired["velocity_px_per_s"] is None
    assert reacquired["contact_event"] is None
