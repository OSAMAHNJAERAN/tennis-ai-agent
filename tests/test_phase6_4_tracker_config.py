from pathlib import Path

import pytest

from src.tracking.tracker_factory import (
    build_temporal_ball_tracker,
    resolve_temporal_tracker_settings,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase6_analytics" / "pipeline.yaml"


def test_canonical_config_resolves_every_tracker_setting() -> None:
    settings = resolve_temporal_tracker_settings(CONFIG)
    assert settings.max_prediction_gap == 4
    assert settings.max_interpolation_gap == 3
    assert settings.max_valid_speed_px_per_frame == 80.0
    assert settings.base_gating_radius_px == 45.0
    assert settings.high_conf_thresh == 0.08
    assert settings.low_conf_thresh == 0.01
    assert settings.enable_multi_candidate_association is True
    assert settings.enable_adaptive_gate is True
    assert settings.enable_short_gap_reacquisition is True
    assert settings.enable_camera_motion_compensation is False
    assert settings.enable_scale_normalization is True
    assert settings.enable_frame_bounds_filter is True


def test_factory_instance_exactly_matches_resolved_config() -> None:
    settings = resolve_temporal_tracker_settings(CONFIG)
    tracker = build_temporal_ball_tracker(CONFIG)
    assert tracker.high_conf_thresh == settings.high_conf_thresh
    assert tracker.low_conf_thresh == settings.low_conf_thresh
    assert tracker.max_prediction_gap == settings.max_prediction_gap
    assert tracker.max_interpolation_gap == settings.max_interpolation_gap
    assert tracker.max_valid_speed == settings.max_valid_speed_px_per_frame
    assert tracker.base_gating_radius == settings.base_gating_radius_px
    assert tracker.enable_multi_candidate == settings.enable_multi_candidate_association
    assert tracker.enable_adaptive_gate == settings.enable_adaptive_gate
    assert tracker.enable_reacquisition == settings.enable_short_gap_reacquisition
    assert tracker.enable_camera_comp == settings.enable_camera_motion_compensation
    assert tracker.enable_scale_norm == settings.enable_scale_normalization
    assert tracker.enable_frame_bounds_filter == settings.enable_frame_bounds_filter


def test_factory_rejects_incomplete_config_instead_of_falling_back() -> None:
    incomplete = {
        "ball_detection": {"high_conf": 0.08, "low_conf": 0.01},
        "temporal_tracking": {"max_prediction_gap": 6},
    }
    with pytest.raises(ValueError, match="Incomplete temporal tracker configuration"):
        resolve_temporal_tracker_settings(incomplete)


def test_factory_rejects_inverted_confidence_thresholds() -> None:
    import yaml

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["ball_detection"]["low_conf"] = 0.2
    with pytest.raises(ValueError, match="low <= high"):
        resolve_temporal_tracker_settings(config)


def test_frame_bounds_filter_rejects_out_of_frame_candidates() -> None:
    from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallTracker

    candidates = [
        [BallObservation(-2.0, 50.0, 0.9)],
        [BallObservation(102.0, 50.0, 0.9)],
        [BallObservation(50.0, 50.0, 0.9)],
    ]
    filtered = TemporalBallTracker(enable_frame_bounds_filter=True).track_video_candidates(
        candidates, fps=30.0, frame_size=(100, 100)
    )
    unfiltered = TemporalBallTracker(enable_frame_bounds_filter=False).track_video_candidates(
        candidates, fps=30.0, frame_size=(100, 100)
    )
    assert filtered[0].state == BallState.MISSING
    assert filtered[1].state == BallState.MISSING
    assert filtered[2].state in {BallState.DETECTED, BallState.TRACKED}
    assert unfiltered[0].state == BallState.DETECTED
