"""Canonical construction path for the production temporal ball tracker."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Union

import yaml

from src.tracking.temporal_ball_tracker import TemporalBallTracker


@dataclass(frozen=True)
class TemporalTrackerSettings:
    high_conf_thresh: float
    low_conf_thresh: float
    max_prediction_gap: int
    max_interpolation_gap: int
    max_valid_speed_px_per_frame: float
    base_gating_radius_px: float
    enable_multi_candidate_association: bool
    enable_adaptive_gate: bool
    enable_short_gap_reacquisition: bool
    enable_camera_motion_compensation: bool
    enable_scale_normalization: bool
    enable_frame_bounds_filter: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_conf_thresh <= self.high_conf_thresh <= 1.0:
            raise ValueError("Tracker confidence thresholds must satisfy 0 <= low <= high <= 1")
        if self.max_prediction_gap < 0 or self.max_interpolation_gap < 0:
            raise ValueError("Tracker gap limits must be non-negative")
        if self.max_valid_speed_px_per_frame <= 0 or self.base_gating_radius_px <= 0:
            raise ValueError("Tracker motion limits must be positive")

    def constructor_kwargs(self) -> dict[str, Any]:
        return asdict(self)


ConfigSource = Union[str, Path, Mapping[str, Any]]


def resolve_temporal_tracker_settings(source: ConfigSource) -> TemporalTrackerSettings:
    """Resolve every tracker setting explicitly; no class-default fallback is allowed."""
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    else:
        config = source
    try:
        ball = config["ball_detection"]
        tracking = config["temporal_tracking"]
        return TemporalTrackerSettings(
            high_conf_thresh=float(ball["high_conf"]),
            low_conf_thresh=float(ball["low_conf"]),
            max_prediction_gap=int(tracking["max_prediction_gap"]),
            max_interpolation_gap=int(tracking["max_interpolation_gap"]),
            max_valid_speed_px_per_frame=float(tracking["max_valid_speed_px_per_frame"]),
            base_gating_radius_px=float(tracking["base_gating_radius_px"]),
            enable_multi_candidate_association=bool(tracking["enable_multi_candidate_association"]),
            enable_adaptive_gate=bool(tracking["enable_adaptive_gate"]),
            enable_short_gap_reacquisition=bool(tracking["enable_short_gap_reacquisition"]),
            enable_camera_motion_compensation=bool(tracking["enable_camera_motion_compensation"]),
            enable_scale_normalization=bool(tracking["enable_scale_normalization"]),
            enable_frame_bounds_filter=bool(tracking["enable_frame_bounds_filter"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete temporal tracker configuration: {exc}") from exc


def build_temporal_ball_tracker(source: ConfigSource) -> TemporalBallTracker:
    settings = resolve_temporal_tracker_settings(source)
    return TemporalBallTracker(**settings.constructor_kwargs())
