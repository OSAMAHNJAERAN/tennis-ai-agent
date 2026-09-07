"""
Tennis Ball Proposal Filter.
Enforces physical tennis ball observation priors, playable scene envelope,
and video-wide persistent static distractor elimination.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple
from collections import defaultdict

from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.bbox_utils import BBox


@dataclass(frozen=True)
class ProposalFilterSettings:
    """Configurable thresholds for visual ball candidate filtering."""
    min_bbox_dim_px: float = 2.0
    max_bbox_dim_px: float = 35.0
    max_aspect_ratio: float = 2.5
    top_margin_fraction: float = 0.04
    bottom_margin_fraction: float = 0.02
    side_margin_fraction: float = 0.02
    static_distractor_radius_px: float = 16.0
    static_distractor_min_occurrences: int = 8
    static_distractor_min_span_frames: int = 25
    enable_bbox_filter: bool = True
    enable_static_filter: bool = True
    enable_margin_filter: bool = True


class BallProposalFilter:
    """
    Pre-filters raw candidate proposals before temporal association.
    Eliminates persistent static background landmarks (scoreboards, court logos,
    net hardware) and oversized non-ball detections (spectator heads, racquets).
    """

    def __init__(self, settings: Optional[ProposalFilterSettings] = None):
        self.settings = settings or ProposalFilterSettings()

    def identify_persistent_distractors(
        self,
        frame_candidates: Sequence[Sequence[BallObservation]],
        scale: float = 1.0,
    ) -> List[Tuple[float, float, float]]:
        """
        Scans frames to find true static background landmarks (logos, net hardware,
        scoreboards) that remain motionless across consecutive frame runs (run_len >= 8).
        Returns list of (center_x, center_y, radius).
        """
        num_frames = len(frame_candidates)
        min_run = max(5, int(self.settings.static_distractor_min_occurrences))
        step_tol = 3.5 * scale
        cluster_rad = 6.0 * scale

        distractor_clusters: List[Tuple[float, float, float]] = []

        # Find consecutive runs of stationary candidates
        for fi in range(max(0, num_frames - min_run)):
            for c0 in frame_candidates[fi]:
                run = [c0]
                for offset in range(1, min_run + 4):
                    nxt_idx = fi + offset
                    if nxt_idx >= num_frames:
                        break
                    match = None
                    for c_next in frame_candidates[nxt_idx]:
                        if math.hypot(c_next.x_px - c0.x_px, c_next.y_px - c0.y_px) <= step_tol:
                            match = c_next
                            break
                    if match is not None:
                        run.append(match)
                    else:
                        break

                if len(run) >= min_run:
                    avg_x = sum(c.x_px for c in run) / len(run)
                    avg_y = sum(c.y_px for c in run) / len(run)
                    # Check if already clustered
                    if not any(math.hypot(avg_x - dx, avg_y - dy) <= cluster_rad * 1.5 for dx, dy, _ in distractor_clusters):
                        distractor_clusters.append((avg_x, avg_y, cluster_rad))

        return distractor_clusters

    def filter_video_candidates(
        self,
        frame_candidates: List[List[BallObservation]],
        frame_size: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
    ) -> List[List[BallObservation]]:
        """
        Applies bounding-box sanity, scene margin bounds, and persistent static
        distractor rejection across the video sequence.
        """
        w, h = frame_size if frame_size is not None else (1280, 720)
        max_dim = self.settings.max_bbox_dim_px * scale
        min_dim = self.settings.min_bbox_dim_px * scale
        top_y = h * self.settings.top_margin_fraction
        bottom_y = h * (1.0 - self.settings.bottom_margin_fraction)
        left_x = w * self.settings.side_margin_fraction
        right_x = w * (1.0 - self.settings.side_margin_fraction)

        # 1. Identify persistent static distractors if enabled
        static_distractors: List[Tuple[float, float, float]] = []
        if self.settings.enable_static_filter:
            static_distractors = self.identify_persistent_distractors(frame_candidates, scale=scale)

        filtered_frames: List[List[BallObservation]] = []

        for frame_idx, candidates in enumerate(frame_candidates):
            valid_candidates: List[BallObservation] = []
            for c in candidates:
                # Margin filter: ignore extreme outer margins
                if self.settings.enable_margin_filter:
                    if c.y_px < top_y or c.y_px > bottom_y or c.x_px < left_x or c.x_px > right_x:
                        continue

                # Bounding box filter
                if self.settings.enable_bbox_filter and c.bbox is not None:
                    bw = abs(c.bbox.x2 - c.bbox.x1)
                    bh = abs(c.bbox.y2 - c.bbox.y1)
                    if bw > max_dim or bh > max_dim or bw < min_dim or bh < min_dim:
                        continue
                    ratio = max(bw, bh) / max(min(bw, bh), 1e-3)
                    if ratio > self.settings.max_aspect_ratio:
                        continue

                # Persistent static distractor filter
                if static_distractors:
                    is_distractor = False
                    for sx, sy, srad in static_distractors:
                        if math.hypot(c.x_px - sx, c.y_px - sy) <= srad:
                            is_distractor = True
                            break
                    if is_distractor:
                        continue

                valid_candidates.append(c)

            filtered_frames.append(valid_candidates)

        return filtered_frames
