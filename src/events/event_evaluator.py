"""Canonical Phase 6.4 Event Evaluator & Stage Lineage Architecture.

Defines the authoritative 7-stage event processing hierarchy (Stages 0 to 6),
centralizes 1-to-1 bipartite matching with temporal tolerance,
and generates complete lineage traces for every ground-truth event.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from src.events.event_detector import (
    EventCandidate,
    EventType,
    PhysicalEventType,
    TennisEvent,
    TennisEventDetector,
    _Feature,
)
from src.tracking.temporal_ball_tracker import BallObservation, BallState, TemporalBallPoint
from src.utils.bbox_utils import BBox


class CanonicalStage(str, Enum):
    STAGE_0_RAW_PROPOSAL = "STAGE_0_RAW_PROPOSAL"
    STAGE_1_USABLE_OBSERVATION = "STAGE_1_USABLE_OBSERVATION"
    STAGE_2_PHYSICAL_CANDIDATE = "STAGE_2_PHYSICAL_CANDIDATE"
    STAGE_3_PHYSICAL_CONTACT = "STAGE_3_PHYSICAL_CONTACT"
    STAGE_4_SEMANTIC_TYPE = "STAGE_4_SEMANTIC_TYPE"
    STAGE_5_PLAYER_ATTRIBUTION = "STAGE_5_PLAYER_ATTRIBUTION"
    STAGE_6_AUTHORITATIVE_EVENT = "STAGE_6_AUTHORITATIVE_EVENT"


class FirstFailureStage(str, Enum):
    NO_RAW_PROPOSAL = "NO_RAW_PROPOSAL"
    TRACK_NOT_USABLE = "TRACK_NOT_USABLE"
    NO_EVENT_CANDIDATE = "NO_EVENT_CANDIDATE"
    PHYSICAL_CONTACT_REJECTED = "PHYSICAL_CONTACT_REJECTED"
    CONTACT_FAMILY_WRONG = "CONTACT_FAMILY_WRONG"
    SEMANTIC_TYPE_WRONG = "SEMANTIC_TYPE_WRONG"
    PLAYER_ATTRIBUTION_WRONG = "PLAYER_ATTRIBUTION_WRONG"
    TEMPORAL_MATCH_FAILURE = "TEMPORAL_MATCH_FAILURE"
    FINAL_SUPPRESSION = "FINAL_SUPPRESSION"
    FULLY_CORRECT = "FULLY_CORRECT"


@dataclass
class EventLineageRecord:
    video_id: str
    gt_event_id: int
    gt_event_type: str
    gt_player_id: Optional[int]
    gt_frame: int
    gt_frame_min: int
    gt_frame_max: int
    gt_timestamp_s: float
    raw_proposal_present: bool
    usable_ball_observation: bool
    candidate_generated: bool
    candidate_frame: Optional[int]
    candidate_time_error_ms: Optional[float]
    physical_contact_verified: bool
    contact_family: str  # PLAYER_CONTACT, COURT_CONTACT, UNKNOWN_CONTACT, NONE
    semantic_type: Optional[str]
    predicted_player: Optional[int]
    authoritative_event_emitted: bool
    final_match_status: str  # MATCHED_EXACT, MATCHED_WRONG_TYPE, MATCHED_WRONG_PLAYER, UNMATCHED
    first_failure_stage: str
    first_failure_reason: str


def canonical_one_to_one_matches(
    predictions: Sequence[Dict[str, Any]],
    ground_truth: Sequence[Dict[str, Any]],
    tolerance_s: float = 0.200,
    fps: float = 30.0,
    require_event_type: bool = False,
    require_player: bool = False,
) -> List[Tuple[int, int, float]]:
    """Greedy 1-to-1 bipartite matching within temporal tolerance.
    
    Returns list of (pred_idx, gt_idx, distance_frames).
    """
    tol_frames = max(1, round(tolerance_s * fps))
    pairs: List[Tuple[float, int, int]] = []

    for p_idx, pred in enumerate(predictions):
        p_frame = int(pred.get("frame", pred.get("frame_index", 0)))
        p_type = pred.get("event_type")
        p_player = pred.get("player_id")

        for g_idx, gt in enumerate(ground_truth):
            g_frame = int(gt.get("frame_best", gt.get("frame", 0)))
            g_min = int(gt.get("frame_min", g_frame))
            g_max = int(gt.get("frame_max", g_frame))
            g_type = gt.get("event_type")
            g_player = gt.get("player_id")

            if p_frame < g_min:
                dist = g_min - p_frame
            elif p_frame > g_max:
                dist = p_frame - g_max
            else:
                dist = 0.0

            if dist > tol_frames:
                continue

            if require_event_type:
                if str(p_type).upper() != str(g_type).upper():
                    continue

            if require_player and g_type != "BOUNCE" and g_player is not None:
                if p_player != g_player:
                    continue

            pairs.append((dist, p_idx, g_idx))

    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_preds: Set[int] = set()
    matched_gts: Set[int] = set()
    matches: List[Tuple[int, int, float]] = []

    for dist, p_idx, g_idx in pairs:
        if p_idx in matched_preds or g_idx in matched_gts:
            continue
        matched_preds.add(p_idx)
        matched_gts.add(g_idx)
        matches.append((p_idx, g_idx, dist))

    return matches


def evaluate_event_lineage(
    video_id: str,
    raw_proposals: Sequence[Sequence[Dict[str, Any]]],
    trajectory: Sequence[TemporalBallPoint],
    player1_boxes: Sequence[Optional[BBox]],
    player2_boxes: Sequence[Optional[BBox]],
    gt_events: Sequence[Dict[str, Any]],
    detector: TennisEventDetector,
    fps: float = 30.0,
    frame_size: Tuple[int, int] = (1280, 720),
) -> Tuple[List[EventLineageRecord], Dict[str, Any]]:
    """Evaluates the full 7-stage event lineage for all GT events in a video."""
    tol_frames = max(1, round(0.200 * fps))
    analysis = detector.analyze(
        trajectory, player1_boxes, player2_boxes, fps=fps, frame_size=frame_size
    )

    candidates = detector.detect_candidates(trajectory, fps=fps, frame_size=frame_size)
    cand_records = [
        {"frame": c.frame_index, "timestamp_s": c.timestamp_s, "score": c.score}
        for c in candidates
    ]

    event_records = [
        {
            "frame": e.frame_index,
            "timestamp_s": e.timestamp_s,
            "event_type": e.event_type.value,
            "player_id": e.player_id,
            "confidence": e.confidence,
        }
        for e in analysis.events
    ]

    # Map candidate matches
    cand_matches = canonical_one_to_one_matches(
        cand_records, gt_events, tolerance_s=0.200, fps=fps, require_event_type=False
    )
    cand_by_gt = {g_idx: (p_idx, dist) for p_idx, g_idx, dist in cand_matches}

    # Map authoritative event matches (ignoring type)
    phys_matches = canonical_one_to_one_matches(
        event_records, gt_events, tolerance_s=0.200, fps=fps, require_event_type=False
    )
    phys_by_gt = {g_idx: (p_idx, dist) for p_idx, g_idx, dist in phys_matches}

    # Map authoritative event matches (requiring exact type)
    exact_matches = canonical_one_to_one_matches(
        event_records, gt_events, tolerance_s=0.200, fps=fps, require_event_type=True
    )
    exact_by_gt = {g_idx: (p_idx, dist) for p_idx, g_idx, dist in exact_matches}

    lineage_records: List[EventLineageRecord] = []

    for g_idx, gt in enumerate(gt_events):
        g_id = int(gt.get("event_id", g_idx + 1))
        g_type = str(gt.get("event_type", "UNKNOWN"))
        g_player = gt.get("player_id")
        g_frame = int(gt.get("frame_best", gt.get("frame", 0)))
        g_min = int(gt.get("frame_min", g_frame))
        g_max = int(gt.get("frame_max", g_frame))
        g_time = float(g_frame / fps)

        # Stage 0: Raw ball proposal presence
        raw_count_in_win = sum(
            len(raw_proposals[f])
            for f in range(max(0, g_min - tol_frames), min(len(raw_proposals), g_max + tol_frames + 1))
        )
        raw_present = raw_count_in_win > 0

        # Stage 1: Usable ball observation
        tracked_count_in_win = sum(
            1
            for f in range(max(0, g_min - tol_frames), min(len(trajectory), g_max + tol_frames + 1))
            if trajectory[f].state in (BallState.DETECTED, BallState.TRACKED, BallState.INTERPOLATED)
            and trajectory[f].x_px is not None
        )
        usable_obs = tracked_count_in_win >= 3

        # Stage 2: Physical event candidate
        cand_info = cand_by_gt.get(g_idx)
        cand_gen = cand_info is not None
        cand_frame = candidates[cand_info[0]].frame_index if cand_gen else None
        cand_err_ms = (cand_info[1] / fps * 1000.0) if cand_gen else None

        # Stage 3, 4, 5: Physical contact, Semantic type, Player attribution
        phys_info = phys_by_gt.get(g_idx)
        exact_info = exact_by_gt.get(g_idx)

        phys_verified = phys_info is not None
        pred_event = analysis.events[phys_info[0]] if phys_verified else None
        sem_type = pred_event.event_type.value if pred_event else None
        pred_player = pred_event.player_id if pred_event else None

        # Contact family
        if sem_type in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT"):
            contact_family = "PLAYER_CONTACT"
        elif sem_type == "BOUNCE":
            contact_family = "COURT_CONTACT"
        elif sem_type == "UNKNOWN_EVENT":
            contact_family = "UNKNOWN_CONTACT"
        else:
            contact_family = "NONE"

        auth_emitted = phys_verified

        # Match status
        if exact_info is not None:
            # Check player match for player contacts
            if g_type != "BOUNCE" and g_player is not None and pred_player != g_player:
                final_status = "MATCHED_WRONG_PLAYER"
            else:
                final_status = "MATCHED_EXACT"
        elif phys_info is not None:
            final_status = "MATCHED_WRONG_TYPE"
        else:
            final_status = "UNMATCHED"

        # Determine FIRST failure stage
        if not raw_present:
            first_fail_stage = FirstFailureStage.NO_RAW_PROPOSAL.value
            first_fail_reason = "No raw detector-level ball proposals in temporal match window."
        elif not usable_obs:
            first_fail_stage = FirstFailureStage.TRACK_NOT_USABLE.value
            first_fail_reason = f"Tracking gap: only {tracked_count_in_win} usable points in temporal window."
        elif not cand_gen:
            first_fail_stage = FirstFailureStage.NO_EVENT_CANDIDATE.value
            first_fail_reason = "Ball tracked but kinematic score / speed below candidate threshold."
        elif not phys_verified:
            first_fail_stage = FirstFailureStage.PHYSICAL_CONTACT_REJECTED.value
            first_fail_reason = "Candidate generated but failed physical multi-cue verification or debounce."
        elif final_status == "MATCHED_WRONG_TYPE":
            # Check if contact family was wrong
            gt_is_player = g_type in ("PLAYER_HIT", "SERVE_CONTACT")
            pred_is_player = sem_type in ("PLAYER_1_HIT", "PLAYER_2_HIT", "SERVE_CONTACT")
            if gt_is_player != pred_is_player:
                first_fail_stage = FirstFailureStage.CONTACT_FAMILY_WRONG.value
                first_fail_reason = f"Contact family mismatch: GT is {g_type} but predicted as {sem_type}."
            else:
                first_fail_stage = FirstFailureStage.SEMANTIC_TYPE_WRONG.value
                first_fail_reason = f"Semantic subtype mismatch: GT is {g_type} but predicted as {sem_type}."
        elif final_status == "MATCHED_WRONG_PLAYER":
            first_fail_stage = FirstFailureStage.PLAYER_ATTRIBUTION_WRONG.value
            first_fail_reason = f"Player attribution error: GT player {g_player} vs predicted {pred_player}."
        else:
            first_fail_stage = FirstFailureStage.FULLY_CORRECT.value
            first_fail_reason = "Fully correct event detection, exact type, and player attribution."

        rec = EventLineageRecord(
            video_id=video_id,
            gt_event_id=g_id,
            gt_event_type=g_type,
            gt_player_id=g_player,
            gt_frame=g_frame,
            gt_frame_min=g_min,
            gt_frame_max=g_max,
            gt_timestamp_s=g_time,
            raw_proposal_present=raw_present,
            usable_ball_observation=usable_obs,
            candidate_generated=cand_gen,
            candidate_frame=cand_frame,
            candidate_time_error_ms=cand_err_ms,
            physical_contact_verified=phys_verified,
            contact_family=contact_family,
            semantic_type=sem_type,
            predicted_player=pred_player,
            authoritative_event_emitted=auth_emitted,
            final_match_status=final_status,
            first_failure_stage=first_fail_stage,
            first_failure_reason=first_fail_reason,
        )
        lineage_records.append(rec)

    metrics = {
        "video_id": video_id,
        "gt_count": len(gt_events),
        "raw_proposal_count": sum(1 for r in lineage_records if r.raw_proposal_present),
        "usable_obs_count": sum(1 for r in lineage_records if r.usable_ball_observation),
        "candidate_count": sum(1 for r in lineage_records if r.candidate_generated),
        "physical_verified_count": sum(1 for r in lineage_records if r.physical_contact_verified),
        "authoritative_count": len(analysis.events),
        "exact_matched_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_EXACT"),
        "wrong_type_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_WRONG_TYPE"),
        "wrong_player_count": sum(1 for r in lineage_records if r.final_match_status == "MATCHED_WRONG_PLAYER"),
        "unmatched_count": sum(1 for r in lineage_records if r.final_match_status == "UNMATCHED"),
    }
    return lineage_records, metrics
