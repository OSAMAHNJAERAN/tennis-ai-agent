from typing import List, Optional, Tuple, Dict, Any, Set, Sequence
from contextlib import closing
import numpy as np

from src.utils.bbox_utils import BBox
from src.tracking.temporal_ball_tracker import TemporalBallPoint
from src.events.event_detector import TennisEvent, EventType
from src.shot_analysis.shot_types import (
    ShotType,
    ShotDirection,
    ShotOutcome,
    ShotClassificationSource,
    ShotEventEvidence,
    PlayerHandedness
)
from src.shot_analysis.shot_classifier import TennisShotClassifier
from src.shot_analysis.shot_direction import ShotDirectionClassifier
from src.shot_analysis.court_zones import CourtZoneEngine

class TennisShotLinker:
    """
    Temporal Shot Linker Engine.
    Sequentially links player hits/serves to subsequent bounces, classifies shot mechanics,
    evaluates direction, and tags 3x3 landing zones while strictly honoring dead-ball suppression.
    """
    
    def __init__(
        self,
        classifier: TennisShotClassifier,
        direction_classifier: Optional[ShotDirectionClassifier] = None,
        zone_engine: Optional[CourtZoneEngine] = None
    ):
        self.classifier = classifier
        self.direction_classifier = direction_classifier or ShotDirectionClassifier()
        self.zone_engine = zone_engine or CourtZoneEngine()

    def link_shots(
        self,
        events: Sequence[TennisEvent],
        ball_trajectory: List[TemporalBallPoint],
        player1_boxes: List[Optional[BBox]],
        player2_boxes: List[Optional[BBox]],
        raw_frames: Optional[List[np.ndarray]] = None,
        dead_event_ids: Optional[Set[int]] = None,
        *, result_output=None, event_output=None,
    ) -> Sequence[ShotEventEvidence]:
        """
        Process physical events; default callers retain the established list API.

        Explicit result_output is a caller-owned open ShotEvidenceSequence whose
        context must outlive consumers. Nonempty native inputs must be the exact
        event_output.events alias. A withheld empty event input stays empty;
        diagnostic events are never implicitly given authority by this option.
        """
        dead_event_ids = dead_event_ids or set()
        if result_output is not None:
            from src.utils.shot_evidence_sequence import ShotEvidenceSequence
            from src.utils.event_analysis_sequence import EventAnalysisSequence
            if not isinstance(result_output, ShotEvidenceSequence):
                raise TypeError('Shot result output requires a caller-owned ShotEvidenceSequence')
            result_output._open()
            if len(result_output):
                raise ValueError('Shot result output must be empty')
            if event_output is not None:
                if not isinstance(event_output, EventAnalysisSequence) or events is not event_output.events:
                    raise ValueError('Native shots require the exact authoritative event sequence alias')
                event_output._open()
                links = self._native_event_links(event_output)
            elif len(events):
                raise ValueError('Native shots require an explicit authoritative event owner')
            else:
                links = (value for value in ())  # Withheld authority never reads diagnostic events.
            shot_evidences = result_output
        else:
            if event_output is not None:
                raise ValueError('An event owner requires caller-owned shot result output')
            shot_evidences: List[ShotEventEvidence] = []
            sorted_events = sorted(events, key=lambda e: e.frame_index)
            hit_events = [e for e in sorted_events if e.event_type in (EventType.SERVE_CONTACT, EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT)]
            bounce_events = [e for e in sorted_events if e.event_type == EventType.BOUNCE]
            links = ((hit, next((bounce for bounce in bounce_events
                        if hit.frame_index < bounce.frame_index <=
                        (hit_events[index+1].frame_index if index+1 < len(hit_events) else float('inf')) + 5), None))
                     for index, hit in enumerate(hit_events))

        shot_counter = 1
        with closing(links):
            for hit_ev, next_bounce in links:
                hit_id = hit_ev.event_id
                hit_frame = hit_ev.frame_index
                hit_time = hit_ev.timestamp_s
                # Preserve the verified event semantic. Absolute clip position is not
                # valid evidence that a stroke is a serve.
                ev_type_str = hit_ev.event_type.value
                p_id = hit_ev.player_id
                is_dead = hit_id in dead_event_ids

                # Find player box
                p_boxes = player1_boxes if p_id == 1 else (player2_boxes if p_id == 2 else None)
                p_box = p_boxes[hit_frame] if (p_boxes and hit_frame < len(p_boxes)) else None

                # Ball point at hit
                ball_pt = ball_trajectory[hit_frame] if hit_frame < len(ball_trajectory) else None
                source_pos = (ball_pt.court_x_m, ball_pt.court_y_m) if (ball_pt and ball_pt.court_x_m is not None) else hit_ev.court_position_m

                landing_pos = None
                landing_zone = None
                bounce_id = None
                bounce_fr = None

                if next_bounce is not None:
                    landing_pos = next_bounce.court_position_m
                    bounce_id = next_bounce.event_id
                    bounce_fr = next_bounce.frame_index
                    if landing_pos is not None:
                        landing_zone = self.zone_engine.get_3x3_zone(landing_pos[0], landing_pos[1])

                # Classify stroke
                shot_type, shot_conf, class_src, conf_comps, reason = self.classifier.classify_shot(
                    event_type=ev_type_str,
                    hit_frame=hit_frame,
                    player_id=p_id,
                    player_box=p_box,
                    ball_point=ball_pt,
                    frames=raw_frames,
                    all_player_boxes=p_boxes,
                    is_dead_ball=is_dead
                )

                # Classify direction
                direction, dir_conf, dir_reason = self.direction_classifier.classify_direction(
                    start_pos_m=source_pos,
                    end_pos_m=landing_pos
                )

                # Speed
                speed_val = ball_pt.speed_kmh if (ball_pt and hasattr(ball_pt, 'speed_kmh')) else None

                # Outcome
                outcome = ShotOutcome.IN_PLAY
                if is_dead:
                    outcome = ShotOutcome.FAULT if shot_type == ShotType.SERVE else ShotOutcome.UNKNOWN

                evidence = ShotEventEvidence(
                    shot_id=shot_counter,
                    match_event_id=hit_id,
                    frame_index=hit_frame,
                    timestamp_s=hit_time,
                    player_id=p_id or 0,
                    shot_type=shot_type,
                    shot_confidence=shot_conf,
                    classification_source=class_src,
                    direction=direction,
                    direction_confidence=dir_conf,
                    direction_source="GEOMETRY_DERIVED",
                    source_court_position_m=source_pos,
                    landing_court_position_m=landing_pos,
                    landing_zone=landing_zone,
                    bounce_event_id=bounce_id,
                    bounce_frame=bounce_fr,
                    speed_kmh=speed_val,
                    outcome=outcome,
                    is_dead_ball=is_dead,
                    confidence_components=conf_comps,
                    reason=reason
                )
                shot_evidences.append(evidence)
                shot_counter += 1

        return shot_evidences

    @staticmethod
    def _native_event_links(event_output):
        """Two ordered reads and one-hit lookahead; never consume a used bounce.

        The same first bounce can belong to multiple nearby hits under the
        existing +5-frame linking rule. Equal-frame stable ordering is retained.
        """
        hit_rows = event_output.events_by_frame()
        bounce_rows = event_output.events_by_frame(event_type=EventType.BOUNCE)
        hits = (value for value in hit_rows if value.event_type in
                (EventType.SERVE_CONTACT, EventType.PLAYER_1_HIT, EventType.PLAYER_2_HIT))
        try:
            current = next(hits, None)
            bounce = next(bounce_rows, None)
            while current is not None:
                following = next(hits, None)
                while bounce is not None and bounce.frame_index <= current.frame_index:
                    bounce = next(bounce_rows, None)
                upper = following.frame_index + 5 if following is not None else float('inf')
                yield current, bounce if bounce is not None and bounce.frame_index <= upper else None
                current = following
        finally:
            hits.close()
            hit_rows.close()
            bounce_rows.close()
