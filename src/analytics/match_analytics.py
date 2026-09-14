from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.scoring.match_state import MatchState
from src.shot_analysis.shot_types import ShotEventEvidence
from src.analytics.rally_analyzer import RallySegment, RallyAnalyzer
from src.analytics.serve_analyzer import ServeAnalyzer
from src.analytics.shot_statistics import ShotStatisticsAnalyzer
from src.line_calling.line_call_engine import LineCallEvidence

def _round_known(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None and np.isfinite(value) else None

class MatchAnalyticsAggregator:
    """
    Master Match Analytics Aggregator for T88J709.
    Consolidates point-level, game-level, set-level, and match-level analytics into structured JSON outputs.
    """

    @classmethod
    def build_point_analytics(
        cls,
        match_state: MatchState,
        shots: List[ShotEventEvidence],
        rallies: List[RallySegment],
        line_calls: List[LineCallEvidence],
        p1_distance_m: Optional[float] = None,
        p2_distance_m: Optional[float] = None,
        point_id: int = 1
    ) -> List[Dict[str, Any]]:
        """Constructs detailed per-point records."""
        pts = []
        
        serve_shot = next((s for s in shots if s.shot_type.value == "SERVE"), None)
        rally_obj = rallies[0] if rallies else None

        pts.append({
            "point_id": point_id,
            "set_number": match_state.current_set,
            "game_number": match_state.current_game,
            "server_id": match_state.server_id,
            "receiver_id": match_state.receiver_id,
            "serve_attempt": match_state.serve_attempt,
            "service_side": match_state.service_side.value,
            "point_winner_id": match_state.last_point_winner,
            "ending_reason": match_state.last_point_reason or "Point in progress / Fault",
            "is_review_pending": (match_state.point_state.value == "POINT_REVIEW_PENDING"),
            "total_shots_including_serve": rally_obj.total_strokes_including_serve if rally_obj else len([s for s in shots if not s.is_dead_ball]),
            "rally_hits_excluding_serve": rally_obj.rally_hits_excluding_serve if rally_obj else 0,
            "duration_s": rally_obj.duration_s if rally_obj else 0.0,
            "player_1_movement_m": _round_known(p1_distance_m, 2),
            "player_2_movement_m": _round_known(p2_distance_m, 2),
            "shot_ids": [s.shot_id for s in shots],
            "line_call_ids": [lc.event_id for lc in line_calls]
        })
        return pts

    @classmethod
    def build_match_analytics(
        cls,
        match_state: MatchState,
        shots: List[ShotEventEvidence],
        rallies: List[RallySegment],
        line_calls: List[LineCallEvidence],
        p1_positions: List[Optional[Tuple[float, float]]],
        p2_positions: List[Optional[Tuple[float, float]]],
        p1_distance_m: Optional[float],
        p2_distance_m: Optional[float],
        p1_speed_kmh: Optional[float],
        p2_speed_kmh: Optional[float],
        ball_speed_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Constructs the comprehensive match analytics record."""
        # Player 1 & 2 Shot stats
        p1_shot_stats = ShotStatisticsAnalyzer.compute_player_shot_stats(shots, player_id=1)
        p2_shot_stats = ShotStatisticsAnalyzer.compute_player_shot_stats(shots, player_id=2)

        # Serve Stats
        server_id = match_state.server_id
        serve_stats = ServeAnalyzer.compute_serve_statistics(shots, line_calls, server_id=server_id)

        # Heatmaps
        p1_heatmap = ShotStatisticsAnalyzer.compute_court_occupancy_heatmap(p1_positions)
        p2_heatmap = ShotStatisticsAnalyzer.compute_court_occupancy_heatmap(p2_positions)
        bounce_heatmap = ShotStatisticsAnalyzer.compute_bounce_landing_heatmap(shots)

        # Rally summary
        live_rallies = [r for r in rallies if not r.is_dead_ball_point]
        avg_rally_len = float(np.mean([r.total_strokes_including_serve for r in live_rallies])) if live_rallies else (1.0 if shots else 0.0)
        longest_rally = max([r.total_strokes_including_serve for r in live_rallies], default=0)

        return {
            "match_id": match_state.match_id,
            "format": match_state.format.value,
            "match_complete": match_state.match_complete,
            "winner_id": match_state.winner_id,
            "score_summary": match_state.get_score_summary_str(),
            "overview": {
                "total_points_played": 1,
                "total_shots_recorded": len(shots),
                "live_shots_recorded": len([s for s in shots if not s.is_dead_ball]),
                "dead_ball_shots_suppressed": len([s for s in shots if s.is_dead_ball]),
                "total_rallies": len(live_rallies),
                "average_rally_length": round(avg_rally_len, 1),
                "longest_rally_length": longest_rally
            },
            "players": {
                "player_1": {
                    "id": 1,
                    "movement": {
                        "distance_meters": _round_known(p1_distance_m, 2),
                        "average_speed_kmh": _round_known(p1_speed_kmh, 1)
                    },
                    "shots": p1_shot_stats["shot_type_counts"],
                    "direction": p1_shot_stats["direction_counts"],
                    "landing_zones": p1_shot_stats["landing_zone_counts"],
                    "court_occupancy_heatmap": p1_heatmap
                },
                "player_2": {
                    "id": 2,
                    "movement": {
                        "distance_meters": _round_known(p2_distance_m, 2),
                        "average_speed_kmh": _round_known(p2_speed_kmh, 1)
                    },
                    "shots": p2_shot_stats["shot_type_counts"],
                    "direction": p2_shot_stats["direction_counts"],
                    "landing_zones": p2_shot_stats["landing_zone_counts"],
                    "court_occupancy_heatmap": p2_heatmap
                }
            },
            "serve_analytics": serve_stats,
            "ball_landing_heatmap": bounce_heatmap,
            "ball_speed_analytics": ball_speed_summary or {}
        }
