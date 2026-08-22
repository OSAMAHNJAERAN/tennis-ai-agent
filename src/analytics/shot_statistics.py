from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.shot_analysis.shot_types import ShotEventEvidence, ShotType, ShotDirection, CourtZone3x3

class ShotStatisticsAnalyzer:
    """
    Player Shot Distribution, Direction Ratios, and Spatial Heatmap Binning Engine.
    """

    @classmethod
    def compute_player_shot_stats(
        cls,
        shots: List[ShotEventEvidence],
        player_id: int
    ) -> Dict[str, Any]:
        """Aggregates stroke types, directions, and landing zones for a specific player."""
        p_shots = [s for s in shots if s.player_id == player_id and not s.is_dead_ball]
        
        type_counts = {
            "total": len(p_shots),
            "forehand": sum(1 for s in p_shots if s.shot_type == ShotType.FOREHAND),
            "backhand": sum(1 for s in p_shots if s.shot_type == ShotType.BACKHAND),
            "serve": sum(1 for s in p_shots if s.shot_type == ShotType.SERVE),
            "unknown": sum(1 for s in p_shots if s.shot_type == ShotType.UNKNOWN)
        }

        dir_counts = {
            "cross_court": sum(1 for s in p_shots if s.direction == ShotDirection.CROSS_COURT),
            "down_the_line": sum(1 for s in p_shots if s.direction == ShotDirection.DOWN_THE_LINE),
            "middle": sum(1 for s in p_shots if s.direction == ShotDirection.MIDDLE),
            "unknown": sum(1 for s in p_shots if s.direction == ShotDirection.UNKNOWN)
        }

        # 3x3 Landing Zone Breakdown
        zone_counts = {}
        for z in CourtZone3x3:
            zone_counts[z.value] = sum(1 for s in p_shots if s.landing_zone == z)

        return {
            "player_id": player_id,
            "shot_type_counts": type_counts,
            "direction_counts": dir_counts,
            "landing_zone_counts": zone_counts
        }

    @classmethod
    def compute_court_occupancy_heatmap(
        cls,
        positions: List[Optional[Tuple[float, float]]],
        grid_x_bins: int = 10,
        grid_y_bins: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generates 2D canonical court occupancy bins for player movement.
        Court dimensions: X in [0.0, 10.97], Y in [0.0, 23.77].
        """
        valid_pts = [p for p in positions if p is not None]
        if not valid_pts:
            return []

        x_edges = np.linspace(0.0, 10.97, grid_x_bins + 1)
        y_edges = np.linspace(0.0, 23.77, grid_y_bins + 1)

        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]

        H, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
        
        bins_data = []
        total_samples = len(valid_pts)
        for i in range(grid_x_bins):
            for j in range(grid_y_bins):
                count = int(H[i, j])
                if count > 0:
                    bins_data.append({
                        "x_min_m": round(float(x_edges[i]), 2),
                        "x_max_m": round(float(x_edges[i+1]), 2),
                        "y_min_m": round(float(y_edges[j]), 2),
                        "y_max_m": round(float(y_edges[j+1]), 2),
                        "count": count,
                        "density_pct": round(float(count / total_samples * 100.0), 2)
                    })
        return bins_data

    @classmethod
    def compute_bounce_landing_heatmap(
        cls,
        shots: List[ShotEventEvidence]
    ) -> List[Dict[str, Any]]:
        """Extracts structured list of verified bounce landings for heatmap visualization."""
        bounces = []
        for s in shots:
            if s.landing_court_position_m is not None:
                bounces.append({
                    "shot_id": s.shot_id,
                    "player_id": s.player_id,
                    "shot_type": s.shot_type.value,
                    "x_m": round(s.landing_court_position_m[0], 2),
                    "y_m": round(s.landing_court_position_m[1], 2),
                    "landing_zone": s.landing_zone.value if s.landing_zone else None,
                    "is_dead_ball": s.is_dead_ball
                })
        return bounces
