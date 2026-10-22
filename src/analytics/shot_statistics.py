from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.shot_analysis.shot_types import ShotEventEvidence, ShotType, ShotDirection, CourtZone3x3
from src.utils.shot_evidence_sequence import require_shot_result_owner


def _landing_record(shot):
    return {
        'shot_id': shot.shot_id,
        'player_id': shot.player_id,
        'shot_type': shot.shot_type.value,
        'x_m': round(shot.landing_court_position_m[0], 2),
        'y_m': round(shot.landing_court_position_m[1], 2),
        'landing_zone': shot.landing_zone.value if shot.landing_zone else None,
        'is_dead_ball': shot.is_dead_ball,
    }

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
        type_counts = dict.fromkeys(('total', 'forehand', 'backhand', 'serve', 'unknown'), 0)
        dir_counts = dict.fromkeys(('cross_court', 'down_the_line', 'middle', 'unknown'), 0)
        zone_counts = {zone.value: 0 for zone in CourtZone3x3}
        types = {ShotType.FOREHAND: 'forehand', ShotType.BACKHAND: 'backhand',
                 ShotType.SERVE: 'serve', ShotType.UNKNOWN: 'unknown'}
        directions = {ShotDirection.CROSS_COURT: 'cross_court', ShotDirection.DOWN_THE_LINE: 'down_the_line',
                      ShotDirection.MIDDLE: 'middle', ShotDirection.UNKNOWN: 'unknown'}
        for shot in shots:
            if shot.player_id != player_id or shot.is_dead_ball:
                continue
            type_counts['total'] += 1
            if shot.shot_type in types:
                type_counts[types[shot.shot_type]] += 1
            if shot.direction in directions:
                dir_counts[directions[shot.direction]] += 1
            if shot.landing_zone is not None and isinstance(shot.landing_zone, CourtZone3x3):
                zone_counts[shot.landing_zone.value] += 1

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
        shots: List[ShotEventEvidence],
        *,
        result_owner=None,
    ) -> List[Dict[str, Any]]:
        """Extracts structured list of verified bounce landings for heatmap visualization."""
        if result_owner is not None:
            owner = require_shot_result_owner(shots, result_owner)
            return owner.select(has_landing=True).project(_landing_record)
        return [_landing_record(shot) for shot in shots if shot.landing_court_position_m is not None]
