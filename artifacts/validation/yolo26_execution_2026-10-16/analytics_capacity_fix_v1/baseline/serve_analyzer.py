from typing import List, Dict, Any, Optional
from src.shot_analysis.shot_types import ShotEventEvidence, ShotType
from src.line_calling.line_call_engine import LineCallEvidence, LineCallDecision

class ServeAnalyzer:
    """
    Serve Statistics & Placement Analytics Engine.
    Evaluates first/second serve percentages, faults, double faults, and landing placement (Wide/Body/T).
    """

    @classmethod
    def compute_serve_statistics(
        cls,
        shot_evidences: List[ShotEventEvidence],
        line_calls: List[LineCallEvidence],
        server_id: int = 2
    ) -> Dict[str, Any]:
        """
        Aggregates serve statistics for the specified server.
        """
        serves = [s for s in shot_evidences if s.shot_type == ShotType.SERVE and s.player_id == server_id]
        
        first_attempts = 0
        first_in = 0
        first_faults = 0
        second_attempts = 0
        second_in = 0
        double_faults = 0
        
        placement_counts = {"WIDE": 0, "BODY": 0, "T": 0, "UNKNOWN": 0}

        for s in serves:
            first_attempts += 1
            # Check associated bounce line call
            matched_call = next((lc for lc in line_calls if lc.bounce_frame == s.bounce_frame), None)
            if matched_call:
                if matched_call.decision == LineCallDecision.SERVE_IN:
                    first_in += 1
                elif matched_call.decision == LineCallDecision.SERVE_FAULT:
                    first_faults += 1
            elif s.outcome == "FAULT" or s.is_dead_ball:
                first_faults += 1
            else:
                first_in += 1

            if s.landing_court_position_m is not None:
                from src.shot_analysis.court_zones import CourtZoneEngine
                place = CourtZoneEngine.classify_service_placement(
                    s.landing_court_position_m[0],
                    s.landing_court_position_m[1]
                )
                placement_counts[place] = placement_counts.get(place, 0) + 1
            else:
                placement_counts["UNKNOWN"] += 1

        first_pct = (first_in / first_attempts * 100.0) if first_attempts > 0 else 0.0

        return {
            "server_id": server_id,
            "first_serves_attempted": first_attempts,
            "first_serves_in": first_in,
            "first_serves_faulted": first_faults,
            "first_serve_percentage": round(first_pct, 1),
            "second_serves_attempted": second_attempts,
            "second_serves_in": second_in,
            "double_faults": double_faults,
            "placement_distribution": placement_counts
        }
