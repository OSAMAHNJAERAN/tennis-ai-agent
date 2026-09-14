"""Experimental relative image proximity constraint; not ownership ground truth."""
import math

from src.tracking.global_racket_tracking import GlobalRacketTracking


def nearest_person_sources(players, candidates):
    """Intersect pooled crop provenance with nearest observed person rectangles.

    Distance is in raw image pixels to avoid making large foreground people
    artificially attractive. All people compete, even if their crop did not
    detect this racket. Tied overlapping people remain ambiguous for assignment.
    Inputs are the validated candidates from GlobalRacketTracking.pool_candidates.
    """
    constrained = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate['bbox_xyxy']
        gaps = {
            identity: math.hypot(max(person.x1 - x2, x1 - person.x2, 0),
                                 max(person.y1 - y2, y1 - person.y2, 0))
            for identity, person in players.items() if person is not None
        }
        minimum = min(gaps.values(), default=math.inf)
        sources = set(candidate['sources']).intersection(
            identity for identity, gap in gaps.items()
            if math.isclose(gap, minimum, rel_tol=0, abs_tol=1e-9))
        if sources:
            constrained.append({**candidate, 'sources': sources})
    return constrained


class NearestRacketOwnership(GlobalRacketTracking):
    """Keep detector and temporal association unchanged; constrain owner choices."""

    def assign_candidates(self, players, candidates, timestamp):
        pooled = self.pool_candidates(candidates)
        constrained = nearest_person_sources(players, pooled)
        return super().assign_candidates(players, constrained, timestamp)
