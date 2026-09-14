"""Court-aware player selection for calibrated singles-camera segments.

Near/far labels are segment-local roles, not biometric identity across cuts or
changeovers. Missing boxes remain missing. A calibration failure abstains.
Optional per-frame calibrations support registered camera motion within a segment.
"""

from collections import Counter
import math

from src.court.calibration import CourtCalibration
from src.court.court_geometry import TennisCourtGeometry as Court


class CourtPlayerTracker:
    def __init__(self, calibration: CourtCalibration, side_margin_m=3., baseline_margin_m=6.,
                 memory_seconds=.5, max_speed_mps=12., localization_slack_m=.8):
        self.calibration = calibration
        self.side_margin_m = side_margin_m
        self.baseline_margin_m = baseline_margin_m
        self.memory_seconds = memory_seconds
        self.max_speed_mps = max_speed_mps
        self.localization_slack_m = localization_slack_m

    def _candidate(self, box, calibration=None):
        if not all(math.isfinite(float(v)) for v in (box.x1, box.y1, box.x2, box.y2, box.confidence)):
            return None
        if box.x2 <= box.x1 or box.y2 <= box.y1:
            return None
        point = (calibration or self.calibration).project_ground_point(((box.x1 + box.x2) / 2, box.y2))
        if point is None:
            return None
        x, y = point
        if not (-self.side_margin_m <= x <= Court.COURT_WIDTH_DOUBLES + self.side_margin_m
                and -self.baseline_margin_m <= y <= Court.COURT_LENGTH + self.baseline_margin_m):
            return None
        role = 1 if y >= Court.NET_TO_BASELINE else 2
        return role, point, box

    def select(self, detections, fps, calibrations=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be positive and finite")
        selected = {1: [None] * len(detections), 2: [None] * len(detections)}
        if calibrations is not None and len(calibrations) != len(detections):
            raise ValueError('Per-frame calibrations must align with detections')
        calibrations = calibrations if calibrations is not None else [self.calibration] * len(detections)
        if not any(calibration.is_valid for calibration in calibrations):
            return selected
        candidates = [[value for box in frame if (value := self._candidate(box, calibration)) is not None]
                      for frame, calibration in zip(detections, calibrations)]
        # Whole-segment presence is an initialization prior, not filled coverage.
        presence = Counter((role, box.track_id) for frame in candidates for role, _, box in frame
                           if box.track_id is not None)
        memory = {1: None, 2: None}
        for index, frame in enumerate(candidates):
            if not calibrations[index].is_valid:
                memory = {1: None, 2: None}
                continue
            used_ids = set()
            for role in (1, 2):
                previous = memory[role]
                if previous is not None and (index - previous[0]) / fps > self.memory_seconds:
                    previous = memory[role] = None
                options = []
                for candidate_role, point, box in frame:
                    if candidate_role != role or (box.track_id is not None and box.track_id in used_ids):
                        continue
                    if previous is not None:
                        last_frame, last_point, last_id = previous
                        distance = math.dist(point, last_point)
                        allowed = self.localization_slack_m + self.max_speed_mps * (index - last_frame) / fps
                        if distance > allowed:
                            continue
                        score = distance - (.5 if box.track_id is not None and box.track_id == last_id else 0.)
                    else:
                        baseline_y = Court.COURT_LENGTH if role == 1 else 0.
                        court_cost = abs(point[0] - Court.COURT_WIDTH_DOUBLES / 2) + .15 * abs(point[1] - baseline_y)
                        score = court_cost - 3 * presence[(role, box.track_id)] / max(1, len(detections))
                    score -= .2 * float(box.confidence)
                    options.append((score, point, box))
                if options:
                    _, point, box = min(options, key=lambda option: option[0])
                    selected[role][index] = box
                    memory[role] = (index, point, box.track_id)
                    if box.track_id is not None:
                        used_ids.add(box.track_id)
        return selected
