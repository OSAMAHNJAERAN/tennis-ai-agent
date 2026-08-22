from typing import Tuple, Optional
from src.court.court_geometry import TennisCourtGeometry
from src.shot_analysis.shot_types import CourtDepthZone, CourtLateralZone, CourtZone3x3

class CourtZoneEngine:
    """
    Canonical 3x3 Court Zoning and Service Placement Geometry.
    Maps metric court coordinates (X, Y) to standardized depth and lateral zones.
    """
    NET_Y = 11.885
    COURT_WIDTH = 10.97
    COURT_LENGTH = 23.77
    SINGLES_LEFT = 1.37
    SINGLES_RIGHT = 9.60

    @classmethod
    def get_depth_zone(cls, court_y_m: float) -> CourtDepthZone:
        """Determines depth zone relative to net."""
        if court_y_m < -0.5 or court_y_m > cls.COURT_LENGTH + 0.5:
            return CourtDepthZone.OUT_OF_BOUNDS

        # Distance from net
        dist_from_net = abs(court_y_m - cls.NET_Y)
        if dist_from_net < 4.5:
            return CourtDepthZone.SHORT
        elif dist_from_net <= 9.0:
            return CourtDepthZone.MID
        else:
            return CourtDepthZone.DEEP

    @classmethod
    def get_lateral_zone(cls, court_x_m: float) -> CourtLateralZone:
        """Determines horizontal corridor."""
        if court_x_m < -0.5 or court_x_m > cls.COURT_WIDTH + 0.5:
            return CourtLateralZone.OUT_OF_BOUNDS

        # 3 horizontal sectors across full/singles court
        if court_x_m < 3.65:
            return CourtLateralZone.LEFT
        elif court_x_m <= 7.32:
            return CourtLateralZone.CENTER
        else:
            return CourtLateralZone.RIGHT

    @classmethod
    def get_3x3_zone(cls, court_x_m: float, court_y_m: float) -> CourtZone3x3:
        """Combines depth and lateral zones into canonical 3x3 zone."""
        depth = cls.get_depth_zone(court_y_m)
        lateral = cls.get_lateral_zone(court_x_m)

        if depth == CourtDepthZone.OUT_OF_BOUNDS or lateral == CourtLateralZone.OUT_OF_BOUNDS:
            return CourtZone3x3.OUT_OF_BOUNDS

        zone_name = f"{depth.value}_{lateral.value}"
        try:
            return CourtZone3x3(zone_name)
        except ValueError:
            return CourtZone3x3.OUT_OF_BOUNDS

    @classmethod
    def classify_service_placement(
        cls,
        bounce_x_m: float,
        bounce_y_m: float,
        target_service_box: Optional[str] = None
    ) -> str:
        """
        Classifies serve placement within service box into WIDE, BODY, or T.
        Service boxes:
          - Near Deuce: X in [1.37, 5.485], Y in [11.885, 18.285] -> T is near center line (X=5.485), Wide is near singles (X=1.37)
          - Near Ad: X in [5.485, 9.60], Y in [11.885, 18.285] -> T is near center line (X=5.485), Wide is near singles (X=9.60)
          - Far Deuce: X in [5.485, 9.60], Y in [5.485, 11.885] -> T is near center line (X=5.485), Wide is near singles (X=9.60)
          - Far Ad: X in [1.37, 5.485], Y in [5.485, 11.885] -> T is near center line (X=5.485), Wide is near singles (X=1.37)
        """
        center_x = 5.485
        dist_from_center = abs(bounce_x_m - center_x)

        if dist_from_center <= 0.85:
            return "T"
        elif dist_from_center >= 2.5:
            return "WIDE"
        else:
            return "BODY"
