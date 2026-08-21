import numpy as np

class TennisCourtGeometry:
    """
    Canonical tennis court geometry using ITF standard dimensions in meters.
    The origin (0,0) is at the top-left corner of the doubles court.
    X-axis points right (width), Y-axis points down (length).
    
    Layout Diagram:
    (0,0)                                                (10.97,0)
      +------------------------------------------------------+
      | Doubles Alley                                        |
      +----+--------------------------------------------+----+
      |    |                Backcourt                   |    |
      |    +----------------------+---------------------+    |
      |    |    Service Box       |    Service Box      |    |
      |    |                      |                     |    |
      +----+----------------------+---------------------+----+ Net (y=11.885)
      |    |                      |                     |    |
      |    |    Service Box       |    Service Box      |    |
      |    +----------------------+---------------------+    |
      |    |                Backcourt                   |    |
      +----+--------------------------------------------+----+
      | Doubles Alley                                        |
      +------------------------------------------------------+
    (0,23.77)                                          (10.97,23.77)
    """
    
    # ITF Standard Dimensions (meters)
    COURT_LENGTH = 23.77
    COURT_WIDTH_DOUBLES = 10.97
    COURT_WIDTH_SINGLES = 8.23
    SERVICE_LINE_DISTANCE = 6.40  # Distance from net to service line
    NET_TO_BASELINE = 11.885      # Half of court length
    DOUBLES_ALLEY_WIDTH = 1.37
    NET_HEIGHT_CENTER = 0.914
    NET_HEIGHT_POST = 1.067
    
    @staticmethod
    def get_canonical_keypoints() -> np.ndarray:
        """
        Returns the 14 canonical keypoint positions in meters.
        Indices match standard tutorial conventions:
        0: (1.37, 0)       # top-left baseline singles
        1: (9.60, 0)       # top-right baseline singles  
        2: (1.37, 23.77)   # bottom-left baseline singles
        3: (9.60, 23.77)   # bottom-right baseline singles
        4: (0, 0)          # top-left baseline doubles
        5: (0, 23.77)      # bottom-left baseline doubles
        6: (10.97, 0)      # top-right baseline doubles
        7: (10.97, 23.77)  # bottom-right baseline doubles
        8: (1.37, 5.485)   # top-left service
        9: (1.37, 18.285)  # bottom-left service
        10: (9.60, 5.485)  # top-right service
        11: (9.60, 18.285) # bottom-right service
        12: (5.485, 5.485) # top center T
        13: (5.485, 18.285)# bottom center T
        """
        return np.array([
            [1.37, 0.0],       # 0
            [9.60, 0.0],       # 1
            [1.37, 23.77],     # 2
            [9.60, 23.77],     # 3
            [0.0, 0.0],        # 4
            [0.0, 23.77],      # 5
            [10.97, 0.0],      # 6
            [10.97, 23.77],    # 7
            [1.37, 5.485],     # 8
            [1.37, 18.285],    # 9
            [9.60, 5.485],     # 10
            [9.60, 18.285],    # 11
            [5.485, 5.485],    # 12
            [5.485, 18.285]    # 13
        ], dtype=np.float32)

    @staticmethod
    def get_court_polygon() -> np.ndarray:
        """Returns the full doubles court boundary polygon."""
        return np.array([
            [0.0, 0.0],
            [10.97, 0.0],
            [10.97, 23.77],
            [0.0, 23.77]
        ], dtype=np.float32)

    @staticmethod
    def get_singles_court_polygon() -> np.ndarray:
        """Returns the singles court boundary polygon."""
        return np.array([
            [1.37, 0.0],
            [9.60, 0.0],
            [9.60, 23.77],
            [1.37, 23.77]
        ], dtype=np.float32)
