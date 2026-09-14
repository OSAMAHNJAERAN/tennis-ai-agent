"""Chronological broadcast-style review of observed tracking evidence.

The court panel displays player ground proxies only. It never displays an
airborne ball as a court-plane point or invents speed, score or bounce events.
"""

from collections import deque
import math

import cv2
import numpy as np

from src.tracking.player_motion_tracking import JOINT_NAMES, SKELETON_EDGES


class VisionReviewRenderer:
    # BGR palette; layout uses a 1280 x 720 design coordinate system.
    BACKGROUND = (19, 25, 27)
    PANEL = (27, 35, 37)
    WHITE = (239, 242, 237)
    MUTED = (148, 162, 161)
    LINE = (60, 73, 73)
    BALL = (124, 237, 212)
    COLORS = {'1': (226, 205, 102), '2': (139, 158, 247)}

    def __init__(self, fps, calibration_valid):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Review requires positive finite FPS')
        self.fps = fps
        self.calibration_valid = calibration_valid
        self.current_calibration_valid = calibration_valid
        self.ball_trail = deque(maxlen=max(2, round(fps * .6)))
        self.player_trails = {key: deque(maxlen=max(2, round(fps * 1.5))) for key in self.COLORS}
        self.occupancy = np.zeros((18, 10), dtype=np.float32)
        self.last_frame = -1

    @classmethod
    def text(cls, image, value, x, y, scale=.45, color=None, thickness=1):
        cv2.putText(image, str(value), (round(x), round(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color or cls.WHITE, thickness, cv2.LINE_AA)

    @staticmethod
    def point(value):
        return value is not None and len(value) == 2 and np.isfinite(value).all()

    @staticmethod
    def court_point(point):
        # Include 3 m side and 6 m baseline run-off; preserve metric aspect ratio.
        return round(1042 + (point[0] + 3) * 11), round(167 + (point[1] + 6) * 11)

    def court_panel(self, image, motions):
        cv2.rectangle(image, (1012, 80), (1256, 623), self.PANEL, -1)
        self.text(image, 'COURT POSITION', 1030, 110, .43, self.MUTED)
        self.text(image, 'Observed player movement', 1030, 133, .34, self.MUTED)
        if not self.current_calibration_valid:
            self.text(image, 'Calibration unavailable', 1030, 240, .4)
            return
        for row in range(18):
            for column in range(10):
                intensity = min(float(self.occupancy[row, column]) / 3., 1.)
                if intensity > 0:
                    a = self.court_point((-3 + column * 16.97 / 10, -6 + row * 35.77 / 18))
                    b = self.court_point((-3 + (column + 1) * 16.97 / 10, -6 + (row + 1) * 35.77 / 18))
                    color = tuple(round(base * (1 - intensity * .6) + tint * intensity * .6)
                                  for base, tint in zip(self.PANEL, (99, 128, 98)))
                    cv2.rectangle(image, a, b, color, -1)
        p = self.court_point
        cv2.rectangle(image, p((0, 0)), p((10.97, 23.77)), self.MUTED, 1, cv2.LINE_AA)
        for x in (1.37, 9.60):
            cv2.line(image, p((x, 0)), p((x, 23.77)), self.MUTED, 1, cv2.LINE_AA)
        for y in (5.485, 18.285):
            cv2.line(image, p((1.37, y)), p((9.60, y)), self.MUTED, 1, cv2.LINE_AA)
        cv2.line(image, p((5.485, 5.485)), p((5.485, 18.285)), self.MUTED, 1, cv2.LINE_AA)
        cv2.line(image, p((-.4, 11.885)), p((11.37, 11.885)), self.WHITE, 2, cv2.LINE_AA)
        for identity, trail in self.player_trails.items():
            color = self.COLORS[identity]
            for a, b in zip(list(trail)[:-1], list(trail)[1:]):
                cv2.line(image, p(a), p(b), color, 2, cv2.LINE_AA)
            current = motions[identity].get('ground_position_m')
            if self.point(current):
                x, y = p(current)
                if 1025 <= x <= 1240 and 150 <= y <= 570:
                    cv2.circle(image, (x, y), 5, color, -1, cv2.LINE_AA)
        self.text(image, 'FOOT-PROXY ESTIMATES', 1030, 585, .35, self.MUTED)
        self.text(image, 'Heat: observed occupancy', 1030, 607, .34, self.MUTED)

    def render(self, frame, index, detection, poses, rackets, motions):
        if index != self.last_frame + 1:
            raise ValueError('Review frames must be sequential; history cannot cross skipped frames')
        self.last_frame = index
        current_valid = self.calibration_valid and detection.get('court_registration_valid', True)
        if self.current_calibration_valid != current_valid:
            self.ball_trail.clear()
        self.current_calibration_valid = current_valid
        source = frame.copy()
        thickness = max(1, round(frame.shape[1] / 960))
        for identity, color in self.COLORS.items():
            player = detection.get(f'player_{identity}')
            if player:
                x1, y1, x2, y2 = map(round, player['bbox'])
                # Short corner marks keep the body visible.
                corner = max(6, round((x2 - x1) * .15))
                for x, y, sx, sy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                     (x1, y2, 1, -1), (x2, y2, -1, -1)):
                    cv2.line(source, (x, y), (x + sx * corner, y), color, thickness, cv2.LINE_AA)
                    cv2.line(source, (x, y), (x, y + sy * corner), color, thickness, cv2.LINE_AA)
            joints = poses.get(identity)
            if joints:
                for a, b in SKELETON_EDGES:
                    first, second = joints[JOINT_NAMES[a]]['position_px'], joints[JOINT_NAMES[b]]['position_px']
                    if self.point(first) and self.point(second):
                        cv2.line(source, tuple(map(round, first)), tuple(map(round, second)), color, thickness, cv2.LINE_AA)
            racket = rackets.get(identity, {})
            if racket.get('bbox_xyxy') is not None:
                x1, y1, x2, y2 = map(round, racket['bbox_xyxy'])
                cv2.rectangle(source, (x1, y1), (x2, y2), self.BALL, thickness, cv2.LINE_AA)
            ground = motions[identity].get('ground_position_m')
            if self.current_calibration_valid and self.point(ground):
                self.player_trails[identity].append(ground)
                column, row = math.floor((ground[0] + 3) / 16.97 * 10), math.floor((ground[1] + 6) / 35.77 * 18)
                if 0 <= row < 18 and 0 <= column < 10:
                    self.occupancy[row, column] += 1 / self.fps
            else:
                self.player_trails[identity].clear()
        ball = detection.get('ball', {})
        xy = ball.get('position_px')
        if self.point(xy):
            self.ball_trail.append(xy)
            trail = list(self.ball_trail)
            for j, (a, b) in enumerate(zip(trail[:-1], trail[1:])):
                cv2.line(source, tuple(map(round, a)), tuple(map(round, b)), self.BALL,
                         max(1, round(thickness * (j + 1) / len(trail))), cv2.LINE_AA)
            cv2.circle(source, tuple(map(round, xy)), max(5, thickness * 4), self.BALL, thickness, cv2.LINE_AA)
        else:
            self.ball_trail.clear()

        canvas = np.full((720, 1280, 3), self.BACKGROUND, np.uint8)
        self.text(canvas, 'TENNIS / VISION', 24, 38, .7, self.WHITE, 2)
        self.text(canvas, 'TRACKING REVIEW', 278, 37, .4, self.MUTED)
        seconds = index / self.fps
        self.text(canvas, f'{int(seconds // 60):02}:{seconds % 60:05.2f}', 1060, 38, .6)
        self.text(canvas, f'FRAME {index:05}', 1164, 37, .34, self.MUTED)
        cv2.line(canvas, (24, 57), (1256, 57), self.LINE, 1)
        scale = min(966 / source.shape[1], 543 / source.shape[0])
        rw, rh = round(source.shape[1] * scale), round(source.shape[0] * scale)
        left, top = 24 + (966 - rw) // 2, 80 + (543 - rh) // 2
        canvas[top:top + rh, left:left + rw] = cv2.resize(source, (rw, rh), interpolation=cv2.INTER_AREA)
        self.court_panel(canvas, motions)
        for identity, x in (('1', 24), ('2', 278)):
            color = self.COLORS[identity]
            cv2.circle(canvas, (x + 4, 650), 4, color, -1, cv2.LINE_AA)
            self.text(canvas, f'PLAYER {identity} / ' + ('NEAR' if identity == '1' else 'FAR'), x + 17, 654, .38, color)
            speed = motions[identity].get('speed_kmh') if self.current_calibration_valid else None
            self.text(canvas, f'{speed:.1f} km/h est.' if speed is not None else 'Speed unavailable', x, 680, .49)
        self.text(canvas, 'BALL OBSERVATION', 540, 654, .38, self.MUTED)
        self.text(canvas, 'Visible candidate' if self.point(xy) else 'Missing', 540, 680, .49, self.BALL)
        self.text(canvas, 'PHYSICAL BALL SPEED', 785, 654, .38, self.MUTED)
        self.text(canvas, 'Unverified', 785, 680, .49)
        self.text(canvas, 'EVENTS / SCORE', 1040, 654, .38, self.MUTED)
        self.text(canvas, 'Awaiting validation', 1040, 680, .44)
        self.text(canvas, 'Experimental observations  /  Gaps remain visible  /  Court heat reflects player presence, not bounce events',
                  24, 708, .33, self.MUTED)
        return cv2.resize(canvas, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
