"""Causal ground-motion estimates with bounded history and resumable totals.

This consumes independently qualified ground observations when available. It
does not establish ground contact, camera calibration or physical accuracy.
"""
from collections import deque
import math

import numpy as np


class GroundMotionAccumulator:
    """Keep only the backward fit window and the preceding observation.

    Source timestamps must increase. A camera/identity/PTS discontinuity is an
    explicit transition barrier; neither distance nor velocity bridges it.
    A checkpoint contains no frame archive and resumes the same totals.
    """

    def __init__(self, *, max_gap_seconds=.15, window_seconds=.15,
                 max_speed_mps=12., max_window_samples=256):
        for name, value in (('max_gap_seconds', max_gap_seconds), ('window_seconds', window_seconds),
                            ('max_speed_mps', max_speed_mps)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if type(max_window_samples) is not int or not 2 <= max_window_samples <= 4096:
            raise ValueError('max_window_samples must be an integer in [2, 4096]')
        self.settings = dict(max_gap_seconds=float(max_gap_seconds), window_seconds=float(window_seconds),
                             max_speed_mps=float(max_speed_mps), max_window_samples=max_window_samples)
        self.window = deque()
        self.last_timestamp = self.last_position = self.previous_velocity = None
        self.frames = self.accepted = self.raw_intervals = 0
        self.distance = self.duration = self.raw_distance = 0.

    def observe(self, timestamp, position, *, transition_allowed=True):
        if (isinstance(timestamp, (bool, np.bool_)) or not math.isfinite(timestamp)
                or (self.last_timestamp is not None and timestamp <= self.last_timestamp)):
            raise ValueError('Source timestamps must be finite and strictly increasing')
        if type(transition_allowed) is not bool:
            raise TypeError('Transition eligibility must be explicit boolean evidence')
        timestamp = float(timestamp)
        if position is not None:
            point = np.asarray(position, dtype=float)
            if point.shape != (2,):
                raise ValueError('Ground positions require two coordinates')
            position = tuple(map(float, point)) if np.isfinite(point).all() else None
        record = dict(timestamp_s=timestamp, ground_position_m=list(position) if position is not None else None,
                      speed_kmh=None, velocity_mps=None, acceleration_mps2=None, direction_deg=None,
                      velocity_window_seconds=None, velocity_effective_timestamp_s=None, position_fit_rmse_m=None)
        elapsed = timestamp - self.last_timestamp if self.last_timestamp is not None else None
        bridge = (transition_allowed and elapsed is not None and elapsed <= self.settings['max_gap_seconds']
                  and self.last_position is not None and position is not None)
        if not bridge:
            self.window.clear()
            self.previous_velocity = None
        if position is not None:
            # The existing estimator keeps at least two observations even when
            # their interval exceeds the requested fit window (but not max gap).
            while len(self.window) > 1 and timestamp - self.window[0][0] > self.settings['window_seconds']:
                self.window.popleft()
            if len(self.window) >= self.settings['max_window_samples']:
                raise ValueError('Source sample density exceeds the bounded motion window; no silent dropping')
            self.window.append((timestamp, position))
            if len(self.window) >= 2:
                times = np.array([item[0] for item in self.window], dtype=float)
                locations = np.array([item[1] for item in self.window], dtype=float)
                # Subtract a nearby origin before fitting long or wall-clock PTS.
                relative = times - times[0]
                mean_time = float(relative.mean())
                centered = relative - mean_time
                center = locations.mean(axis=0)
                velocity = centered @ (locations - center) / (centered @ centered)
                speed = float(np.linalg.norm(velocity))
                effective_time = float(times[0] + mean_time)
                if speed <= self.settings['max_speed_mps']:
                    residual = locations - (center + centered[:, None] * velocity)
                    record.update(position_fit_rmse_m=float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
                                  velocity_window_seconds=float(relative[-1]),
                                  velocity_effective_timestamp_s=effective_time, velocity_mps=velocity.tolist(),
                                  speed_kmh=speed * 3.6,
                                  direction_deg=math.degrees(math.atan2(velocity[1], velocity[0])) if speed > .1 else None)
                    if self.previous_velocity is not None:
                        previous_time, previous_velocity = self.previous_velocity
                        effective_elapsed = effective_time - previous_time
                        # Wall-clock float resolution can round two distinct
                        # window means together. Do not divide by zero or
                        # invent acceleration beyond source clock precision.
                        if effective_elapsed > 0:
                            record['acceleration_mps2'] = ((velocity - previous_velocity) / effective_elapsed).tolist()
                    self.previous_velocity = effective_time, velocity
                else:
                    self.window.clear()
                    self.window.append((timestamp, position))
                    self.previous_velocity = None
        if bridge:
            displacement = math.dist(self.last_position, position)
            if displacement / elapsed <= self.settings['max_speed_mps']:
                self.raw_distance += displacement
                self.raw_intervals += 1
            if record['speed_kmh'] is not None:
                self.distance += record['speed_kmh'] / 3.6 * elapsed
                self.duration += elapsed
                self.accepted += 1
        self.last_timestamp, self.last_position = timestamp, position
        self.frames += 1
        return record

    def summary(self):
        return dict(observed_distance_m=self.distance if self.accepted else None,
                    mean_speed_kmh=self.distance / self.duration * 3.6 if self.duration else None,
                    measured_duration_s=self.duration, accepted_intervals=self.accepted,
                    measurement_scope='OBSERVED_CONTIGUOUS_FOOT_PROXY_INTERVALS_ONLY',
                    distance_estimator='INTEGRAL_OF_BACKWARD_WINDOW_LEAST_SQUARES_SPEED',
                    velocity_window_seconds=self.settings['window_seconds'],
                    raw_observed_polyline_distance_m=self.raw_distance if self.raw_intervals else None,
                    uncertainty_scope='SHORT_WINDOW_FIT_RESIDUAL_IS_NOT_INDEPENDENT_SENSOR_ERROR; STATIONARY_NOISE_CAN_REMAIN')

    def checkpoint(self):
        return dict(schema_version=1, settings=self.settings.copy(), frames=self.frames,
                    last_timestamp=self.last_timestamp, last_position=self.last_position,
                    window=list(self.window), previous_velocity=None if self.previous_velocity is None else
                    [self.previous_velocity[0], self.previous_velocity[1].tolist()],
                    accepted=self.accepted, raw_intervals=self.raw_intervals,
                    distance=self.distance, duration=self.duration, raw_distance=self.raw_distance)

    @classmethod
    def restore(cls, checkpoint):
        if not isinstance(checkpoint, dict) or checkpoint.get('schema_version') != 1:
            raise ValueError('Unsupported motion checkpoint')
        expected = {'schema_version', 'settings', 'frames', 'last_timestamp', 'last_position', 'window',
                    'previous_velocity', 'accepted', 'raw_intervals', 'distance', 'duration', 'raw_distance'}
        if set(checkpoint) != expected:
            raise ValueError('Malformed motion checkpoint fields')
        restored = cls(**checkpoint['settings'])
        for field in ('frames', 'accepted', 'raw_intervals'):
            value = checkpoint[field]
            if type(value) is not int or value < 0:
                raise ValueError('Invalid motion checkpoint count')
            setattr(restored, field, value)
        if restored.accepted >= max(1, restored.frames) or restored.raw_intervals >= max(1, restored.frames):
            raise ValueError('Motion intervals exceed retained frame count')
        for field in ('distance', 'duration', 'raw_distance'):
            value = checkpoint[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid motion checkpoint aggregate')
            setattr(restored, field, float(value))
        if ((restored.accepted == 0 and (restored.distance != 0 or restored.duration != 0))
                or (restored.accepted > 0 and restored.duration == 0)
                or (restored.raw_intervals == 0 and restored.raw_distance != 0)
                or restored.duration > max(0, restored.frames - 1) * restored.settings['max_gap_seconds'] + 1e-8
                or restored.distance > restored.duration * restored.settings['max_speed_mps'] + 1e-8
                or restored.raw_distance > restored.raw_intervals * restored.settings['max_gap_seconds'] * restored.settings['max_speed_mps'] + 1e-8):
            raise ValueError('Motion checkpoint aggregates disagree with accepted intervals')
        window = checkpoint['window']
        if not isinstance(window, list) or len(window) > restored.settings['max_window_samples']:
            raise ValueError('Motion checkpoint exceeds its bounded window')
        def finite_time(value):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Invalid motion checkpoint timestamp')
            return float(value)
        def finite_point(value):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('Invalid motion checkpoint position')
            return tuple(finite_time(item) for item in value)
        for entry in window:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError('Invalid motion checkpoint window')
            timestamp, point = finite_time(entry[0]), finite_point(entry[1])
            if restored.window and timestamp <= restored.window[-1][0]:
                raise ValueError('Motion checkpoint window timestamps must increase')
            restored.window.append((timestamp, point))
        restored.last_timestamp = finite_time(checkpoint['last_timestamp']) if checkpoint['last_timestamp'] is not None else None
        restored.last_position = finite_point(checkpoint['last_position']) if checkpoint['last_position'] is not None else None
        if (bool(restored.frames) != (restored.last_timestamp is not None)
                or (restored.window and (restored.window[-1] != (restored.last_timestamp, restored.last_position)))
                or (restored.last_position is not None and not restored.window)):
            raise ValueError('Motion checkpoint does not align with its final observation')
        previous = checkpoint['previous_velocity']
        if (previous is None) != (len(restored.window) < 2):
            raise ValueError('Motion checkpoint fit does not agree with its window')
        if previous is not None:
            if not isinstance(previous, (tuple, list)) or len(previous) != 2 or len(restored.window) < 2:
                raise ValueError('Invalid motion checkpoint previous fit')
            time, velocity = finite_time(previous[0]), finite_point(previous[1])
            if time > restored.last_timestamp or math.hypot(*velocity) > restored.settings['max_speed_mps']:
                raise ValueError('Invalid motion checkpoint previous velocity')
            times = np.array([entry[0] for entry in restored.window], dtype=float)
            locations = np.array([entry[1] for entry in restored.window], dtype=float)
            relative = times - times[0]
            centered = relative - relative.mean()
            expected_time = float(times[0] + relative.mean())
            expected_velocity = centered @ (locations - locations.mean(axis=0)) / (centered @ centered)
            if time != expected_time or not np.allclose(velocity, expected_velocity, rtol=1e-12, atol=1e-12):
                raise ValueError('Motion checkpoint prior fit differs from its retained observations')
            restored.previous_velocity = time, np.array(velocity, dtype=float)
        return restored
