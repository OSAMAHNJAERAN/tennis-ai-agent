"""Continuous player pose and motion evidence; missing joints remain unknown."""

import math

import numpy as np

JOINT_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
               "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
               "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
SKELETON_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
                  (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6))


def decode_pose(keypoints, offset, confidence_threshold=.35):
    data = np.asarray(keypoints, dtype=float)
    if data.shape != (17, 3):
        raise ValueError("Expected 17 COCO keypoints with x, y, confidence")
    joints = {}
    for name, (x, y, confidence) in zip(JOINT_NAMES, data):
        valid = np.isfinite([x, y, confidence]).all() and confidence >= confidence_threshold
        joints[name] = {"position_px": [float(x + offset[0]), float(y + offset[1])] if valid else None,
                        "confidence": float(confidence) if math.isfinite(confidence) else None}
    return joints


def motion_samples(positions, timestamps, max_gap_seconds=.15, window_seconds=.15, max_speed_mps=12.):
    """Backward-window least-squares ground motion, preserving missing intervals.

    Positions are calibrated bbox-foot proxies. Speed/acceleration are estimates
    from that proxy, not independently validated biomechanics or foot contacts.
    """
    if len(positions) != len(timestamps):
        raise ValueError("Positions and timestamps must align")
    if any(not math.isfinite(t) for t in timestamps) or any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Timestamps must be finite and strictly increasing")
    result, start, previous_velocity = [], 0, None
    for index, point in enumerate(positions):
        record = {"timestamp_s": timestamps[index], "ground_position_m": list(point) if point is not None else None,
                  "speed_kmh": None, "velocity_mps": None, "acceleration_mps2": None, "direction_deg": None,
                  "velocity_window_seconds": None, "velocity_effective_timestamp_s": None, "position_fit_rmse_m": None}
        valid = point is not None and np.isfinite(point).all()
        if not valid:
            record["ground_position_m"] = None
            start, previous_velocity = index + 1, None
        else:
            if index and timestamps[index] - timestamps[index - 1] > max_gap_seconds:
                start, previous_velocity = index, None
            while start < index - 1 and timestamps[index] - timestamps[start] > window_seconds:
                start += 1
            if start < index:
                dt = timestamps[index] - timestamps[start]
                times = np.asarray(timestamps[start:index + 1], dtype=float)
                centered_times = times - times.mean()
                locations = np.asarray(positions[start:index + 1], dtype=float)
                center = locations.mean(axis=0)
                velocity = centered_times @ (locations - center) / (centered_times @ centered_times)
                speed = float(np.linalg.norm(velocity))
                if speed <= max_speed_mps:
                    residuals = locations - (center + centered_times[:, None] * velocity)
                    record['position_fit_rmse_m'] = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
                    record['velocity_window_seconds'] = dt
                    record['velocity_effective_timestamp_s'] = float(times.mean())
                    record["velocity_mps"] = velocity.tolist()
                    record["speed_kmh"] = speed * 3.6
                    record["direction_deg"] = math.degrees(math.atan2(velocity[1], velocity[0])) if speed > .1 else None
                    if previous_velocity is not None:
                        prev_time, prev_velocity = previous_velocity
                        record["acceleration_mps2"] = ((velocity - prev_velocity) / (times.mean() - prev_time)).tolist()
                    previous_velocity = float(times.mean()), velocity
                else:
                    start, previous_velocity = index, None
        result.append(record)
    return result


def summarize_motion(positions, timestamps):
    samples = motion_samples(positions, timestamps)
    distance, duration, accepted, raw_distance, raw_intervals = 0., 0., 0, 0., 0
    for index in range(1, len(positions)):
        first, second = positions[index - 1:index + 1]
        dt = timestamps[index] - timestamps[index - 1]
        if first is None or second is None or dt > .15:
            continue
        displacement = math.dist(first, second)
        if math.isfinite(displacement) and displacement / dt <= 12.:
            raw_distance += displacement
            raw_intervals += 1
        if samples[index]['speed_kmh'] is not None:
            # Integrate observed window velocity, not every noisy detector step.
            # There is no interpolation across missing frames or long gaps.
            distance += samples[index]['speed_kmh'] / 3.6 * dt
            duration += dt
            accepted += 1
    return {"observed_distance_m": distance if accepted else None,
            "mean_speed_kmh": distance / duration * 3.6 if duration else None,
            "measured_duration_s": duration, "accepted_intervals": accepted,
            "measurement_scope": "OBSERVED_CONTIGUOUS_FOOT_PROXY_INTERVALS_ONLY",
            "distance_estimator": "INTEGRAL_OF_BACKWARD_WINDOW_LEAST_SQUARES_SPEED",
            "velocity_window_seconds": .15,
            "raw_observed_polyline_distance_m": raw_distance if raw_intervals else None,
            "uncertainty_scope": "SHORT_WINDOW_FIT_RESIDUAL_IS_NOT_INDEPENDENT_SENSOR_ERROR; STATIONARY_NOISE_CAN_REMAIN",
            "samples": samples}


class PlayerMotionTracking:
    def __init__(self, model_path="yolo11n-pose.pt", device="cpu", confidence=.35):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence

    def observe(self, frame, boxes):
        output = {}
        height, width = frame.shape[:2]
        for identity, box in boxes.items():
            if box is None:
                output[identity] = None
                continue
            bw, bh = box.x2 - box.x1, box.y2 - box.y1
            left, top = max(0, int(box.x1 - bw * .25)), max(0, int(box.y1 - bh * .15))
            right, bottom = min(width, int(box.x2 + bw * .25)), min(height, int(box.y2 + bh * .15))
            if right <= left or bottom <= top:
                output[identity] = None
                continue
            result = self.model.predict(frame[top:bottom, left:right], imgsz=640, device=self.device,
                                        conf=.25, verbose=False)[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                output[identity] = None
                continue
            # Associate pose to the selected player by overlap, not result order.
            target = np.array([box.x1 - left, box.y1 - top, box.x2 - left, box.y2 - top])
            candidates = result.boxes.xyxy.cpu().numpy()
            intersection = np.maximum(0, np.minimum(candidates[:, 2:], target[2:]) - np.maximum(candidates[:, :2], target[:2])).prod(axis=1)
            union = (candidates[:, 2:] - candidates[:, :2]).prod(axis=1) + bw * bh - intersection
            overlaps = intersection / np.maximum(union, 1e-6)
            best = int(np.argmax(overlaps))
            output[identity] = (decode_pose(result.keypoints.data[best].cpu().numpy(), (left, top), self.confidence)
                                if overlaps[best] >= .25 else None)
        return output
