"""Crop-based visual racket observations with bounded association memory.

No racket is inferred from a player's wrist or a held position. Missing detector
evidence stays missing. Image motion is not a validated swing/contact event.
"""

import math

import numpy as np


class RacketTracking:
    def __init__(self, model=None, model_path="yolo11m.pt", device="cpu", confidence=.25, imgsz=640):
        if model is None:
            from ultralytics import YOLO
            model = YOLO(model_path)
        self.model = model
        self.device, self.confidence, self.imgsz = device, confidence, imgsz
        self.memory = {}

    def associate(self, identity, candidates, timestamp, player_height):
        previous = self.memory.get(identity)
        if previous and timestamp <= previous[0]:
            raise ValueError("Racket timestamps must increase")
        if previous and timestamp - previous[0] > .25:
            previous = None
            self.memory.pop(identity, None)
        ranked = []
        for candidate in candidates:
            box = candidate["bbox_xyxy"]
            if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            distance = math.dist(center, previous[1]) if previous else 0
            if previous and distance > player_height * (1 + 10 * (timestamp - previous[0])):
                continue
            score = float(candidate["confidence"]) - .15 * distance / max(player_height, 1)
            ranked.append((score, center, candidate))
        if not ranked:
            return {"state": "MISSING", "position_px": None, "bbox_xyxy": None,
                    "confidence": None, "velocity_px_per_s": None, "motion_direction_deg": None,
                    "contact_event": None}
        _, center, candidate = max(ranked, key=lambda value: value[0])
        velocity = None
        if previous and timestamp - previous[0] <= .1:
            velocity = [(center[k] - previous[1][k]) / (timestamp - previous[0]) for k in (0, 1)]
        self.memory[identity] = (timestamp, center)
        return {"state": "DETECTED", "position_px": list(center), "bbox_xyxy": candidate["bbox_xyxy"],
                "confidence": float(candidate["confidence"]), "velocity_px_per_s": velocity,
                "motion_direction_deg": math.degrees(math.atan2(velocity[1], velocity[0])) if velocity and math.hypot(*velocity) > 1 else None,
                "contact_event": None}

    def observe(self, frame, players, timestamp):
        height, width = frame.shape[:2]
        outputs, used_centers = {}, []
        for identity, player in players.items():
            candidates = []
            player_height = player.y2 - player.y1 if player is not None else 1
            if player is not None:
                margin = .75 * player_height
                left, right = max(0, int(player.x1 - margin)), min(width, int(player.x2 + margin))
                top, bottom = max(0, int(player.y1 - margin * .6)), min(height, int(player.y2 + margin * .2))
                if right > left and bottom > top:
                    result = self.model.predict(frame[top:bottom, left:right], classes=[38], conf=self.confidence,
                                                imgsz=self.imgsz, device=self.device, verbose=False)[0]
                    for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        global_box = (box + np.array([left, top, left, top])).tolist()
                        center = [(global_box[0] + global_box[2]) / 2, (global_box[1] + global_box[3]) / 2]
                        if any(math.dist(center, used) < 10 for used in used_centers):
                            continue
                        candidates.append({"bbox_xyxy": global_box, "confidence": float(confidence)})
            outputs[identity] = self.associate(identity, candidates, timestamp, player_height)
            if outputs[identity]["position_px"] is not None:
                used_centers.append(outputs[identity]["position_px"])
        return outputs
