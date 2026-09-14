"""Experimental assignment of shared crop detections to distinct players.

One global IoU-.5 suppression pass precedes a maximum-score assignment. Crop
provenance limits eligibility; proximity and bounded visual memory rank choices.
This heuristic does not establish racket ownership or infer contact events.
"""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.racket_tracking import RacketTracking


class GlobalRacketTracking(RacketTracking):
    def pool_candidates(self, candidates):
        """Keep the strongest geometry per overlap cluster and union its sources."""
        valid = []
        for candidate in candidates:
            box = np.asarray(candidate['bbox_xyxy'], dtype=float)
            score = float(candidate['confidence'])
            if box.shape != (4,) or not np.isfinite(box).all() or not math.isfinite(score):
                continue
            if np.any(box[2:] <= box[:2]) or not self.confidence <= score <= 1:
                continue
            valid.append({'bbox_xyxy': box.tolist(), 'confidence': score,
                          'sources': set(candidate['sources'])})
        valid.sort(key=lambda r: (-r['confidence'], *r['bbox_xyxy']))
        kept = []
        for candidate in valid:
            box = np.array(candidate['bbox_xyxy'])
            duplicate = None
            for previous in kept:
                other = np.array(previous['bbox_xyxy'])
                intersection = np.prod(np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])))
                union = np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]) - intersection
                if intersection / union > .5:
                    duplicate = previous
                    break
            if duplicate is None:
                kept.append(candidate)
            else:
                duplicate['sources'].update(candidate['sources'])
        return kept

    def assign_candidates(self, players, candidates, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Racket timestamp must be finite')
        identities = sorted(players, key=lambda identity: (type(identity).__name__, repr(identity)))
        for identity in identities:
            previous = self.memory.get(identity)
            if previous and timestamp <= previous[0]:
                raise ValueError('Racket timestamps must increase')
        for identity, previous in list(self.memory.items()):
            if timestamp - previous[0] > .25:
                self.memory.pop(identity)
        pooled = self.pool_candidates(candidates)
        # A separate zero-reward dummy for every player preserves missing states.
        reward = np.full((len(identities), len(pooled) + len(identities)), -1e6)
        reward[:, len(pooled):] = 0
        for row, identity in enumerate(identities):
            player = players[identity]
            if player is None:
                continue
            height = max(player.y2 - player.y1, 1.)
            previous = self.memory.get(identity)
            for column, candidate in enumerate(pooled):
                if identity not in candidate['sources']:
                    continue
                x1, y1, x2, y2 = candidate['bbox_xyxy']
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                motion = math.dist(center, previous[1]) if previous else 0
                if previous and motion > height * (1 + 10 * (timestamp - previous[0])):
                    continue
                # Rectangle gap is an image-space ranking term, not hand/contact evidence.
                gap = math.hypot(max(player.x1 - x2, x1 - player.x2, 0),
                                 max(player.y1 - y2, y1 - player.y2, 0))
                reward[row, column] = candidate['confidence'] - .15 * (motion + gap) / height
        rows, columns = linear_sum_assignment(reward, maximize=True)
        selected = {identities[row]: [pooled[column]] for row, column in zip(rows, columns)
                    if column < len(pooled) and reward[row, column] > 0}
        return {identity: self.associate(identity, selected.get(identity, []), timestamp,
                                         players[identity].y2 - players[identity].y1 if players[identity] else 1)
                for identity in identities}

    def observe(self, frame, players, timestamp):
        height, width = frame.shape[:2]
        candidates = []
        for identity, player in players.items():
            if player is None:
                continue
            margin = .75 * (player.y2 - player.y1)
            left, right = max(0, int(player.x1 - margin)), min(width, int(player.x2 + margin))
            top, bottom = max(0, int(player.y1 - margin * .6)), min(height, int(player.y2 + margin * .2))
            if right <= left or bottom <= top:
                continue
            result = self.model.predict(frame[top:bottom, left:right], classes=[38], conf=self.confidence,
                                        imgsz=self.imgsz, iou=.7, max_det=300, augment=False,
                                        device=self.device, verbose=False)[0]
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                candidates.append({'bbox_xyxy': (box + np.array([left, top, left, top])).tolist(),
                                   'confidence': float(score), 'sources': {identity}})
        self.last_candidates = candidates
        return self.assign_candidates(players, candidates, timestamp)
