"""Optional causal rejection of one already-selected ball observation."""

from collections import deque
import math

import cv2
import numpy as np

from src.tracking.candidate_patch_similarity import shifted_patch_similarity, aligned_local_residual


class SelectedPatchPersistence:
    def __init__(self, minimum_similarity=.98, maximum_local_residual=12., lag_seconds=.1,
                 reference_size=(960,540), patch_radius=5, translation_radius=2, center_radius=3):
        if (not all(math.isfinite(v) for v in (minimum_similarity,maximum_local_residual,lag_seconds))
                or not 0 < minimum_similarity <= 1 or not 0 <= maximum_local_residual <= 255 or lag_seconds <= 0
                or len(reference_size) != 2 or any(not isinstance(v,int) or v <= 0 for v in reference_size)
                or any(not isinstance(v,int) for v in (patch_radius,translation_radius,center_radius))
                or not 0 <= center_radius < patch_radius or translation_radius < 0):
            raise ValueError('Invalid selected-patch persistence settings')
        self.minimum_similarity = minimum_similarity
        self.maximum_local_residual = maximum_local_residual
        self.lag_seconds = lag_seconds
        self.reference_size = tuple(reference_size)
        self.patch_radius, self.translation_radius, self.center_radius = patch_radius, translation_radius, center_radius
        self.history = deque()
        self.timestamp = None
        self.shape = None

    def filter(self, frame, point, timestamp):
        if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8
                or not math.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp)
                or (self.shape is not None and frame.shape != self.shape)):
            raise ValueError('Require fixed uint8 BGR geometry and increasing finite timestamps')
        if self.timestamp is not None and timestamp-self.timestamp > max(.25,2*self.lag_seconds):
            self.history.clear()
        self.timestamp, self.shape = timestamp, frame.shape
        gray = cv2.cvtColor(cv2.resize(frame,self.reference_size,interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
        while len(self.history)>1 and self.history[1][0] <= timestamp-self.lag_seconds+1e-9:
            self.history.popleft()
        previous = self.history[0][1] if self.history and self.history[0][0] <= timestamp-self.lag_seconds+1e-9 else None
        similarity, residual = None, None
        if point is not None:
            height,width = frame.shape[:2]
            if (not all(math.isfinite(v) for v in (point.x_px,point.y_px))
                    or not 0 <= point.x_px < width or not 0 <= point.y_px < height):
                raise ValueError('Selected point must lie inside the image')
            if previous is not None:
                position = point.x_px*self.reference_size[0]/width, point.y_px*self.reference_size[1]/height
                similarity = shifted_patch_similarity(gray,previous,position,self.patch_radius,self.translation_radius)
                residual = aligned_local_residual(gray,previous,position,self.patch_radius,self.translation_radius,self.center_radius)
        rejected = (similarity is not None and residual is not None
                    and similarity >= self.minimum_similarity and residual <= self.maximum_local_residual)
        self.history.append((timestamp,gray))
        return (None if rejected else point), {'history_available':previous is not None,
            'selected_xy': [point.x_px,point.y_px] if point else None,
            'similarity':similarity,'local_residual':residual,'rejected':rejected}


def validate_spatial_candidate_config(config):
    ball = config.get('ball_detection',{})
    enabled = (ball.get('spatial_crops',{}).get('enabled',False)
               or ball.get('patch_persistence',{}).get('enabled',False))
    if enabled and (ball.get('backend') != 'wasb' or ball.get('temporal_step',3) != 1
                    or config.get('temporal_tracking',{}).get('strategy') != 'model_top1'
                    or config.get('event_detection',{}).get('authoritative_enabled',True)):
        raise ValueError('Experimental spatial/patch candidate requires WASB step1, model_top1 and withheld event authority')
