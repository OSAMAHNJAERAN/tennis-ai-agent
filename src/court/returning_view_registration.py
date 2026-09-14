"""Recover court registration only when the original anchor view returns.

This does not detect a new court or validate the original landmark fit. Missing
intervals remain missing, and recovery needs consecutive image-supported fits.
"""
import math

from src.court.calibration import CourtCalibration
from src.court.camera_registration import CourtCameraRegistration, RegistrationResult


class ReturningViewRegistration:
    def __init__(self, anchor_calibration, court_keypoints, *, fps,
                 retry_seconds=.5, confirmation_seconds=.1, **registration_settings):
        if (isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0 or
                not math.isfinite(retry_seconds) or retry_seconds <= 0 or
                not math.isfinite(confirmation_seconds) or confirmation_seconds <= 0):
            raise ValueError('Recovery requires positive finite FPS and intervals')
        self.anchor_calibration = anchor_calibration
        self.court_keypoints = court_keypoints
        self.settings = registration_settings
        self.retry_frames = max(1, math.ceil(retry_seconds * fps))
        self.confirmation_frames = max(3, math.ceil(confirmation_seconds * fps))
        self.active = self._new_tracker()
        self.anchor_frame = None
        self.anchor_people = ()
        self.probe = None
        self.confirmations = 0
        self.next_retry = 0
        self.index = -1
        self.segment_id = 0
        self.recovery_count = 0

    def _new_tracker(self):
        return CourtCameraRegistration(self.anchor_calibration, self.court_keypoints, **self.settings)

    def _emit(self, result, state, *, publish=False, attempted=False, recovered=False):
        audit = {**result.audit, 'frame_index': self.index,
                 'recovery_state': state, 'recovery_attempted': attempted,
                 'recovered_this_frame': recovered, 'recovery_count': self.recovery_count,
                 'registration_segment_id': self.segment_id if publish else None,
                 'confirmation_count': self.confirmations,
                 'required_confirmation_frames': self.confirmation_frames,
                 'retry_interval_frames': self.retry_frames,
                 'recovery_scope': 'ORIGINAL_ANCHOR_VIEW_ONLY; NO_NEW_COURT_DETECTION'}
        if publish:
            return RegistrationResult(result.calibration, result.anchor_to_frame_px, audit)
        reason = 'RETURNING_VIEW_CONFIRMATION_PENDING' if state == 'CONFIRMING' else (result.audit.get('rejection_reason') or 'ANCHOR_VIEW_LOST')
        # Even a valid trial fit cannot publish coordinates before confirmation.
        audit.update(is_valid=False, rejection_reason=reason, reinitialization_required=True,
                     image_to_court=None, anchor_to_frame_px=None,
                     max_landmark_displacement_source_px=None)
        return RegistrationResult(CourtCalibration(None, rejection_reason=reason), None, audit)

    def update(self, frame, people=()):
        self.index += 1
        if self.anchor_frame is None:
            self.anchor_frame = frame.copy()
            self.anchor_people = tuple(people)
        if not self.active.lost_reason:
            result = self.active.update(frame, people)
            if result.calibration.is_valid:
                return self._emit(result, 'TRACKING', publish=True)
            self.last_failure = result
            self.next_retry = self.index + 1
            self.confirmations = 0
            return self._emit(result, 'LOST')
        if not self.anchor_calibration.is_valid:
            return self._emit(self.last_failure, 'INVALID_ANCHOR')
        if self.probe is None and self.index < self.next_retry:
            return self._emit(self.last_failure, 'WAITING')
        if self.probe is None:
            self.probe = self._new_tracker()
            # Initialization is always on the original image, never the new view.
            self.probe.update(self.anchor_frame, self.anchor_people)
        result = self.probe.update(frame, people)
        if not result.calibration.is_valid:
            self.last_failure = result
            self.probe = None
            self.confirmations = 0
            self.next_retry = self.index + self.retry_frames
            return self._emit(result, 'LOST', attempted=True)
        self.confirmations += 1
        if self.confirmations < self.confirmation_frames:
            return self._emit(result, 'CONFIRMING', attempted=True)
        self.active = self.probe
        self.probe = None
        self.segment_id += 1
        self.recovery_count += 1
        return self._emit(result, 'TRACKING', publish=True, attempted=True, recovered=True)
