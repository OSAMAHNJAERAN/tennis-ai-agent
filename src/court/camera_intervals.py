"""Bounded causal scene evidence and conservative match-measurement intervals.

Classical histogram/image/registration changes suggest discontinuities. They do
not identify tennis, replays, advertisements or artificial slow motion. Explicit
review and qualified view evidence remain separate from those suggestions.
Original PTS tokens come from the worker's source_frame_mapping.jsonl; no assumed
30 FPS clock is introduced. Cut/zoom/gap intervals require fresh observed court
geometry, never a reused homography or an invented position in a missing shot.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math

import cv2
import numpy as np

from src.court.calibration import CourtCalibration
from src.court.camera_registration import RegistrationResult

SUPPORTED_VIEW = 'elevated_endcourt_singles'
EXCLUSIONS = {'replay', 'advertisement', 'close_up', 'slow_motion', 'warmup', 'between_points', 'unsupported_view'}


@dataclass(frozen=True)
class IntervalReview:
    evidence_id: str
    start_source_seconds: float
    end_source_seconds: float
    label: str
    viewpoint: str | None = None
    normal_speed_verified: bool = False
    provenance: str = 'manual_review'

    def __post_init__(self):
        if (not isinstance(self.evidence_id, str) or not 1 <= len(self.evidence_id) <= 160
                or self.provenance not in {'manual_review', 'independent_verified_interval'}):
            raise ValueError('Interval review needs an evidence ID and independent provenance.')
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
               for value in (self.start_source_seconds, self.end_source_seconds)) or self.end_source_seconds < self.start_source_seconds:
            raise ValueError('Reviewed source interval must have finite ordered times.')
        if self.label not in EXCLUSIONS | {'match_play', 'unknown'} or not isinstance(self.normal_speed_verified, bool):
            raise ValueError('Use a supported review label and explicit normal-speed flag.')
        if self.viewpoint is not None and self.viewpoint not in {SUPPORTED_VIEW, 'unsupported_view', 'unknown'}:
            raise ValueError('Unknown camera views cannot qualify elevated singles analysis.')


@dataclass(frozen=True)
class CourtViewEvidence:
    frame_index: int
    landmarks_px: object
    observed: tuple[bool, ...]
    calibration: CourtCalibration
    viewpoint: str = 'unknown'
    view_qualification_id: str | None = None

    def __post_init__(self):
        points = np.asarray(self.landmarks_px, float)
        if type(self.frame_index) is not int or self.frame_index < 0 or points.shape != (14, 2):
            raise ValueError('Court evidence uses a current frame and 14 semantic landmark coordinates.')
        if len(self.observed) != 14 or any(type(value) is not bool for value in self.observed):
            raise ValueError('Fourteen explicit observation/visibility flags are required.')
        if not isinstance(self.calibration, CourtCalibration):
            raise ValueError('Supply the actual observed court calibration.')
        if self.viewpoint not in {SUPPORTED_VIEW, 'unsupported_view', 'unknown'}:
            raise ValueError('Unsupported court viewpoint evidence.')
        if self.view_qualification_id is not None and (not isinstance(self.view_qualification_id, str) or not 1 <= len(self.view_qualification_id) <= 160):
            raise ValueError('View qualification needs its measured evidence identifier.')

    @property
    def fresh_geometry_valid(self):
        points = np.asarray(self.landmarks_px, float)
        matrix = self.calibration.image_to_court
        return (all(self.observed) and np.isfinite(points).all() and self.calibration.is_valid
                and len(self.calibration.inlier_mask) == 14 and self.calibration.inlier_count >= 6
                and np.asarray(matrix).shape == (3, 3) and np.isfinite(matrix).all()
                and abs(np.linalg.det(matrix)) > 1e-12)


class CameraIntervals:
    def __init__(self, analysis_id, *, max_source_gap_seconds=.5,
                 histogram_cut_threshold=.55, thumbnail_change_threshold=.25, zoom_change_threshold=.08):
        if not isinstance(analysis_id, str) or not 1 <= len(analysis_id) <= 160:
            raise ValueError('A unique analysis/live-session ID is required.')
        guards = (max_source_gap_seconds, histogram_cut_threshold, thumbnail_change_threshold, zoom_change_threshold)
        if not all(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0 for value in guards):
            raise ValueError('Scene guards must be positive finite engineering settings.')
        if max(histogram_cut_threshold, thumbnail_change_threshold, zoom_change_threshold) > 1:
            raise ValueError('Scene/zoom compatibility thresholds use unit fractions.')
        self.analysis_id = analysis_id
        self.max_gap = Fraction(str(max_source_gap_seconds))
        self.histogram_cut_threshold, self.thumbnail_change_threshold = histogram_cut_threshold, thumbnail_change_threshold
        self.zoom_change_threshold = zoom_change_threshold
        self.epoch, self.previous, self.thumbnail, self.histogram, self.shape, self.zoom = 0, None, None, None, None, None
        self.reanchor_required, self.anchor_frame = True, None
        self.qualified_view_id = None
        self.last_fingerprint = self.last_result = None

    @property
    def segment_id(self):
        digest = hashlib.sha256(self.analysis_id.encode()).hexdigest()[:16]
        return f'camera_{digest}_{self.epoch}'

    def _timing(self, record):
        if not isinstance(record, dict) or type(record.get('frame_index')) is not int or record['frame_index'] < 0:
            raise ValueError('Use the frame-ordered source mapping record, not an assumed FPS clock.')
        normalized = record.get('timestamp_seconds')
        if isinstance(normalized, bool) or not isinstance(normalized, (float, int)) or not math.isfinite(normalized) or normalized < 0:
            raise ValueError('Normalized playback time must be finite nonnegative seconds.')
        pts, base, source_index = record.get('source_pts'), record.get('source_time_base'), record.get('source_frame_index')
        valid = isinstance(pts, str) and isinstance(base, str) and type(source_index) is int and source_index >= 0
        time = None
        if valid:
            try:
                ticks, fraction = int(pts), Fraction(base)
                valid = fraction > 0 and ticks != 9223372036854775807
                time = ticks * fraction if valid else None
                if time is not None and not math.isfinite(float(time)):
                    valid, time = False, None
            except (ValueError, ZeroDivisionError, OverflowError):
                valid = False
        return {'frame_index': record['frame_index'], 'normalized_seconds': normalized,
                'source_pts': pts, 'source_time_base': base, 'source_frame_index': source_index,
                'source_time': time, 'source_valid': valid,
                'transition_valid': record.get('validMeasurementTransitionFromPrevious') is True}

    @staticmethod
    def _scale(registration, width, height):
        if registration is None or not registration.calibration.is_valid or registration.anchor_to_frame_px is None:
            return None
        matrix = np.asarray(registration.anchor_to_frame_px, float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            return None
        points = np.array([[[width / 2, height / 2], [width / 2 + 1, height / 2], [width / 2, height / 2 + 1]]], np.float64)
        moved = cv2.perspectiveTransform(points, matrix)[0]
        scale = math.sqrt(abs(np.linalg.det(np.column_stack((moved[1]-moved[0], moved[2]-moved[0])))))
        return scale if math.isfinite(scale) and scale > 0 else None

    def update(self, frame, timing, *, registration=None, court_observation=None, review=None):
        """Emit one source-aligned eligibility record without buffering intervals.

        A cut suggestion creates a segment and disables geometry until the
        caller supplies current-frame observed landmarks/calibration. Reviewed
        normal match play and the supported viewpoint are separate eligibility
        gates. Keep records on disk rather than accumulating a match-long list.
        """
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
            raise ValueError('Use a nonempty uint8 BGR source frame.')
        current = self._timing(timing)
        if review is not None and not isinstance(review, IntervalReview):
            review = IntervalReview(**review)
        if court_observation is not None and not isinstance(court_observation, CourtViewEvidence):
            raise ValueError('Use typed current-frame court observation evidence.')
        if court_observation is not None and court_observation.frame_index != current['frame_index']:
            raise ValueError('Recalibration requires landmarks observed in this exact source-mapping frame.')
        height, width = frame.shape[:2]
        thumb = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        histogram = cv2.calcHist([thumb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).ravel()
        histogram /= histogram.sum()
        evidence_key = {'timing': timing, 'review': asdict(review) if review else None,
                        'court': [court_observation.frame_index, list(court_observation.observed),
                                  court_observation.viewpoint, court_observation.view_qualification_id,
                                  court_observation.calibration.to_dict()] if court_observation else None,
                        'registration': registration.calibration.to_dict() if registration else None}
        evidence_bytes = np.asarray(court_observation.landmarks_px, np.float64).tobytes() if court_observation else b''
        registration_bytes = np.asarray(registration.anchor_to_frame_px, np.float64).tobytes() if registration and registration.anchor_to_frame_px is not None else b''
        fingerprint = hashlib.sha256(thumb.tobytes() + evidence_bytes + registration_bytes + json.dumps(evidence_key, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if fingerprint == self.last_fingerprint:
            return json.loads(json.dumps(self.last_result))
        if self.previous and current['frame_index'] <= self.previous['frame_index']:
            raise ValueError('Frame mapping moved backwards or repeated with different image evidence.')
        reasons = []
        histogram_distance = thumbnail_difference = zoom_change = None
        scale = self._scale(registration, width, height)
        if self.previous:
            if current['frame_index'] != self.previous['frame_index'] + 1:
                reasons.append('frame_mapping_gap')
            if current['source_valid'] and self.previous['source_valid']:
                delta = current['source_time'] - self.previous['source_time']
                if delta <= 0 or delta > self.max_gap:
                    reasons.append('source_pts_gap_or_duplicate')
                if current['source_frame_index'] != self.previous['source_frame_index'] + 1:
                    reasons.append('source_frame_gap_or_duplicate')
            elif current['source_valid'] != self.previous['source_valid']:
                reasons.append('source_pts_unavailable')
            if self.shape != (height, width):
                reasons.append('source_resolution_changed')
            histogram_distance = float(np.abs(histogram - self.histogram).sum() / 2)
            thumbnail_difference = float(np.abs(thumb.astype(float)-self.thumbnail.astype(float)).mean() / 255)
            if histogram_distance >= self.histogram_cut_threshold and thumbnail_difference >= self.thumbnail_change_threshold:
                reasons.append('provisional_scene_cut')
            if scale is not None and self.zoom is not None:
                zoom_change = abs(scale / self.zoom - 1)
                if zoom_change >= self.zoom_change_threshold:
                    reasons.append('provisional_zoom_change')
        old_reanchor_required = self.reanchor_required
        if reasons:
            self.epoch += 1
            self.reanchor_required, self.anchor_frame = True, None
            self.qualified_view_id = None
        points = np.asarray(court_observation.landmarks_px, float) if court_observation else None
        fresh = (court_observation is not None and court_observation.fresh_geometry_valid
                 and bool((points[:, 0] >= 0).all() and (points[:, 0] < width).all()
                          and (points[:, 1] >= 0).all() and (points[:, 1] < height).all()))
        needs_new_registration = bool(fresh and (old_reanchor_required or reasons or registration is None
                                                 or not registration.calibration.is_valid))
        if fresh:
            self.reanchor_required = False
            if needs_new_registration:
                self.anchor_frame = current['frame_index']
            self.qualified_view_id = (court_observation.view_qualification_id
                                      if court_observation.viewpoint == SUPPORTED_VIEW else None)
        if registration is not None and not registration.calibration.is_valid and not fresh:
            self.reanchor_required = True
        active_review = review if (review is not None and current['source_time'] is not None
                                  and review.start_source_seconds <= float(current['source_time']) <= review.end_source_seconds) else None
        reviewed_unsupported_view = active_review is not None and active_review.viewpoint == 'unsupported_view'
        view_supported = not reviewed_unsupported_view and ((active_review is not None and active_review.viewpoint == SUPPORTED_VIEW)
                                                           or self.qualified_view_id is not None)
        play_verified = (active_review is not None and active_review.label == 'match_play' and active_review.normal_speed_verified)
        excluded = active_review is not None and active_review.label in EXCLUSIONS
        geometry_valid = not self.reanchor_required and (fresh or (registration is not None and registration.calibration.is_valid))
        allowed = bool(view_supported and play_verified and not excluded and geometry_valid and current['source_valid'])
        blocked = []
        if excluded:
            blocked.append('reviewed_' + active_review.label)
        if not view_supported:
            blocked.append('reviewed_unsupported_view' if reviewed_unsupported_view else 'supported_view_unconfirmed')
        if not play_verified:
            blocked.append('normal_speed_match_play_unconfirmed')
        if not current['source_valid']:
            blocked.append('source_timing_unavailable')
        if not geometry_valid:
            blocked.append('fresh_court_calibration_required')
        transition = bool(allowed and self.previous and not reasons and current['transition_valid']
                          and self.last_result['analysis_allowed'])
        result = {'schemaVersion': 1, 'analysis_id': self.analysis_id, 'frame_index': current['frame_index'],
                  'timestamp_seconds': current['normalized_seconds'], 'source_pts': current['source_pts'],
                  'source_time_base': current['source_time_base'], 'source_frame_index': current['source_frame_index'],
                  'source_timestamp_seconds': float(current['source_time']) if current['source_time'] is not None else None,
                  'segment_id': self.segment_id, 'epoch': self.epoch, 'local_reset': bool(reasons),
                  'discontinuity_reasons': reasons, 'discontinuity_status': 'provisional' if reasons else 'none_observed',
                  'histogram_distance': histogram_distance, 'thumbnail_difference': thumbnail_difference,
                  'relative_zoom_change': zoom_change, 'reanchor_required': self.reanchor_required,
                  'needs_new_registration': needs_new_registration, 'anchor_frame_index': self.anchor_frame,
                  'view_status': 'supported_reviewed' if active_review is not None and active_review.viewpoint == SUPPORTED_VIEW else 'supported_qualified_geometry' if view_supported else 'unknown',
                  'activity_status': active_review.label if active_review else 'unknown',
                  'review_evidence_id': active_review.evidence_id if active_review else None,
                  'view_qualification_id': self.qualified_view_id,
                  'analysis_allowed': allowed, 'measurement_transition_allowed': transition,
                  'unavailable_reasons': blocked, 'classification_probability': None,
                  'scope': 'CLASSICAL_DISCONTINUITY_SUGGESTIONS_AND_REVIEW_GATES; NOT_REPLAY_OR_CAMERA_ACCURACY'}
        # CourtCameraRegistration starts with an identity anchor transform.
        # Reset that baseline when the caller is asked to construct one; an
        # old-anchor scale must not produce a false zoom on its next frame.
        next_zoom = 1. if needs_new_registration else scale
        self.previous, self.thumbnail, self.histogram, self.shape, self.zoom = current, thumb, histogram, (height, width), next_zoom
        self.last_fingerprint, self.last_result = fingerprint, result
        return json.loads(json.dumps(result))

    @staticmethod
    def gate_registration(interval, registration):
        """Fit the existing RegistrationResult seam; unknown intervals stay null."""
        if interval['analysis_allowed'] and registration.calibration.is_valid:
            return registration
        reason = ';'.join(interval['unavailable_reasons']) or 'INTERVAL_MEASUREMENT_UNAVAILABLE'
        audit = {**registration.audit, 'is_valid': False, 'rejection_reason': reason,
                 'image_to_court': None, 'anchor_to_frame_px': None,
                 'reinitialization_required': interval['reanchor_required'], 'camera_interval': interval}
        return RegistrationResult(CourtCalibration(None, rejection_reason=reason), None, audit)
