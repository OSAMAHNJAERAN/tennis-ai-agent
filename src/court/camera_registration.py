"""Anchor-frame court-plane registration using observed image features.

Sparse optical flow is checked forward/backward and fitted with RANSAC. Features
are restricted to the initial court polygon and exclude detected people. A lost
registration latches until explicit reinitialization; this is not automatic
camera-cut recognition or independently validated geometric accuracy.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.court.calibration import CourtCalibration


@dataclass
class RegistrationResult:
    calibration: CourtCalibration
    anchor_to_frame_px: object
    audit: dict


class CourtCameraRegistration:
    def __init__(self, anchor_calibration, court_keypoints, reference_width=960, minimum_features=8,
                 minimum_inlier_ratio=.65, maximum_fb_error=1.5, maximum_patch_error=20.,
                 ransac_threshold=1.5, minimum_coverage=.1):
        values = (minimum_inlier_ratio, maximum_fb_error, maximum_patch_error, ransac_threshold, minimum_coverage)
        if (not np.isfinite(values).all() or reference_width < 64 or minimum_features < 4
                or not 0 < minimum_inlier_ratio <= 1 or not 0 < minimum_coverage <= 1
                or min(maximum_fb_error, maximum_patch_error, ransac_threshold) <= 0):
            raise ValueError('Invalid court registration settings')
        self.anchor_calibration = anchor_calibration
        self.court_keypoints = np.asarray(court_keypoints, np.float32)
        self.reference_width, self.minimum_features = reference_width, minimum_features
        self.minimum_inlier_ratio, self.maximum_fb_error = minimum_inlier_ratio, maximum_fb_error
        self.maximum_patch_error, self.ransac_threshold = maximum_patch_error, ransac_threshold
        self.minimum_coverage = minimum_coverage
        self.index, self.anchor_gray, self.anchor_points, self.guess = -1, None, None, None
        self.lost_reason = None

    def _person_mask(self, shape, boxes):
        mask = np.full(shape, 255, np.uint8)
        for box in boxes:
            coords = np.asarray([box.x1, box.y1, box.x2, box.y2]) * self.scale
            if np.isfinite(coords).all():
                x1, y1, x2, y2 = np.rint(coords).astype(int)
                cv2.rectangle(mask, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), 0, -1)
        return mask

    def _result(self, reason=None, transform=None, candidates=0, inliers=0, error=None, coverage=None):
        valid = reason is None
        matrix = None
        if valid:
            try:
                matrix = self.anchor_calibration.image_to_court @ np.linalg.inv(transform)
                if not np.isfinite(matrix).all():
                    raise ValueError('Nonfinite camera transform')
            except (np.linalg.LinAlgError, ValueError):
                valid, reason = False, 'DEGENERATE_CAMERA_TRANSFORM'
        if not valid:
            self.lost_reason = reason
        calibration = CourtCalibration(matrix if valid else None, rejection_reason=reason)
        drift = None
        if valid:
            projected = cv2.perspectiveTransform(self.court_keypoints[None], transform)[0]
            drift = float(np.max(np.linalg.norm(projected - self.court_keypoints, axis=1)))
        audit = {'frame_index': self.index, 'is_valid': valid, 'rejection_reason': reason,
                 'tracked_feature_count': candidates, 'inlier_count': inliers,
                 'inlier_mean_residual_reference_px': error, 'anchor_court_feature_coverage': coverage,
                 'max_landmark_displacement_source_px': drift,
                 'anchor_to_frame_px': transform.tolist() if valid else None,
                 'image_to_court': matrix.tolist() if valid else None,
                 'reference_width': self.reference_width,
                 'validity_scope': 'ANCHOR_COURT_FEATURE_REGISTRATION; NOT_INDEPENDENT_COURT_ACCURACY',
                 'reinitialization_required': not valid}
        return RegistrationResult(calibration, transform if valid else None, audit)

    def update(self, frame, people=()):
        self.index += 1
        if self.lost_reason:
            return self._result(self.lost_reason)
        if not self.anchor_calibration.is_valid:
            return self._result('INVALID_ANCHOR_CALIBRATION')
        height, width = frame.shape[:2]
        if self.anchor_gray is None:
            self.source_shape = (height, width)
            self.scale = min(1., self.reference_width / width)
        elif self.source_shape != (height, width):
            return self._result('SOURCE_RESOLUTION_CHANGED')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (round(width * self.scale), round(height * self.scale)))
        people_mask = self._person_mask(gray.shape, people)
        if self.anchor_gray is None:
            self.anchor_gray = gray
            polygon = cv2.convexHull(self.court_keypoints * self.scale)
            self.court_area = float(cv2.contourArea(polygon))
            court_mask = np.zeros_like(gray)
            cv2.fillConvexPoly(court_mask, np.rint(polygon).astype(np.int32), 255)
            court_mask = cv2.bitwise_and(court_mask, people_mask)
            self.anchor_points = cv2.goodFeaturesToTrack(gray, maxCorners=350, qualityLevel=.01,
                                                        minDistance=7, mask=court_mask, blockSize=5)
            if self.anchor_points is not None:
                self.guess = self.anchor_points.copy()
            # Initial landmark fit is independently available at frame zero.
            return self._result(transform=np.eye(3), candidates=0 if self.anchor_points is None else len(self.anchor_points))
        if self.anchor_points is None or len(self.anchor_points) < self.minimum_features or self.court_area <= 0:
            return self._result('INSUFFICIENT_ANCHOR_FEATURES')
        settings = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        current, status, errors = cv2.calcOpticalFlowPyrLK(self.anchor_gray, gray, self.anchor_points,
                                                         self.guess.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **settings)
        if current is None:
            return self._result('OPTICAL_FLOW_FAILED')
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.anchor_gray, current, None, **settings)
        if backward is None:
            return self._result('BACKWARD_FLOW_FAILED')
        a, b = self.anchor_points[:, 0], current[:, 0]
        fb = np.linalg.norm(backward[:, 0] - a, axis=1)
        keep = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                & np.isfinite(b).all(axis=1) & np.isfinite(fb) & (fb <= self.maximum_fb_error)
                & (errors.ravel() <= self.maximum_patch_error))
        for index in np.flatnonzero(keep):
            x, y = np.rint(b[index]).astype(int)
            if not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]) or people_mask[y, x] == 0:
                keep[index] = False
        if keep.sum() < self.minimum_features:
            return self._result('INSUFFICIENT_TRACKED_FEATURES', candidates=int(keep.sum()))
        transform, inlier_mask = cv2.findHomography(a[keep], b[keep], cv2.RANSAC, self.ransac_threshold)
        if transform is None or inlier_mask is None:
            return self._result('CAMERA_HOMOGRAPHY_FAILED', candidates=int(keep.sum()))
        inlier_mask = inlier_mask.ravel().astype(bool)
        count = int(inlier_mask.sum())
        if count < self.minimum_features or inlier_mask.mean() < self.minimum_inlier_ratio:
            return self._result('INSUFFICIENT_CAMERA_INLIERS', candidates=int(keep.sum()), inliers=count)
        anchor_inliers = a[keep][inlier_mask]
        coverage = cv2.contourArea(cv2.convexHull(anchor_inliers)) / self.court_area
        if coverage < self.minimum_coverage:
            return self._result('CAMERA_FEATURES_TOO_LOCALIZED', candidates=int(keep.sum()), inliers=count, coverage=coverage)
        projected = cv2.perspectiveTransform(anchor_inliers[None], transform)[0]
        residual = float(np.linalg.norm(projected - b[keep][inlier_mask], axis=1).mean())
        self.guess = cv2.perspectiveTransform(self.anchor_points, transform)
        scaling = np.diag([self.scale, self.scale, 1.])
        native_transform = np.linalg.inv(scaling) @ transform @ scaling
        return self._result(transform=native_transform, candidates=int(keep.sum()), inliers=count,
                            error=residual, coverage=float(coverage))
