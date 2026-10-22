"""Source-bound assisted court anchors; isolated preparation, not production.

The caller verifies source identity and declares actual annotation provenance.
Payload claims cannot replace that context. Actor labels do not authenticate a
human identity. All supplied landmarks must be visible native-pixel observations.
The resulting fit is not accuracy evidence. No video IO, model inference,
mutation or measurement authority lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

import numpy as np

from src.court.calibration import CourtCalibration, calibrate_court
from src.court.court_geometry import TennisCourtGeometry


MAXIMUM_PAYLOAD_BYTES = 16384
PIXEL_FORMAT = "BGR_UINT8_ROW_MAJOR"
_ANALYSIS_ID = re.compile(r"match_[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ANNOTATOR_METHODS = {
    "LOCAL_OPERATOR": "LOCAL_POINT_SUBMISSION",
    "AI_AGENT": "AI_IMAGE_REVIEW",
    "TEST": "SYNTHETIC_TEST",
}


class AssistedAnchorError(ValueError):
    """A bounded source/annotation contract violation with a stable reason."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise AssistedAnchorError(code, detail)


def _exact_fields(value: Any, expected: set[str], section: str) -> dict:
    _require(type(value) is dict and set(value) == expected,
             "INVALID_FIELDS", f"{section} requires exactly its declared fields.")
    return value


def _finite_number(value: Any) -> bool:
    # Python integers are finite even when too large to cast to a float. Bounds
    # are checked before casting, so oversized integers reject without overflow.
    return type(value) is int or type(value) is float and math.isfinite(value)


def _plain_identifier(value: Any) -> bool:
    return (type(value) is str and 0 < len(value) <= 160 and value == value.strip()
            and not any(ord(character) < 32 or ord(character) == 127 for character in value))


@dataclass(frozen=True)
class SourceBinding:
    parent_analysis_id: str
    normalized_source_sha256: str
    decoded_frame_sha256: str
    width: int
    height: int
    frame_index: int = 0
    pixel_format: str = PIXEL_FORMAT

    def __post_init__(self) -> None:
        _require(type(self.parent_analysis_id) is str and
                 _ANALYSIS_ID.fullmatch(self.parent_analysis_id) is not None,
                 "INVALID_PARENT_ANALYSIS", "Expected parent analysis identity is invalid.")
        for digest in (self.normalized_source_sha256, self.decoded_frame_sha256):
            _require(type(digest) is str and _SHA256.fullmatch(digest) is not None,
                     "INVALID_SOURCE_HASH", "Expected source hashes must be lowercase SHA-256.")
        _require(type(self.width) is int and 0 < self.width <= 3840 and
                 type(self.height) is int and 0 < self.height <= 2160,
                 "INVALID_NATIVE_SIZE", "Expected native frame dimensions are invalid.")
        _require(type(self.frame_index) is int and self.frame_index == 0,
                 "INVALID_ANCHOR_FRAME", "This assisted anchor supports frame zero only.")
        _require(self.pixel_format == PIXEL_FORMAT,
                 "INVALID_PIXEL_FORMAT", "Expected pixels must be uint8 BGR in row-major order.")

    def to_dict(self) -> dict:
        return {
            "parentAnalysisId": self.parent_analysis_id,
            "normalizedSourceSha256": self.normalized_source_sha256,
            "decodedFrameSha256": self.decoded_frame_sha256,
            "frameIndex": self.frame_index,
            "width": self.width,
            "height": self.height,
            "pixelFormat": self.pixel_format,
        }

    @classmethod
    def from_decoded_frame(cls, parent_analysis_id: str, normalized_source_sha256: str,
                           frame: np.ndarray) -> "SourceBinding":
        digest = decoded_frame_sha256(frame)
        height, width = frame.shape[:2]
        return cls(parent_analysis_id, normalized_source_sha256, digest, width, height)


@dataclass(frozen=True)
class AnnotationProvenance:
    kind: str
    identifier: str
    method: str

    def __post_init__(self) -> None:
        _require(type(self.kind) is str and self.kind in _ANNOTATOR_METHODS and
                 self.method == _ANNOTATOR_METHODS[self.kind] and
                 _plain_identifier(self.identifier),
                 "INVALID_ANNOTATOR", "Expected annotation provenance is invalid.")

    def to_dict(self) -> dict:
        return {"kind": self.kind, "identifier": self.identifier, "method": self.method}


@dataclass(frozen=True)
class SuppliedLandmark:
    index: int
    x_px: float
    y_px: float

    def to_dict(self) -> dict:
        return {"index": self.index, "imagePx": [self.x_px, self.y_px], "visible": True}


@dataclass(frozen=True)
class AssistedAnchor:
    binding: SourceBinding
    annotator: AnnotationProvenance
    landmarks: tuple[SuppliedLandmark, ...]
    payload_sha256: str

    def to_dict(self) -> dict:
        return {"schemaVersion": 1, "mode": "ASSISTED", "binding": self.binding.to_dict(),
                "annotator": self.annotator.to_dict(),
                "landmarks": [landmark.to_dict() for landmark in self.landmarks]}


@dataclass(frozen=True)
class AssistedAnchorFit:
    assisted_calibration: CourtCalibration
    template_image_points: np.ndarray | None
    evidence: dict


def decoded_frame_sha256(frame: np.ndarray) -> str:
    """Hash actual decoded BGR pixels without changing dimensions or samples."""
    _require(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
             frame.ndim == 3 and frame.shape[2] == 3 and
             0 < frame.shape[1] <= 3840 and 0 < frame.shape[0] <= 2160,
             "INVALID_DECODED_FRAME", "Decode a nonempty native uint8 BGR frame first.")
    return hashlib.sha256(frame.tobytes(order="C")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "DUPLICATE_JSON_FIELD", "Annotation JSON contains a repeated field.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AssistedAnchorError("NONFINITE_JSON", "Annotation JSON contains a nonfinite number.")


def _payload_object(payload: Any) -> dict:
    if type(payload) in (str, bytes):
        _require(len(payload) <= MAXIMUM_PAYLOAD_BYTES, "PAYLOAD_TOO_LARGE", "Annotation exceeds its size limit.")
        try:
            raw = payload.encode("utf-8") if type(payload) is str else payload
        except UnicodeError as error:
            raise AssistedAnchorError("INVALID_JSON", "Annotation is not valid UTF-8 JSON.") from error
        _require(len(raw) <= MAXIMUM_PAYLOAD_BYTES, "PAYLOAD_TOO_LARGE", "Annotation exceeds its size limit.")
        try:
            payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                                 parse_constant=_reject_constant)
        except (ValueError, UnicodeError, RecursionError) as error:
            if isinstance(error, AssistedAnchorError):
                raise
            raise AssistedAnchorError("INVALID_JSON", "Annotation is not valid JSON.") from error
    _require(type(payload) is dict, "INVALID_PAYLOAD", "Annotation must be one JSON object.")
    try:
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        raise AssistedAnchorError("INVALID_JSON_VALUE", "Annotation contains unsupported JSON values.") from error
    _require(len(encoded) <= MAXIMUM_PAYLOAD_BYTES, "PAYLOAD_TOO_LARGE", "Annotation exceeds its size limit.")
    return payload


def parse_assisted_anchor(payload: dict | str | bytes, *, expected_binding: SourceBinding,
                          expected_annotator: AnnotationProvenance) -> AssistedAnchor:
    """Validate against verified source binding and caller-declared actor context."""
    _require(type(expected_binding) is SourceBinding and type(expected_annotator) is AnnotationProvenance,
             "INVALID_TRUSTED_CONTEXT", "Verified source binding and annotator are required.")
    data = _exact_fields(_payload_object(payload),
                         {"schemaVersion", "mode", "binding", "annotator", "landmarks"}, "Annotation")
    _require(type(data["schemaVersion"]) is int and data["schemaVersion"] == 1 and data["mode"] == "ASSISTED",
             "INVALID_SCHEMA", "Use the assisted annotation schema version one.")
    binding = _exact_fields(data["binding"], set(expected_binding.to_dict()), "Source binding")
    _require(type(binding["frameIndex"]) is int and type(binding["width"]) is int and
             type(binding["height"]) is int, "INVALID_BINDING_TYPES", "Frame index and dimensions must be integers.")
    _require(binding == expected_binding.to_dict(), "SOURCE_BINDING_MISMATCH",
             "Annotation belongs to another analysis, source, decoded frame or native size.")
    annotator = _exact_fields(data["annotator"], set(expected_annotator.to_dict()), "Annotator")
    _require(annotator == expected_annotator.to_dict(), "ANNOTATOR_MISMATCH",
             "Annotation provenance does not match the trusted annotator.")
    landmarks = data["landmarks"]
    _require(type(landmarks) is list and 6 <= len(landmarks) <= 14,
             "INSUFFICIENT_VISIBLE_LANDMARKS", "Supply six to fourteen visible indexed landmarks.")
    parsed, previous_index, locations = [], -1, set()
    for item in landmarks:
        item = _exact_fields(item, {"index", "imagePx", "visible"}, "Landmark")
        index, image = item["index"], item["imagePx"]
        _require(type(index) is int and 0 <= index < 14 and index > previous_index,
                 "INVALID_LANDMARK_ORDER", "Landmark indices must be distinct and strictly increasing from zero to thirteen.")
        _require(item["visible"] is True, "HIDDEN_LANDMARK", "Only visible supplied landmarks may enter this fit.")
        _require(type(image) is list and len(image) == 2 and all(_finite_number(value) for value in image),
                 "INVALID_IMAGE_POINT", "Each image point must contain two finite numbers.")
        x_px, y_px = image
        _require(0 <= x_px < expected_binding.width and 0 <= y_px < expected_binding.height,
                 "POINT_OUT_OF_BOUNDS", "Supplied landmarks must lie inside the native anchor image.")
        location = (float(x_px), float(y_px))
        _require(location not in locations, "DUPLICATE_IMAGE_POINT", "Distinct landmarks cannot share an image point.")
        locations.add(location)
        parsed.append(SuppliedLandmark(index, *location))
        previous_index = index
    normalized = {"schemaVersion": 1, "mode": "ASSISTED", "binding": expected_binding.to_dict(),
                  "annotator": expected_annotator.to_dict(), "landmarks": [item.to_dict() for item in parsed]}
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()
    return AssistedAnchor(expected_binding, expected_annotator, tuple(parsed), digest)


def _derived_template(calibration: CourtCalibration, canonical: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    if not calibration.is_valid:
        return None, None
    try:
        court_to_image = np.linalg.inv(calibration.image_to_court)
        homogeneous = np.column_stack((canonical, np.ones(len(canonical)))) @ court_to_image.T
        denominators = homogeneous[:, 2]
        if not np.isfinite(homogeneous).all():
            return None, "ASSISTED_TEMPLATE_NONFINITE"
        # A homography may fit all visible points while placing a horizon inside
        # the unobserved portion of the full court. It cannot supply a safe full
        # registration polygon in that case. This is topology, not accuracy.
        if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
            return None, "ASSISTED_TEMPLATE_INTERSECTS_HORIZON"
        template = homogeneous[:, :2] / denominators[:, None]
        if not np.isfinite(template).all():
            return None, "ASSISTED_TEMPLATE_NONFINITE"
        template.setflags(write=False)
        return template, None
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None, "ASSISTED_TEMPLATE_PROJECTION_FAILED"


def fit_assisted_anchor(payload: dict | str | bytes, *, expected_binding: SourceBinding,
                        expected_annotator: AnnotationProvenance,
                        automatic_image_points: np.ndarray,
                        automatic_calibration: CourtCalibration) -> AssistedAnchorFit:
    """Fit only supplied points and return separate automatic/assisted evidence.

    Contract violations raise AssistedAnchorError. A geometrically unsupported
    annotation returns an invalid assisted calibration with its rejection reason;
    it never selects automatic fallback. The caller must preserve the parent
    result and present failed corrections as such.
    """
    anchor = parse_assisted_anchor(payload, expected_binding=expected_binding,
                                   expected_annotator=expected_annotator)
    _require(type(automatic_calibration) is CourtCalibration,
             "INVALID_AUTOMATIC_EVIDENCE", "The original automatic calibration is required separately.")
    try:
        automatic_points = np.asarray(automatic_image_points, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise AssistedAnchorError("INVALID_AUTOMATIC_EVIDENCE", "Original automatic landmarks are invalid.") from error
    _require(automatic_points.shape == (14, 2), "INVALID_AUTOMATIC_EVIDENCE",
             "Preserve the original fourteen indexed automatic landmarks.")
    canonical = TennisCourtGeometry.get_canonical_keypoints().astype(np.float64)
    indices = [landmark.index for landmark in anchor.landmarks]
    supplied_image = np.array([[landmark.x_px, landmark.y_px] for landmark in anchor.landmarks], dtype=np.float64)
    # No settings or augmented points are passed: production fit defaults apply.
    supplied_fit = calibrate_court(supplied_image, canonical[indices])
    template, projection_rejection = _derived_template(supplied_fit, canonical)
    effective_calibration = supplied_fit
    if projection_rejection is not None:
        effective_calibration = CourtCalibration(
            image_to_court=None, inlier_mask=supplied_fit.inlier_mask,
            reprojection_error_px=supplied_fit.reprojection_error_px,
            reprojection_error_m=supplied_fit.reprojection_error_m,
            inlier_reprojection_error_px=supplied_fit.inlier_reprojection_error_px,
            rejection_reason=projection_rejection,
        )
    automatic_finite = np.isfinite(automatic_points)
    automatic_landmarks = [
        {"index": index,
         "imagePx": [float(value) if automatic_finite[index, axis] else None
                     for axis, value in enumerate(location)],
         "observationKind": "AUTOMATIC_MODEL_OUTPUT", "finite": bool(automatic_finite[index].all())}
        for index, location in enumerate(automatic_points)
    ]
    inlier_mask = supplied_fit.inlier_mask
    supplied_landmarks = []
    for position, landmark in enumerate(anchor.landmarks):
        supplied_landmarks.append({**landmark.to_dict(), "observationKind": "SUPPLIED_VISIBLE_POINT",
                                   "fitInlier": bool(inlier_mask[position]) if position < len(inlier_mask) else None})
    derived_landmarks = None
    if template is not None:
        supplied_indices = set(indices)
        derived_landmarks = [
            {"index": index, "imagePx": location.tolist(),
             "positionKind": "DERIVED_FROM_ASSISTED_FIT", "wasSupplied": index in supplied_indices,
             "visibilityNotAsserted": True}
            for index, location in enumerate(template)
        ]
    evidence = {
        "schemaVersion": 1,
        "effectiveMethod": "ASSISTED_VISIBLE_LANDMARKS",
        "selectedCalibrationValid": effective_calibration.is_valid,
        "automatic": {
            "method": "AUTOMATIC_COURT_MODEL", "imageLandmarks": automatic_landmarks,
            "calibration": automatic_calibration.to_dict(),
            "pointSerialization": "FINITE_NATIVE_PIXELS" if automatic_finite.all() else "NONFINITE_COMPONENTS_AS_NULL",
            "independentAccuracyValidated": False,
        },
        "assisted": {
            "method": "ASSISTED_VISIBLE_LANDMARKS", "binding": anchor.binding.to_dict(),
            "annotator": anchor.annotator.to_dict(),
            "annotatorIdentityVerified": False,
            "annotatorProvenanceBasis": "CALLER_DECLARED_ACTOR_KIND_NOT_AUTHENTICATED_IDENTITY",
            "normalizedPayloadSha256": anchor.payload_sha256,
            "suppliedLandmarks": supplied_landmarks,
            "suppliedLandmarkFit": supplied_fit.to_dict(),
            "effectiveCalibration": effective_calibration.to_dict(),
            "derivedTemplateLandmarks": derived_landmarks,
            "fitConsistencyOnly": True,
            "independentAccuracyValidated": False,
            "automaticEvaluationEligible": False,
        },
        "authority": {
            "independentCourtAccuracyValidated": False,
            "physicalBallSpeedEnabled": False,
            "world3dEnabled": False,
        },
        "coordinateSemantics": "COURT_PLANE_FROM_ASSISTED_SUPPLIED_POINT_FIT_NOT_3D_RECONSTRUCTION",
    }
    return AssistedAnchorFit(effective_calibration, template, evidence)
