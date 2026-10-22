"""Meaningful contract/geometry checks; root must dispatch execution explicitly.

All source/annotation identities here are SYNTHETIC_TEST provenance. These
tests cannot establish real-image landmark correctness or product accuracy.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.court.calibration import calibrate_court
from src.court.court_geometry import TennisCourtGeometry


_SPEC = importlib.util.spec_from_file_location(
    "prepared_assisted_anchor_20261007", Path(__file__).with_name("assisted_anchor.py"))
assert _SPEC is not None and _SPEC.loader is not None
anchor_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = anchor_module
_SPEC.loader.exec_module(anchor_module)


def independently_project(points, matrix):
    """Analytic homogeneous projection, independent of OpenCV fitting/projecting."""
    projected = []
    for x, y in points:
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        projected.append([(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
                          (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator])
    return np.asarray(projected, dtype=float)


@pytest.fixture
def geometry():
    canonical = TennisCourtGeometry.get_canonical_keypoints().astype(float)
    matrix = np.array([[60., 8., 300.], [0., 30., 120.], [0., .012, 1.]])
    return canonical, independently_project(canonical, matrix), matrix


@pytest.fixture
def trusted_context():
    binding = anchor_module.SourceBinding("match_" + "1" * 32, "a" * 64, "b" * 64, 1920, 1080)
    annotator = anchor_module.AnnotationProvenance("TEST", "synthetic-contract-fixture", "SYNTHETIC_TEST")
    return binding, annotator


def payload_for(points, trusted_context, indices=(0, 1, 2, 3, 8, 11)):
    binding, annotator = trusted_context
    return {"schemaVersion": 1, "mode": "ASSISTED", "binding": binding.to_dict(),
            "annotator": annotator.to_dict(),
            "landmarks": [{"index": index, "imagePx": points[index].tolist(), "visible": True}
                          for index in indices]}


def parse(payload, trusted_context):
    binding, annotator = trusted_context
    return anchor_module.parse_assisted_anchor(payload, expected_binding=binding, expected_annotator=annotator)


def fit(payload, trusted_context, automatic_points, automatic_calibration):
    binding, annotator = trusted_context
    return anchor_module.fit_assisted_anchor(
        payload, expected_binding=binding, expected_annotator=annotator,
        automatic_image_points=automatic_points, automatic_calibration=automatic_calibration)


def test_six_supplied_points_recover_eight_unsupplied_landmarks(geometry, trusted_context):
    canonical, points, matrix = geometry
    payload = payload_for(points, trusted_context)
    automatic_points = points + [200., -60.]
    automatic_calibration = calibrate_court(automatic_points, canonical)
    automatic_before = deepcopy(automatic_calibration.to_dict())
    points_before = automatic_points.copy()
    payload_before = deepcopy(payload)
    result = fit(payload, trusted_context, automatic_points, automatic_calibration)
    assert result.assisted_calibration.is_valid
    assert result.assisted_calibration.inlier_count == 6
    assert len(result.assisted_calibration.inlier_mask) == 6
    held_out_indices = [4, 5, 6, 7, 9, 10, 12, 13]
    assert result.template_image_points[held_out_indices] == pytest.approx(points[held_out_indices], abs=.002)
    # Independent held-out ground point does not participate in landmark fitting.
    held_out_ground = np.array([[3.2, 13.7]])
    held_out_pixel = independently_project(held_out_ground, matrix)[0]
    assert result.assisted_calibration.project_ground_point(tuple(held_out_pixel)) == pytest.approx(held_out_ground[0], abs=.002)
    assert result.evidence["automatic"]["calibration"] == automatic_before
    assert automatic_calibration.to_dict() == automatic_before
    assert np.array_equal(automatic_points, points_before)
    assert payload == payload_before
    assert result.evidence["automatic"]["imageLandmarks"][0]["imagePx"] == automatic_points[0].tolist()
    assert result.evidence["assisted"]["suppliedLandmarks"][0]["imagePx"] == points[0].tolist()
    assert not result.template_image_points.flags.writeable
    json.dumps(result.evidence, allow_nan=False)


def test_invalid_automatic_fit_is_preserved_alongside_valid_assistance(geometry, trusted_context):
    canonical, points, _ = geometry
    automatic_points = points.copy()
    automatic_points[:, 1] = 100.
    automatic_calibration = calibrate_court(automatic_points, canonical)
    assert not automatic_calibration.is_valid
    result = fit(payload_for(points, trusted_context), trusted_context, automatic_points, automatic_calibration)
    assert result.assisted_calibration.is_valid
    assert result.evidence["automatic"]["calibration"] == automatic_calibration.to_dict()
    assert result.evidence["automatic"]["calibration"]["is_valid"] is False


def test_generated_template_points_never_claim_visibility_or_supplied_observation(geometry, trusted_context):
    canonical, _, _ = geometry
    matrix = np.array([[60., 8., -120.], [0., 30., 120.], [0., .012, 1.]])
    points = independently_project(canonical, matrix)
    indices = (8, 9, 10, 11, 12, 13)
    result = fit(payload_for(points, trusted_context, indices), trusted_context, points, calibrate_court(points, canonical))
    assert result.assisted_calibration.is_valid
    derived = result.evidence["assisted"]["derivedTemplateLandmarks"]
    assert len(derived) == 14
    assert derived[0]["imagePx"][0] < 0
    assert derived[0]["wasSupplied"] is False
    assert all(item["positionKind"] == "DERIVED_FROM_ASSISTED_FIT" and item["visibilityNotAsserted"] is True
               and "visible" not in item for item in derived)
    assert [item["index"] for item in result.evidence["assisted"]["suppliedLandmarks"]] == list(indices)


def test_seven_points_preserve_outlier_and_original_sparse_inlier_order(geometry, trusted_context):
    canonical, points, _ = geometry
    indices = (0, 1, 2, 3, 8, 10, 11)
    payload = payload_for(points, trusted_context, indices)
    payload["landmarks"][4]["imagePx"][0] += 150
    payload["landmarks"][4]["imagePx"][1] -= 90
    result = fit(payload, trusted_context, points, calibrate_court(points, canonical))
    assert result.assisted_calibration.is_valid
    assert result.assisted_calibration.inlier_count == 6
    supplied = result.evidence["assisted"]["suppliedLandmarks"]
    assert [item["index"] for item in supplied] == list(indices)
    assert supplied[4]["imagePx"] == payload["landmarks"][4]["imagePx"]
    assert supplied[4]["fitInlier"] is False
    assert all(item["fitInlier"] is True for offset, item in enumerate(supplied) if offset != 4)
    assert result.evidence["assisted"]["suppliedLandmarkFit"]["reprojection_error_px"] > 5


def test_six_points_do_not_relax_minimum_inliers_after_one_outlier(geometry, trusted_context):
    canonical, points, _ = geometry
    payload = payload_for(points, trusted_context)
    payload["landmarks"][4]["imagePx"][0] += 150
    payload["landmarks"][4]["imagePx"][1] -= 90
    automatic = calibrate_court(points, canonical)
    result = fit(payload, trusted_context, points, automatic)
    assert not result.assisted_calibration.is_valid
    assert result.assisted_calibration.rejection_reason == "INSUFFICIENT_INLIER_SUPPORT"
    assert result.template_image_points is None
    assert result.evidence["assisted"]["derivedTemplateLandmarks"] is None
    assert result.evidence["automatic"]["calibration"] == automatic.to_dict()
    assert result.evidence["effectiveMethod"] == "ASSISTED_VISIBLE_LANDMARKS"
    assert result.evidence["selectedCalibrationValid"] is False


def test_collinear_image_landmarks_reject_without_automatic_fallback(geometry, trusted_context):
    canonical, points, _ = geometry
    # Collapse six otherwise distinct image locations to one line; an accepted
    # automatic fit must remain separately visible when this correction fails.
    payload = payload_for(points, trusted_context)
    for offset, item in enumerate(payload["landmarks"]):
        item["imagePx"] = [100. + offset * 30., 200.]
    automatic = calibrate_court(points, canonical)
    result = fit(payload, trusted_context, points, automatic)
    assert not result.assisted_calibration.is_valid
    assert result.assisted_calibration.rejection_reason == "DEGENERATE_LANDMARKS"
    assert result.template_image_points is None
    assert result.evidence["automatic"]["calibration"]["is_valid"] is True
    assert result.evidence["selectedCalibrationValid"] is False


def test_visible_sparse_fit_cannot_emit_full_court_crossing_a_horizon(geometry, trusted_context):
    canonical, automatic_points, _ = geometry
    horizon_matrix = np.array([[100., 20., 0.], [0., 20., 0.], [0., 1., -10.]])
    points = independently_project(canonical, horizon_matrix)
    indices = (2, 3, 5, 7, 10, 11, 13)
    automatic = calibrate_court(automatic_points, canonical)
    result = fit(payload_for(points, trusted_context, indices), trusted_context, automatic_points, automatic)
    assert result.evidence["assisted"]["suppliedLandmarkFit"]["is_valid"] is True
    assert not result.assisted_calibration.is_valid
    assert result.assisted_calibration.rejection_reason == "ASSISTED_TEMPLATE_INTERSECTS_HORIZON"
    assert result.template_image_points is None
    assert result.evidence["automatic"]["calibration"] == automatic.to_dict()


def test_nonfinite_detector_output_remains_explicit_and_strict_json_serializable(geometry, trusted_context):
    canonical, points, _ = geometry
    automatic_points = points.copy()
    automatic_points[2, 0] = np.nan
    automatic = calibrate_court(automatic_points, canonical)
    result = fit(payload_for(points, trusted_context), trusted_context, automatic_points, automatic)
    observation = result.evidence["automatic"]["imageLandmarks"][2]
    assert observation["imagePx"] == [None, points[2, 1]]
    assert observation["finite"] is False
    assert result.evidence["automatic"]["pointSerialization"] == "NONFINITE_COMPONENTS_AS_NULL"
    assert result.evidence["automatic"]["calibration"] == automatic.to_dict()
    json.dumps(result.evidence, allow_nan=False)


@pytest.mark.parametrize("field,value", [
    ("parentAnalysisId", "match_" + "2" * 32),
    ("normalizedSourceSha256", "c" * 64),
    ("decodedFrameSha256", "d" * 64),
    ("frameIndex", 1), ("width", 1919), ("height", 1079),
    ("pixelFormat", "RGB_UINT8_ROW_MAJOR"),
    ("normalizedSourceSha256", "A" * 64),
])
def test_submission_cannot_change_verified_source_binding(geometry, trusted_context, field, value):
    payload = payload_for(geometry[1], trusted_context)
    payload["binding"][field] = value
    with pytest.raises(anchor_module.AssistedAnchorError, match="another analysis") as error:
        parse(payload, trusted_context)
    assert error.value.code == "SOURCE_BINDING_MISMATCH"


@pytest.mark.parametrize("field,value", [("frameIndex", False), ("frameIndex", 0.),
                                          ("width", True), ("height", 1080.)])
def test_binding_types_do_not_accept_numeric_equality_coercion(geometry, trusted_context, field, value):
    payload = payload_for(geometry[1], trusted_context)
    payload["binding"][field] = value
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        parse(payload, trusted_context)
    assert error.value.code == "INVALID_BINDING_TYPES"


@pytest.mark.parametrize("mutation", [
    "five_points", "fifteen_points", "duplicate_index", "descending_indices", "negative_index",
    "index_fourteen", "float_index", "boolean_index", "hidden", "numeric_visible", "duplicate_point",
    "negative_x", "edge_x", "edge_y", "boolean_coordinate", "string_coordinate", "nan_coordinate",
    "infinite_coordinate", "huge_integer_coordinate", "three_coordinates", "tuple_coordinates",
])
def test_invalid_landmark_observations_reject(geometry, trusted_context, mutation):
    payload = payload_for(geometry[1], trusted_context)
    items = payload["landmarks"]
    first = items[0]
    if mutation == "five_points":
        payload["landmarks"] = items[:5]
    elif mutation == "fifteen_points":
        payload["landmarks"] = [deepcopy(first)] * 15
    elif mutation == "duplicate_index":
        items[1]["index"] = first["index"]
    elif mutation == "descending_indices":
        payload["landmarks"] = items[::-1]
    elif mutation in {"negative_index", "index_fourteen", "float_index", "boolean_index"}:
        first["index"] = {"negative_index": -1, "index_fourteen": 14, "float_index": 0., "boolean_index": False}[mutation]
    elif mutation in {"hidden", "numeric_visible"}:
        first["visible"] = False if mutation == "hidden" else 1
    elif mutation == "duplicate_point":
        items[1]["imagePx"] = deepcopy(first["imagePx"])
    elif mutation in {"negative_x", "edge_x", "boolean_coordinate", "string_coordinate", "nan_coordinate",
                       "infinite_coordinate", "huge_integer_coordinate"}:
        first["imagePx"][0] = {"negative_x": -.1, "edge_x": 1920., "boolean_coordinate": False,
                                "string_coordinate": "300", "nan_coordinate": float("nan"),
                                "infinite_coordinate": float("inf"), "huge_integer_coordinate": 10 ** 400}[mutation]
    elif mutation == "edge_y":
        first["imagePx"][1] = 1080.
    elif mutation == "three_coordinates":
        first["imagePx"].append(0)
    elif mutation == "tuple_coordinates":
        first["imagePx"] = tuple(first["imagePx"])
    else:
        raise AssertionError("Unknown rejection fixture")
    with pytest.raises(anchor_module.AssistedAnchorError):
        parse(payload, trusted_context)


@pytest.mark.parametrize("section", ["root", "binding", "annotator", "landmark"])
def test_unknown_fields_cannot_inject_measurement_authority(geometry, trusted_context, section):
    payload = payload_for(geometry[1], trusted_context)
    target = {"root": payload, "binding": payload["binding"], "annotator": payload["annotator"],
              "landmark": payload["landmarks"][0]}[section]
    target["world3dEnabled"] = True
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        parse(payload, trusted_context)
    assert error.value.code == "INVALID_FIELDS"


@pytest.mark.parametrize("kind,method", [("AI_AGENT", "AI_IMAGE_REVIEW"), ("TEST", "SYNTHETIC_TEST"),
                                         ("LOCAL_OPERATOR", "LOCAL_POINT_SUBMISSION")])
def test_caller_declared_actor_cannot_claim_verified_user_provenance(geometry, trusted_context, kind, method):
    binding, _ = trusted_context
    actual = anchor_module.AnnotationProvenance(kind, "actual-annotator", method)
    payload = payload_for(geometry[1], (binding, actual))
    assert parse(payload, (binding, actual)).annotator.kind == kind
    payload["annotator"] = {"kind": "USER", "identifier": "actual-annotator", "method": "HUMAN_CLICK"}
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        parse(payload, (binding, actual))
    assert error.value.code == "ANNOTATOR_MISMATCH"


def test_local_operator_provenance_is_retained_without_claiming_verified_identity_or_accuracy(geometry, trusted_context):
    binding, _ = trusted_context
    actual = anchor_module.AnnotationProvenance("LOCAL_OPERATOR", "local-operator", "LOCAL_POINT_SUBMISSION")
    canonical, points, _ = geometry
    result = fit(payload_for(points, (binding, actual)), (binding, actual), points, calibrate_court(points, canonical))
    assert result.evidence["assisted"]["annotator"] == actual.to_dict()
    assert result.evidence["assisted"]["annotatorIdentityVerified"] is False
    assert result.evidence["assisted"]["annotatorProvenanceBasis"] == "CALLER_DECLARED_ACTOR_KIND_NOT_AUTHENTICATED_IDENTITY"
    assert result.evidence["assisted"]["automaticEvaluationEligible"] is False
    assert result.evidence["assisted"]["fitConsistencyOnly"] is True
    assert result.evidence["assisted"]["independentAccuracyValidated"] is False
    assert result.evidence["authority"] == {
        "independentCourtAccuracyValidated": False, "physicalBallSpeedEnabled": False, "world3dEnabled": False}


@pytest.mark.parametrize("section", ["schemaVersion", "frameIndex"])
def test_duplicate_json_keys_reject_before_normalization(geometry, trusted_context, section):
    raw = json.dumps(payload_for(geometry[1], trusted_context), separators=(",", ":"))
    value = 1 if section == "schemaVersion" else 0
    raw = raw.replace(f'"{section}":{value}', f'"{section}":{value},"{section}":{value}', 1)
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        parse(raw, trusted_context)
    assert error.value.code == "DUPLICATE_JSON_FIELD"


def test_equivalent_json_encodings_have_one_normalized_payload_digest(geometry, trusted_context):
    payload = payload_for(geometry[1], trusted_context)
    parsed = parse(payload, trusted_context)
    assert parse(json.dumps(payload, indent=2), trusted_context) == parsed
    assert parse(json.dumps(payload).encode("utf-8"), trusted_context) == parsed
    normalized = json.dumps(parsed.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert parsed.payload_sha256 == hashlib.sha256(normalized).hexdigest()


@pytest.mark.parametrize("payload", ["x" * 16385, b"x" * 16385, "[", b"\xff", "\ud800",
                                      '{"n":NaN}', [1], None, "[" * 2000 + "]" * 2000])
def test_invalid_or_unbounded_json_never_enters_calibration(trusted_context, payload):
    with pytest.raises(anchor_module.AssistedAnchorError):
        parse(payload, trusted_context)


@pytest.mark.parametrize("kind,identifier,method", [("USER", "actor", "AI_IMAGE_REVIEW"),
                                                    ("USER", "actor", "HUMAN_CLICK"),
                                                    ("LOCAL_OPERATOR", "actor", "HUMAN_CLICK"),
                                                    ("AI_AGENT", "actor", "HUMAN_CLICK"),
                                                    ("TEST", "actor", "HUMAN_CLICK"),
                                                    ("OTHER", "actor", "HUMAN_CLICK"),
                                                    ("TEST", "", "SYNTHETIC_TEST"),
                                                    ("TEST", " actor", "SYNTHETIC_TEST"),
                                                    ("TEST", "actor\n", "SYNTHETIC_TEST")])
def test_trusted_annotation_context_cannot_misdescribe_its_method(kind, identifier, method):
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        anchor_module.AnnotationProvenance(kind, identifier, method)
    assert error.value.code == "INVALID_ANNOTATOR"


@pytest.mark.parametrize("field,value", [("parent_analysis_id", "../match"),
                                         ("normalized_source_sha256", "A" * 64),
                                         ("decoded_frame_sha256", "short"),
                                         ("width", True), ("width", 3841), ("height", 0),
                                         ("height", 1080.), ("frame_index", 1),
                                         ("frame_index", False), ("pixel_format", "RGB_UINT8_ROW_MAJOR")])
def test_malformed_trusted_source_context_rejects(field, value):
    arguments = {"parent_analysis_id": "match_" + "1" * 32, "normalized_source_sha256": "a" * 64,
                 "decoded_frame_sha256": "b" * 64, "width": 1920, "height": 1080,
                 "frame_index": 0, "pixel_format": "BGR_UINT8_ROW_MAJOR"}
    arguments[field] = value
    with pytest.raises(anchor_module.AssistedAnchorError):
        anchor_module.SourceBinding(**arguments)


def test_a_payload_cannot_supply_its_own_trusted_context(geometry, trusted_context):
    payload = payload_for(geometry[1], trusted_context)
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        anchor_module.parse_assisted_anchor(payload, expected_binding=payload["binding"],
                                            expected_annotator=payload["annotator"])
    assert error.value.code == "INVALID_TRUSTED_CONTEXT"


@pytest.mark.parametrize("field,value", [("schemaVersion", True), ("schemaVersion", 1.),
                                         ("schemaVersion", 2), ("mode", "AUTOMATIC")])
def test_schema_contract_does_not_use_boolean_or_float_equality(geometry, trusted_context, field, value):
    payload = payload_for(geometry[1], trusted_context)
    payload[field] = value
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        parse(payload, trusted_context)
    assert error.value.code == "INVALID_SCHEMA"


def test_decoded_pixel_binding_uses_native_shape_exact_samples_and_stride_independent_bytes():
    frame = np.arange(8 * 6 * 3, dtype=np.uint8).reshape(8, 6, 3)[::2, ::2]
    contiguous = np.ascontiguousarray(frame)
    assert not frame.flags.c_contiguous
    assert anchor_module.decoded_frame_sha256(frame) == hashlib.sha256(contiguous.tobytes()).hexdigest()
    assert anchor_module.decoded_frame_sha256(contiguous) == anchor_module.decoded_frame_sha256(frame)
    before = frame.copy()
    binding = anchor_module.SourceBinding.from_decoded_frame("match_" + "1" * 32, "a" * 64, frame)
    assert (binding.width, binding.height, binding.frame_index, binding.pixel_format) == (3, 4, 0, "BGR_UINT8_ROW_MAJOR")
    assert binding.decoded_frame_sha256 == anchor_module.decoded_frame_sha256(frame)
    assert np.array_equal(frame, before)
    changed = contiguous.copy()
    changed[0, 0, 0] ^= 1
    assert anchor_module.decoded_frame_sha256(changed) != binding.decoded_frame_sha256


@pytest.mark.parametrize("frame", [np.zeros((3, 4, 3), dtype=float), np.zeros((3, 4), dtype=np.uint8),
                                    np.zeros((3, 4, 4), dtype=np.uint8), np.zeros((0, 4, 3), dtype=np.uint8),
                                    [[[1, 2, 3]]]])
def test_wrong_pixel_representation_cannot_be_a_trusted_anchor(frame):
    with pytest.raises(anchor_module.AssistedAnchorError) as error:
        anchor_module.decoded_frame_sha256(frame)
    assert error.value.code == "INVALID_DECODED_FRAME"
