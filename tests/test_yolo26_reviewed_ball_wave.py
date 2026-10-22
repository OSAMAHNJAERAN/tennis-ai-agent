"""Reviewed ball preparation must preserve event isolation and native coordinates.

These tests use small in-memory records. They do not open publisher media, import
image/model runtimes, or create supervised labels from the real review wave.
"""

from copy import deepcopy
import json
import math
import subprocess
import sys

import pytest

from scripts.data.prepare_yolo26_reviewed_ball_wave import (
    build_components,
    jittered_crop,
    normalize_positive,
    translate_box,
    validate_manual_box,
)


def _group(name, source_ids, split="train", exclusion_reason=None):
    return {
        "group": name,
        "source_ids": source_ids,
        "proposed_split": split,
        "exclusion_reason": exclusion_reason,
    }


def _sources(groups):
    return [
        {
            "source_id": source_id,
            "conservative_event_group": group["group"],
            "proposed_split": group["proposed_split"],
            "exclusion_reason": None,
        }
        for group in groups
        for source_id in group["source_ids"]
    ]


@pytest.mark.parametrize(
    "training_event,development_event", [("miami", "washington"), ("rome", "dubai")]
)
@pytest.mark.parametrize("decision", ["REJECT_AMBIGUOUS", "REJECT_UNSUPPORTED_VIEW"])
def test_rejected_observation_still_joins_event_groups_and_preserves_development(
    training_event, development_event, decision
):
    groups = [
        _group(training_event, ["training_a", "training_b"]),
        _group(development_event, ["development_a", "development_b"], "development"),
    ]
    observations = [
        {
            "source_id": "training_b",
            "decision": decision,
            "observed_event_family": development_event,
        }
    ]

    components = build_components(groups, _sources(groups), observations)

    assert len(components) == 1
    component = components[0]
    assert set(component["aliases"]) == {training_event, development_event}
    assert set(component["source_ids"]) == {
        "training_a", "training_b", "development_a", "development_b"
    }
    assert component["derived_split"] == "development"
    assert component["exclusion_reasons"] == []


def test_rejected_and_low_view_edges_form_transitive_component_without_changing_inputs():
    groups = [
        _group("miami", ["m1", "m2"]),
        _group("washington", ["w1"]),
        _group("rome", ["r1", "r2"]),
        _group("dubai", ["d1"], "development"),
        _group("halle", ["h1"]),
    ]
    sources = _sources(groups)
    observations = [
        {"source_id": "m2", "decision": "REJECT_AMBIGUOUS", "observed_event_family": "washington"},
        {"source_id": "w1", "decision": "REJECT_UNSUPPORTED_VIEW", "observed_event_family": "rome"},
        {"source_id": "r2", "decision": "ACCEPT_VISIBLE_BOX", "observed_event_family": "dubai"},
    ]
    original = deepcopy((groups, sources, observations))

    components = build_components(groups, sources, observations)
    merged = next(item for item in components if "miami" in item["aliases"])
    isolated = next(item for item in components if "halle" in item["aliases"])

    assert len(components) == 2
    assert set(merged["aliases"]) == {"miami", "washington", "rome", "dubai"}
    assert set(merged["source_ids"]) == {"m1", "m2", "w1", "r1", "r2", "d1"}
    assert merged["derived_split"] == "development"
    assert isolated["source_ids"] == ["h1"]
    assert isolated["derived_split"] == "train"
    assert (groups, sources, observations) == original


def test_group_quarantine_dominates_development_after_union():
    groups = [
        _group("unknown_event", ["unknown"], exclusion_reason="Event provenance unresolved"),
        _group("washington", ["dev"], "development"),
    ]
    components = build_components(
        groups,
        _sources(groups),
        [{"source_id": "unknown", "decision": "REJECT_AMBIGUOUS", "observed_event_family": "washington"}],
    )

    assert len(components) == 1
    assert components[0]["derived_split"] == "quarantine"
    assert "Event provenance unresolved" in components[0]["exclusion_reasons"]


def test_source_only_exclusion_does_not_quarantine_other_sources_in_event_group():
    groups = [_group("rome", ["primary_camera", "closeup"])]
    sources = _sources(groups)
    sources[1]["exclusion_reason"] = "Unsupported close-up camera"

    components = build_components(groups, sources, [])

    assert len(components) == 1
    assert set(components[0]["source_ids"]) == {"primary_camera", "closeup"}
    assert components[0]["derived_split"] == "train"
    assert components[0]["exclusion_reasons"] == []
    assert sources[1]["exclusion_reason"] == "Unsupported close-up camera"


@pytest.mark.parametrize("source_split", ["development", "test"])
def test_source_split_cannot_be_downgraded_by_inconsistent_training_group(source_split):
    groups = [_group("rome", ["source"])]
    sources = _sources(groups)
    sources[0]["proposed_split"] = source_split

    with pytest.raises(ValueError):
        build_components(groups, sources, [])


def test_component_ids_and_assignments_are_independent_of_input_order():
    groups = [
        _group("miami", ["m2", "m1"]),
        _group("washington", ["w1"], "development"),
        _group("halle", ["h1"]),
    ]
    sources = _sources(groups)
    observations = [
        {"source_id": "m2", "decision": "REJECT_AMBIGUOUS", "observed_event_family": "washington"}
    ]

    def signature(components):
        return sorted(
            (item["component_id"], tuple(sorted(item["aliases"])),
             tuple(sorted(item["source_ids"])), item["derived_split"])
            for item in components
        )

    assert signature(build_components(groups, sources, observations)) == signature(
        build_components(list(reversed(groups)), list(reversed(sources)), list(reversed(observations)))
    )


@pytest.mark.parametrize(
    "box", [[0, 0, 4, 8], [1914, 1072, 1920, 1080], [100, 1074, 112, 1080], [1914, 10, 1920, 18]]
)
def test_jittered_native_crop_contains_edge_ball_and_round_trips_exactly(box):
    crop = jittered_crop(box, "edge-proposal", seed=709)

    assert len(crop) == 4
    assert all(type(value) is int for value in crop)
    assert 0 <= crop[0] <= 1280
    assert 0 <= crop[1] <= 440
    assert crop[2] - crop[0] == crop[3] - crop[1] == 640
    assert crop[0] <= box[0] < box[2] <= crop[2]
    assert crop[1] <= box[1] < box[3] <= crop[3]
    local = translate_box(box, crop[:2])
    assert translate_box(local, crop[:2], inverse=True) == box
    assert jittered_crop(box, "edge-proposal", seed=709) == crop


def test_jitter_varies_ball_placement_by_proposal_and_seed_instead_of_oracle_centering():
    box = [950, 530, 962, 538]
    crops = [jittered_crop(box, f"proposal-{index}", seed=709) for index in range(16)]
    placements = {(box[0] - crop[0], box[1] - crop[1]) for crop in crops}

    assert len(placements) >= 8
    assert any(abs((box[0] + box[2]) / 2 - crop[0] - 320) > 30 for crop in crops)
    assert jittered_crop(box, "same-proposal", seed=709) != jittered_crop(box, "same-proposal", seed=710)


@pytest.mark.parametrize(
    "box,kwargs",
    [
        ([1, 2, 3, math.nan], {}),
        ([1, 2, math.inf, 4], {}),
        ([False, 2, 3, 4], {}),
        ([1, 2, 1, 4], {}),
        ([4, 2, 1, 4], {}),
        ([-1, 2, 3, 4], {}),
        ([1919, 2, 1921, 4], {}),
        ([1, 1079, 3, 1081], {}),
        ([0, 0, 641, 8], {}),
        ([0, 0, 8, 641], {}),
        ([1, 2, 3], {}),
        ([1, 2, 3, 4], {"width": 639}),
        ([1, 2, 3, 4], {"height": 639}),
        ([1, 2, 3, 4], {"size": 0}),
        ([1, 2, 3, 4], {"width": True}),
        ([1, 2, 3, 4], {"size": 640.5}),
    ],
)
def test_impossible_or_nonfinite_native_crops_are_rejected(box, kwargs):
    with pytest.raises((TypeError, ValueError)):
        jittered_crop(box, "invalid-proposal", **kwargs)


def test_crop_translation_preserves_fractional_extent_without_rounding_or_mutation():
    native = [100.25, 201.5, 113.75, 206.0]
    origin = [80, 180]
    assert translate_box(native, origin) == [20.25, 21.5, 33.75, 26.0]
    assert translate_box([20.25, 21.5, 33.75, 26.0], origin, inverse=True) == native
    assert native == [100.25, 201.5, 113.75, 206.0]
    assert origin == [80, 180]


@pytest.mark.parametrize(
    "box,origin",
    [
        ([1, 2, 3, math.nan], [0, 0]),
        ([1, 2, 3, 4], [math.inf, 0]),
        ([True, 2, 3, 4], [0, 0]),
        ([1, 2, 3, 4], [False, 0]),
        ([1, 2, 1, 4], [0, 0]),
        ([1, 2, 3, 4], [0]),
    ],
)
def test_crop_translation_rejects_invalid_numbers_or_shapes(box, origin):
    with pytest.raises((TypeError, ValueError)):
        translate_box(box, origin)


@pytest.mark.parametrize(
    "box,center,patch,origin",
    [
        ([920, 290, 928, 297], [924, 293.5], [44, 44, 52, 51], [876, 246]),
        ([872, 305, 880, 312], [876, 308.5], [44, 45, 52, 52], [828, 260]),
        ([1160, 375, 1182, 383], [1171, 379], [30, 25, 52, 33], [1130, 350]),
        ([1916, 1077, 1920, 1080], [1918, 1078.5], [92, 93, 96, 96], [1824, 984]),
    ],
)
def test_manual_native_box_accepts_variable_visible_extent_and_exact_patch_inverse(box, center, patch, origin):
    validate_manual_box(box, center, patch, origin)


@pytest.mark.parametrize(
    "box,center,patch,origin",
    [
        ([920, 290, 928, 297], [924, 294], [44, 44, 52, 51], [876, 246]),
        ([920, 290, 928, 297], [924, 293.5], [44, 44, 52, 52], [876, 246]),
        ([920, 290, 928, 297], [924, 293.5], [44, 44, 52, 51], [877, 246]),
        ([920, 290, 928, math.nan], [924, 293.5], [44, 44, 52, 51], [876, 246]),
        ([920, 290, 928, 297], [924, math.inf], [44, 44, 52, 51], [876, 246]),
        ([920, 290, 928, 297], [924, 293.5], [True, 44, 52, 51], [876, 246]),
        ([-1, 0, 3, 4], [1, 2], [-1, 0, 3, 4], [0, 0]),
        ([1918, 0, 1922, 4], [1920, 2], [94, 0, 98, 4], [1824, 0]),
        ([0, 1078, 4, 1082], [2, 1080], [0, 94, 4, 98], [0, 984]),
        ([3, 4, 3, 8], [3, 6], [3, 4, 3, 8], [0, 0]),
    ],
)
def test_manual_box_must_not_silently_recenter_clip_or_repair_invalid_review(box, center, patch, origin):
    with pytest.raises((TypeError, ValueError)):
        validate_manual_box(box, center, patch, origin)


def _review_pair(style):
    proposal = {
        "proposal_id": "positive_0007",
        "source_id": "match172",
        "clip": "match172_000",
        "frame_index": 75,
        "source_pts": 38400,
        "source_time_base": "1/12800",
        "timestamp_fraction_seconds": "3",
        "conservative_event_group": "rome",
        "proposed_split": "train",
        "native_patch_origin_xy_px": [1130, 350],
        "images": {"native_frame": {"path": "synthetic-unopened.png", "sha256": "a" * 64}},
    }
    observation = {
        key: proposal[key]
        for key in ("proposal_id", "source_id", "clip", "frame_index", "source_pts", "source_time_base")
    }
    observation.update({
        "frozen_conservative_event_group": "rome",
        "frozen_proposed_split": "train",
        "native_patch_origin_xy_px": [1130, 350],
        "center_definition": "MIDPOINT_OF_MANUALLY_REVIEWED_VISIBLE_EXTENT_NOT_PHYSICAL_3D_CENTER",
        "reviewer": f"AGENT:synthetic-{style}-reviewer",
        "independent_human_review": False,
        "reason": "Synthetic literal visible blur extent for schema validation only",
        "visibility": "VISIBLE_BLURRED",
        "blur_severity": "SEVERE",
    })
    if style == "head":
        observation.update({
            "selection_index": 7,
            "decision": "ACCEPT_VISIBLE_BBOX",
            "accept_visible_bbox_xyxy_native_px": [1160, 375, 1182, 383],
            "observed_visible_extent_xyxy_native_px": [1160, 375, 1182, 383],
            "observed_center_xy_native_px": [1171, 379],
            "manually_selected_bbox_xyxy_patch_px": [30, 25, 52, 33],
            "observed_event_identity": "DUBAI_DUTY_FREE_TENNIS_STADIUM_BRANDING",
            "component_hypothesis_autoaccepted": False,
            "primary_camera_scope": True,
        })
    else:
        observation.update({
            "manifest_positive_index": 7,
            "decision": "ACCEPT_VISIBLE_BOX",
            "bbox_xyxy_native_px": [1160, 375, 1182, 383],
            "visible_center_xy_native_px": [1171, 379],
            "manual_bbox_xyxy_patch_px": [30, 25, 52, 33],
            "observed_event_group": "dubai",
            "component_box_autoaccepted": False,
        })
    return proposal, observation


@pytest.mark.parametrize("style", ["head", "tail"])
def test_both_actual_review_schemas_preserve_manual_extent_originals_and_agent_status(style):
    proposal, observation = _review_pair(style)
    before = deepcopy((proposal, observation))

    normalized = normalize_positive(proposal, observation, 7)

    assert normalized["decision"] == "ACCEPT_VISIBLE_BOX"
    assert normalized["bbox_xyxy_native_px"] == [1160, 375, 1182, 383]
    assert normalized["visible_center_xy_native_px"] == [1171, 379]
    assert normalized["source_pts"] == 38400
    assert normalized["source_time_base"] == "1/12800"
    assert normalized["timestamp_fraction_seconds"] == "3"
    assert normalized["frozen_conservative_event_group"] == "rome"
    assert normalized["frozen_proposed_split"] == "train"
    assert normalized["observed_event_family"] == "dubai"
    assert normalized["reviewer"] == observation["reviewer"]
    assert normalized["reviewer_type"] == "AGENT"
    assert normalized["independent_human_review"] is False
    assert normalized["supervised_export_approved"] is False
    assert normalized["complete_positive_crop_reviewed"] is False
    assert normalized["original_observation"] == observation
    assert (proposal, observation) == before
    observation["reason"] = "Changed after normalization"
    assert normalized["original_observation"]["reason"] == before[1]["reason"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposal_id", "another-proposal"),
        ("source_id", "another-source"),
        ("clip", "another-clip"),
        ("frame_index", 76),
        ("source_pts", 38401),
        ("source_time_base", "1/25"),
        ("manifest_positive_index", 8),
        ("frozen_conservative_event_group", "miami"),
        ("frozen_proposed_split", "development"),
        ("native_patch_origin_xy_px", [1131, 350]),
        ("center_definition", "PUBLISHER_CENTER_WITHOUT_MANUAL_EXTENT"),
        ("independent_human_review", True),
        ("component_box_autoaccepted", True),
    ],
)
def test_normalization_rejects_changed_source_bindings_and_invented_review_provenance(field, value):
    proposal, observation = _review_pair("tail")
    observation[field] = value

    with pytest.raises(ValueError):
        normalize_positive(proposal, observation, 7)


@pytest.mark.parametrize("style", ["head", "tail"])
@pytest.mark.parametrize("field", ["frame_index", "source_pts"])
@pytest.mark.parametrize("invalid", [True, 1.0])
@pytest.mark.parametrize("invalid_record", ["proposal", "observation"])
def test_normalization_does_not_accept_boolean_or_float_timing_as_equal_to_integer(
    style, field, invalid, invalid_record
):
    proposal, observation = _review_pair(style)
    proposal[field] = observation[field] = 1
    (proposal if invalid_record == "proposal" else observation)[field] = invalid

    with pytest.raises(ValueError):
        normalize_positive(proposal, observation, 7)


@pytest.mark.parametrize("style", ["head", "tail"])
@pytest.mark.parametrize("invalid", [True, 1.0])
@pytest.mark.parametrize("invalid_record", ["function_argument", "observation"])
def test_normalization_review_order_requires_integer_not_boolean_equality(style, invalid, invalid_record):
    proposal, observation = _review_pair(style)
    key = "selection_index" if style == "head" else "manifest_positive_index"
    observation[key] = invalid if invalid_record == "observation" else 1
    index = invalid if invalid_record == "function_argument" else 1

    with pytest.raises(ValueError):
        normalize_positive(proposal, observation, index)


def test_negative_integer_source_pts_remains_valid_and_is_not_rebased_to_zero():
    proposal, observation = _review_pair("tail")
    proposal["source_pts"] = observation["source_pts"] = -512
    proposal["timestamp_fraction_seconds"] = "-1/25"

    normalized = normalize_positive(proposal, observation, 7)

    assert normalized["source_pts"] == -512
    assert normalized["source_time_base"] == "1/12800"
    assert normalized["timestamp_fraction_seconds"] == "-1/25"


def test_rejected_low_view_extent_is_preserved_without_becoming_an_accepted_box():
    proposal, observation = _review_pair("head")
    observation["decision"] = "REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED"
    observation["primary_camera_scope"] = False
    observation["accept_visible_bbox_xyxy_native_px"] = None

    normalized = normalize_positive(proposal, observation, 7)

    assert normalized["decision"] == "REJECT_UNSUPPORTED_VIEW_VISIBLE_EXTENT_RECORDED"
    assert normalized["bbox_xyxy_native_px"] is None
    assert normalized["observed_visible_extent_xyxy_native_px"] == [1160, 375, 1182, 383]
    assert normalized["observed_event_family"] == "dubai"
    assert normalized["primary_camera_scope"] is False
    assert normalized["supervised_export_approved"] is False


def test_ambiguous_rejection_with_no_observed_extent_is_not_a_negative_annotation():
    proposal, observation = _review_pair("tail")
    observation.update({
        "decision": "REJECT_AMBIGUOUS",
        "bbox_xyxy_native_px": None,
        "visible_center_xy_native_px": None,
        "manual_bbox_xyxy_patch_px": None,
    })

    normalized = normalize_positive(proposal, observation, 7)

    assert normalized["decision"] == "REJECT_AMBIGUOUS"
    assert normalized["bbox_xyxy_native_px"] is None
    assert normalized["observed_visible_extent_xyxy_native_px"] is None
    assert normalized["supervised_export_approved"] is False
    assert normalized["complete_positive_crop_reviewed"] is False


def test_module_import_does_not_load_any_image_or_model_runtime():
    program = (
        "import sys,json; import scripts.data.prepare_yolo26_reviewed_ball_wave; "
        "print(json.dumps([name for name in ('cv2','numpy','PIL','torch','ultralytics') "
        "if name in sys.modules]))"
    )
    output = subprocess.check_output([sys.executable, "-c", program], text=True)
    assert json.loads(output) == []
