"""Synthetic counterexamples for the frozen conditional surface-boundary cue."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.evaluate.probe_court_line_segments import line_evidence as original_evidence
from scripts.evaluate.probe_court_line_segments import propose as original_propose


spec = importlib.util.spec_from_file_location("frozen_surface_boundary_candidate", Path(__file__).with_name("candidate.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

SEGMENT = [20, 50, 240, 50]
ORIGINAL_FIELDS = {"support_fraction", "bright_fraction", "white_fraction", "similar_sides_fraction",
                   "length", "center_shift", "segment"}


def paint(left=30, right=100, ridge=220):
    image = np.empty((100, 260, 3), np.uint8)
    image[:50] = left
    image[50:] = right
    image[49:52] = ridge
    return image


def upper_surface():
    mask = np.zeros((100, 260), np.uint8)
    mask[:50] = 1
    return mask


def retained(row):
    return row is not None and row["support_fraction"] >= .5


@pytest.mark.parametrize("mask", [None, np.zeros((100, 260), np.uint8), np.ones((100, 260), np.uint8), upper_surface()])
def test_original_passing_record_is_exact_even_when_mask_disagrees(mask):
    image = paint(30, 40)
    expected = original_evidence(image, SEGMENT)
    assert expected["support_fraction"] == 1
    actual = candidate.line_evidence(image, SEGMENT, mask)
    assert {key: actual[key] for key in ORIGINAL_FIELDS} == expected
    assert actual["baseline_record"] == expected
    assert actual["evidence_origin"] == "BASELINE"


def test_white_paint_between_different_surfaces_is_recovered_with_straddle():
    image = paint()
    expected = original_evidence(image, SEGMENT)
    assert expected["support_fraction"] == 0
    actual = candidate.line_evidence(image, SEGMENT, upper_surface())
    assert retained(actual)
    assert actual["support_fraction"] == 1
    assert actual["bright_fraction"] == actual["white_fraction"] == 1
    assert actual["similar_sides_fraction"] == 0
    assert actual["boundary_pattern_fraction"] == 1
    assert actual["boundary_replacement_sample_count"] == 80
    assert actual["boundary_inside_side"] == "NEGATIVE_NORMAL"
    assert actual["baseline_record"] == expected


def test_endpoint_reversal_swaps_normal_side_without_losing_evidence():
    image, mask = paint(), upper_surface()
    forward = candidate.line_evidence(image, SEGMENT, mask)
    backward = candidate.line_evidence(image, SEGMENT[2:] + SEGMENT[:2], mask)
    assert forward["support_fraction"] == backward["support_fraction"] == 1
    assert forward["boundary_inside_side"] == "NEGATIVE_NORMAL"
    assert backward["boundary_inside_side"] == "POSITIVE_NORMAL"


@pytest.mark.parametrize("kind", ["missing", "outside_advertisement", "inside_net", "alternating_inside_side", "one_side_pair_disagrees"])
def test_boundary_recovery_rejects_surface_ambiguity_and_distractors(kind):
    mask = upper_surface()
    if kind == "missing":
        mask = None
    elif kind == "outside_advertisement":
        mask[:] = 0
    elif kind == "inside_net":
        mask[:] = 1
    elif kind == "alternating_inside_side":
        # Every location straddles a region, but no common inside orientation.
        mask[:, 130:] = 1-mask[:, 130:]
    elif kind == "one_side_pair_disagrees":
        # Offsets separated by two pixels always disagree in this striped mask.
        mask[:] = (np.arange(100)[:, None] % 4) < 2
    assert not retained(candidate.line_evidence(paint(), SEGMENT, mask))


def test_surface_step_without_bilateral_paint_contrast_is_rejected():
    assert not retained(candidate.line_evidence(paint(30, 180, 180), SEGMENT, upper_surface()))


def test_colored_ridge_is_not_rescued_by_the_surface_condition():
    # Every sampled center and neighbor is saturated, so the existing
    # central-minimum-saturation condition is genuinely false at every shift.
    image = paint()
    image[:50] = (255, 0, 0)
    image[50:] = (0, 255, 0)
    image[49:52] = (0, 255, 255)
    actual = candidate.line_evidence(image, SEGMENT, upper_surface())
    assert actual["white_fraction"] == 0
    assert not retained(actual)


def test_thin_colored_ridge_on_gray_background_exposes_inherited_whiteness_limit():
    # This failed the first rejection fixture. It documents an inherited
    # minimum-saturation limitation, rather than changing candidate thresholds.
    image = paint()
    image[49:52] = (0, 255, 255)
    actual = candidate.line_evidence(image, SEGMENT, upper_surface())
    assert actual["evidence_origin"] == "SURFACE_BOUNDARY_RECOVERY"
    assert actual["white_fraction"] == 1
    assert retained(actual)


def test_mask_boundary_alone_never_creates_image_line_evidence():
    assert not retained(candidate.line_evidence(np.full((100, 260, 3), 120, np.uint8), SEGMENT, upper_surface()))


@pytest.mark.parametrize("valid_boundary_samples,accepted", [(63, False), (64, True), (65, True)])
def test_consistent_boundary_requires_inclusive_eighty_percent(valid_boundary_samples, accepted):
    # The protocol fixes 80 interior locations. Spatial holes in the actual mask
    # remove selected locations, rather than mocking photometry or evidence.
    mask = upper_surface()
    xs = np.rint(np.linspace(31, 229, 80)).astype(int)
    mask[:, xs[valid_boundary_samples:]] = 0
    actual = candidate.line_evidence(paint(), SEGMENT, mask)
    assert retained(actual) is accepted
    if accepted:
        assert actual["boundary_pattern_fraction"] == valid_boundary_samples / 80
        assert actual["boundary_replacement_sample_count"] == valid_boundary_samples


@pytest.mark.parametrize("mask", [np.ones((100, 259), np.uint8), np.ones((100, 260, 1), np.uint8),
                                  np.full((100, 260), np.nan), np.full((100, 260), 2, np.uint8)])
def test_malformed_mask_is_rejected_before_baseline_short_circuit(mask):
    with pytest.raises(ValueError, match="mask"):
        candidate.line_evidence(paint(30, 40), SEGMENT, mask)


@pytest.mark.parametrize("segment", [[20, 50, np.nan, 50], [20, 50, 240, np.inf], [20, 50, 240]])
def test_invalid_segment_cannot_produce_evidence(segment):
    with pytest.raises(ValueError, match="finite"):
        candidate.line_evidence(paint(), segment, upper_surface())


def test_short_degenerate_and_outside_segments_are_unavailable():
    for segment in ([20, 50, 40, 50], [20, 50, 20, 50], [20, -20, 240, -20]):
        assert candidate.line_evidence(paint(), segment, upper_surface()) is None


@pytest.mark.parametrize("image", [np.zeros((100, 260)), np.zeros((100, 260, 3)),
                                   np.zeros((1, 260, 3), np.uint8), np.zeros((100, 260, 4), np.uint8)])
def test_invalid_image_is_rejected(image):
    with pytest.raises(ValueError, match="BGR"):
        candidate.propose(image)


def test_proposal_baseline_parity_uses_actual_lsd_on_synthetic_paint():
    image = paint(30, 40)
    expected = [row for row in original_propose(image) if row["support_fraction"] >= .5]
    assert expected
    actual = candidate.propose(image, np.ones(image.shape[:2], np.uint8))
    assert [{key: row[key] for key in ORIGINAL_FIELDS} for row in actual] == expected
    assert all(row["evidence_origin"] == "BASELINE" and row["baseline_record"] == reference
               for row, reference in zip(actual, expected, strict=True))


def test_no_mask_cannot_add_previously_rejected_actual_lsd_records():
    image = paint()
    expected = [row for row in original_propose(image) if row["support_fraction"] >= .5]
    actual = candidate.propose(image)
    assert [{key: row[key] for key in ORIGINAL_FIELDS} for row in actual] == expected


def test_proposal_order_is_stable_for_equal_scores(monkeypatch):
    class Detector:
        def detect(self, gray):
            return np.array([[[20, 50, 120, 50]], [[130, 50, 230, 50]]], np.float32), None, None, None
    monkeypatch.setattr(cv2, "createLineSegmentDetector", lambda mode: Detector())
    actual = candidate.propose(paint(), upper_surface())
    assert len(actual) == 2
    assert [row["lsd_index"] for row in actual] == [0, 1]
    assert all(row["evidence_origin"] == "SURFACE_BOUNDARY_RECOVERY" for row in actual)


def test_surface_selection_preserves_component_mask_and_skips_nonquad(monkeypatch):
    bad = {"hull": [[0, 0], [80, 0], [40, 80]], "mask": np.zeros((100, 260), np.uint8)}
    mask = upper_surface()
    good = {"hull": [[20, 10], [240, 10], [240, 90], [20, 90]], "mask": mask}
    later = {"hull": [[30, 20], [230, 20], [230, 80], [30, 80]], "mask": 1-mask}
    monkeypatch.setattr(candidate, "surface_candidates", lambda image: [bad, good, later])
    selected = candidate.select_surface(paint())
    component, quad = selected["component"], selected["quadrilateral"]
    assert component is good and component["mask"] is mask
    assert quad.shape == (4, 2) and np.isfinite(quad).all()
    assert set(map(tuple, quad)) == set(map(tuple, good["hull"]))


def test_surface_selection_has_no_eligible_component_fallback(monkeypatch):
    monkeypatch.setattr(candidate, "surface_candidates", lambda image: [])
    assert candidate.select_surface(paint()) is None


def test_functions_do_not_mutate_source_image_or_mask():
    image, mask = paint(), upper_surface()
    before_image, before_mask = image.copy(), mask.copy()
    candidate.line_evidence(image, SEGMENT, mask)
    candidate.propose(image, mask)
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(mask, before_mask)
