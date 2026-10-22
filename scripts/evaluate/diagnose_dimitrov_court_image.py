"""One frozen image-supported initialization diagnostic; no detector inference/runtime changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_caltennis_projection import LINES
from scripts.evaluate.audit_court_line_support import support
from scripts.evaluate.probe_court_line_segments import propose
from scripts.evaluate.probe_court_surface_support import surface_candidates
from scripts.evaluate.refine_court_boundary_geometry import refine
from src.court.court_geometry import TennisCourtGeometry

MANUAL = [[17, 378], [656, 251], [1064, 955], [1797, 703], [108, 360], [1160, 921], [587, 265], [1707, 733], None, [779, 349], [852, 753], [1378, 592], [553, 405], [1100, 677]]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()

def comparison(points: np.ndarray) -> dict:
    rows = [{"identity": index, "visible": point is not None, "native_error_px": None if point is None else float(np.linalg.norm(points[index] - point)), "review_coordinate_uncertainty_px": 8} for index, point in enumerate(MANUAL)]
    errors = [row["native_error_px"] for row in rows if row["visible"]]
    return {"points": rows, "mean_visible_native_error_px": float(np.mean(errors)), "median_visible_native_error_px": float(np.median(errors)), "maximum_visible_native_error_px": max(errors), "scope": "SINGLE_AI_SOURCE_REVIEW_APPROXIMATE_COORDINATES; NOT_ADJUDICATED_ACCURACY_LABELS"}

def draw_points(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    preview = image.copy()
    for identity, point in enumerate(points):
        location = tuple(np.rint(point / 2).astype(int))
        cv2.circle(preview, location, 4, color, -1)
        cv2.putText(preview, str(identity), (location[0] + 5, location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    for first, last in LINES:
        cv2.line(preview, tuple(np.rint(points[first] / 2).astype(int)), tuple(np.rint(points[last] / 2).astype(int)), color, 1, cv2.LINE_AA)
    return preview

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Preserve every prior diagnostic; choose a new output directory")
    cv2.setNumThreads(1)
    source = ROOT / "artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv"
    geometry_path = ROOT / "outputs/product/analyses/match_02d494b88e99400d8cf7243528302f9b/court_geometry.json"
    protocol_path = ROOT / "docs/experiments/DIMITROV_COURT_IMAGE_DIAGNOSIS_PROTOCOL_2026-10-01.md"
    saved = json.loads(geometry_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(source))
    ok, image = capture.read()
    capture.release()
    if not ok or image.shape != (1080, 1920, 3):
        raise ValueError("Require exact source frame-zero native geometry")
    reference = cv2.resize(image, (960, 540))
    arguments.output.mkdir(parents=True)
    cv2.imwrite(str(arguments.output / "01_original_frame0.png"), image)
    manual_preview = reference.copy()
    for index, point in enumerate(MANUAL):
        if point is not None:
            location = tuple(np.rint(np.asarray(point) / 2).astype(int))
            cv2.circle(manual_preview, location, 4, (255, 255, 0), -1)
            cv2.putText(manual_preview, str(index), (location[0] + 5, location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(arguments.output / "02_source_review_marks.png"), manual_preview)
    raw = np.asarray(saved["court_keypoints_px"], dtype=float)
    cv2.imwrite(str(arguments.output / "03_saved_raw_prediction.png"), draw_points(reference, raw, (0, 0, 255)))
    proposals = propose(reference)
    selected_lines = [row for row in proposals if row["support_fraction"] >= .5]
    segments = np.asarray([row["segment"] for row in selected_lines], dtype=float).reshape(-1, 2, 2)
    line_preview = reference.copy()
    for index, row in enumerate(selected_lines):
        endpoints = np.rint(np.asarray(row["segment"]).reshape(2, 2)).astype(int)
        cv2.line(line_preview, tuple(endpoints[0]), tuple(endpoints[1]), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(arguments.output / "04_frozen_line_proposals.png"), line_preview)
    components = surface_candidates(reference)
    component_rows = []
    selected_component = None
    quadrilateral = None
    for index, component in enumerate(components):
        hull = np.asarray(component["hull"], dtype=np.float32).reshape(-1, 1, 2)
        approximation = cv2.approxPolyDP(hull, .01 * cv2.arcLength(hull, True), True)
        component_rows.append({key: value for key, value in component.items() if key != "mask"} | {"index": index, "hull_approximation": approximation[:, 0].tolist()})
        if selected_component is None and len(approximation) == 4:
            selected_component = component
            quadrilateral = approximation[:, 0].astype(float)
            component_index = index
    report = {"complete": False, "qualification_evidence": False, "model_inference_executed": False, "runtime_changed": False, "source_video_sha256": digest(source), "saved_geometry_sha256": digest(geometry_path), "protocol_sha256": digest(protocol_path), "script_sha256": digest(Path(__file__)), "helper_hashes": {str(path.relative_to(ROOT)): digest(path) for path in [ROOT / "scripts/evaluate/probe_court_line_segments.py", ROOT / "scripts/evaluate/probe_court_surface_support.py", ROOT / "scripts/evaluate/refine_court_boundary_geometry.py", ROOT / "scripts/evaluate/audit_court_line_support.py"]}, "source_only_review_coordinates": MANUAL, "coordinate_uncertainty_px": 8, "occluded_identity": 8, "saved_raw_calibration": saved, "saved_raw_source_comparison": comparison(raw), "saved_raw_line_support": support(raw / 2, segments, 960, 540), "line_proposals": proposals, "selected_line_count": len(selected_lines), "surface_components": component_rows, "candidate_count": 0}
    if selected_component is None:
        report["candidate_rejection"] = "NO_FOUR_CORNER_COMPONENT_AT_FROZEN_EPSILON"
    else:
        center = quadrilateral.mean(axis=0)
        ordering = np.argsort(np.arctan2(quadrilateral[:, 1] - center[1], quadrilateral[:, 0] - center[0]))
        corners = quadrilateral[ordering]
        lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        pair_start = 0 if lengths[0] + lengths[2] <= lengths[1] + lengths[3] else 1
        baselines = [corners[[pair_start, (pair_start + 1) % 4]], corners[[(pair_start + 2) % 4, (pair_start + 3) % 4]]]
        baselines.sort(key=lambda edge: edge[:, 1].mean())
        far = baselines[0][np.argsort(baselines[0][:, 0])]
        near = baselines[1][np.argsort(baselines[1][:, 0])]
        ordered = np.vstack((far, near))
        canonical = TennisCourtGeometry.get_canonical_keypoints().astype(float)
        transform = cv2.getPerspectiveTransform(canonical[:4].astype(np.float32), ordered.astype(np.float32))
        seed = cv2.perspectiveTransform(canonical[None], transform)[0]
        np.testing.assert_allclose(seed[:4], ordered, atol=1e-4)
        if not np.isfinite(seed).all():
            raise ValueError("Nonfinite automatic hypothesis")
        surface_preview = reference.copy()
        mask = selected_component["mask"] > 0
        surface_preview[mask] = (surface_preview[mask] * .65 + np.array([40, 220, 80]) * .35).astype(np.uint8)
        cv2.polylines(surface_preview, [np.rint(corners).astype(int)], True, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(arguments.output / "05_automatic_surface_hull.png"), surface_preview)
        refined, refinement = refine(seed, segments, 960, 540)
        seed_support = support(seed, segments, 960, 540)
        refined_support = support(refined, segments, 960, 540)
        accepted_evidence = bool(refined_support["diagnostic_gate_pass"] and sum(row["supported_fraction"] >= .5 for row in refined_support["lines"]) >= 6)
        report["candidate_count"] = 1
        report["candidate"] = {"selected_component_index": component_index, "ordered_outer_corners_reference_px": ordered.tolist(), "court_to_reference_seed": transform.tolist(), "seed_landmarks_native_px": (seed * 2).tolist(), "refined_landmarks_native_px": (refined * 2).tolist(), "seed_support": seed_support, "refinement": refinement, "returned_support": refined_support, "seed_source_comparison": comparison(seed * 2), "returned_source_comparison": comparison(refined * 2), "image_evidence_gate_pass": accepted_evidence, "automatically_validated_physical_court": False, "production_promoted": False}
        cv2.imwrite(str(arguments.output / "06_automatic_seed.png"), draw_points(reference, seed * 2, (0, 165, 255)))
        cv2.imwrite(str(arguments.output / "07_returned_refinement.png"), draw_points(reference, refined * 2, (0, 255, 0)))
    report["complete"] = True
    report["image_hashes"] = {path.name: digest(path) for path in arguments.output.glob("*.png")}
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "candidate_count": report["candidate_count"], "saved_raw_mean_error_px": report["saved_raw_source_comparison"]["mean_visible_native_error_px"], "candidate_rejection": report.get("candidate_rejection"), "candidate": {key: report.get("candidate", {}).get(key) for key in ("selected_component_index", "image_evidence_gate_pass", "seed_source_comparison", "returned_source_comparison", "seed_support", "refinement")}}, indent=2))

if __name__ == "__main__":
    main()
