"""Audit the frozen single court candidate against exact saved source pixels; no new fit."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/vision_upgrade_audit/dimitrov_court_image01"
report_path = DEST / "report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
# Original marks were reviewed on a tool frame with rounded time labels; the
# exact saved decoded frame zero differs because the camera pans. Preserve the
# original report and mark its coordinate comparison invalid, rather than tuning
# the surface candidate or quietly replacing prior marks/results.
corrected = [[60, 387], [667, 266], [1058, 937], [1754, 696], [145, 370], [1148, 904], [600, 280], [1673, 723], None, [787, 358], [858, 743], [1356, 594], [575, 406], [1090, 670]]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def compare(points):
    distances = [None if point is None else float(np.linalg.norm(np.asarray(points[index]) - point)) for index, point in enumerate(corrected)]
    visible = [distance for distance in distances if distance is not None]
    return {"native_distances_to_approximate_source_marks_px": distances, "mean_native_distance_px": float(np.mean(visible)), "median_native_distance_px": float(np.median(visible)), "maximum_native_distance_px": max(visible), "visible_points": len(visible), "distances_above_20_px": sum(distance > 20 for distance in visible), "coordinate_review_uncertainty_px": 8, "independently_adjudicated_accuracy": False}

for name, expected in report["image_hashes"].items():
    if digest(DEST / name) != expected:
        raise ValueError("Frozen diagnostic image changed")
candidate = report["candidate"]
if report["candidate_count"] != 1 or candidate["image_evidence_gate_pass"]:
    raise ValueError("Expected one rejected frozen hypothesis")
np.testing.assert_array_equal(candidate["seed_landmarks_native_px"], candidate["refined_landmarks_native_px"])
image = cv2.imread(str(DEST / "01_original_frame0.png"))
if image is None or image.shape != (1080, 1920, 3):
    raise ValueError("Exact original source frame unavailable")
preview = cv2.resize(image, (960, 540))
for identity, point in enumerate(corrected):
    if point is not None:
        location = tuple(np.rint(np.asarray(point) / 2).astype(int))
        cv2.circle(preview, location, 4, (255, 255, 0), -1)
        cv2.putText(preview, str(identity), (location[0] + 5, location[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1, cv2.LINE_AA)
corrected_path = DEST / "08_corrected_exact_frame0_review_marks.png"
if corrected_path.exists():
    raise FileExistsError(corrected_path)
cv2.imwrite(str(corrected_path), preview)
raw = np.asarray(report["saved_raw_calibration"]["court_keypoints_px"])
raw_baseline_vector = raw[1] - raw[0]
source_baseline_vector = np.asarray(corrected[1]) - corrected[0]
review = {"report_sha256": digest(report_path), "review_script_sha256": digest(Path(__file__)), "original_coordinate_comparison_valid": False, "coordinate_correction_reason": "Native video tool frame labels rounded a different camera-framing instant to 00:00:00; source-only marks now refer to the exact saved decoded frame-zero PNG. This review correction happened after the sole candidate run and is not a blinded label assessment.", "exact_source_frame_png_sha256": digest(DEST / "01_original_frame0.png"), "corrected_approximate_coordinates_native_px": corrected, "coordinate_uncertainty_px": 8, "occluded_identity": 8, "qualified_ground_truth": False, "saved_raw_comparison": compare(raw), "automatic_seed_comparison": compare(candidate["seed_landmarks_native_px"]), "returned_seed_comparison": compare(candidate["refined_landmarks_native_px"]), "raw_far_baseline_angle_degrees": float(np.degrees(np.arctan2(raw_baseline_vector[1], raw_baseline_vector[0]))), "source_far_baseline_angle_degrees": float(np.degrees(np.arctan2(source_baseline_vector[1], source_baseline_vector[0]))), "line_support": {"raw_mean": report["saved_raw_line_support"]["mean_visible_line_support"], "seed_mean": candidate["seed_support"]["mean_visible_line_support"], "seed_individually_supported_lines": sum(row["supported_fraction"] >= .5 for row in candidate["seed_support"]["lines"]), "seed_outer_boundaries_support": [row["supported_fraction"] for row in candidate["seed_support"]["lines"][:4]], "candidate_gate_pass": False, "refinement_reason": candidate["refinement"]["reason"]}, "sole_candidate_unchanged_after_review": True, "runtime_changed": False, "model_inference_executed": False, "qualification_evidence": False, "corrected_review_image_sha256": digest(corrected_path)}
(DEST / "review.json").write_text(json.dumps(review, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in review.items() if key in ("saved_raw_comparison", "automatic_seed_comparison", "raw_far_baseline_angle_degrees", "source_far_baseline_angle_degrees", "line_support", "sole_candidate_unchanged_after_review")}, indent=2))
