"""Inspect saved frozen line proposals against approximate source boundaries; no new fit."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "outputs/vision_upgrade_audit/dimitrov_court_image01"
report = json.loads((DEST / "report.json").read_text(encoding="utf-8"))
review = json.loads((DEST / "review.json").read_text(encoding="utf-8"))
points = review["corrected_approximate_coordinates_native_px"]
records = []
for identity, (first_index, last_index) in enumerate([(0, 1), (2, 3), (0, 2), (1, 3)]):
    start, end = np.asarray([points[first_index], points[last_index]], dtype=float) / 2
    vector = end - start
    length = np.linalg.norm(vector)
    direction = vector / length
    candidates = []
    for index, proposal in enumerate(report["line_proposals"]):
        first, last = np.asarray(proposal["segment"]).reshape(2, 2)
        delta = last - first
        size = np.linalg.norm(delta)
        if size < 30:
            continue
        tangent = delta / size
        angle = np.degrees(np.arccos(np.clip(abs(tangent @ direction), 0, 1)))
        along = sorted([(first - start) @ direction, (last - start) @ direction])
        overlap = max(0., min(length, along[1]) - max(0., along[0]))
        normal = np.array([-tangent[1], tangent[0]])
        offsets = np.abs((np.stack((start, end)) - first) @ normal)
        if angle <= 6 and overlap >= min(length, size) * .5 and overlap >= 30 and offsets.max() <= 20:
            candidates.append({"proposal_index": index, "angle_error_degrees": float(angle), "maximum_endpoint_to_segment_line_offset_reference_px": float(offsets.max()), "geometric_overlap_reference_px": float(overlap), **proposal})
    records.append({"canonical_boundary_line": identity, "geometrically_compatible_saved_proposals": len(candidates), "retained_with_frozen_support_half": sum(row["support_fraction"] >= .5 for row in candidates), "proposals": candidates})
output = {"model_inference_executed": False, "candidate_fits_added": 0, "runtime_changed": False, "qualification_evidence": False, "scope": "Approximate source-only outer lines compared to existing frozen proposal geometry; manual uncertainty and fragment sampling limit interpretation", "boundaries": records}
(DEST / "boundary_evidence.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
for row in records:
    print(json.dumps({"line": row["canonical_boundary_line"], "compatible": row["geometrically_compatible_saved_proposals"], "retained": row["retained_with_frozen_support_half"], "maximum_support": max((proposal["support_fraction"] for proposal in row["proposals"]), default=None), "fractions": [{key: proposal[key] for key in ("support_fraction", "bright_fraction", "white_fraction", "similar_sides_fraction", "length")} for proposal in row["proposals"]]}, indent=2))
