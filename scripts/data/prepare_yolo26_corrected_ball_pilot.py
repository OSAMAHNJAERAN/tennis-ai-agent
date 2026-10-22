"""Preserve v1 and create the five-image, agent-reviewed ball learning check."""
from copy import deepcopy
import json

from scripts.data.prepare_yolo26_data import prepare
from src.detection.yolo26_bundle import REPO_ROOT, sha256_file


def main():
    original = REPO_ROOT / "data/yolo26/pilot_annotations_v1.json"
    annotation_path = REPO_ROOT / "data/yolo26/pilot_annotations_v2.json"
    output = REPO_ROOT / "data/yolo26/pilot_v2"
    if annotation_path.exists() or output.exists():
        raise FileExistsError("Corrected pilot exists; inspect its receipt instead of rerunning")
    document = deepcopy(json.loads(original.read_text(encoding="utf-8")))
    document["purpose"] = "FIVE_AGENT_REVIEWED_CORRECTED_BALL_IMAGES_DEVELOPMENT_ONLY"
    document["parent_annotation_sha256"] = sha256_file(original)
    document["changes"] = [
        "match172 frame13: box expanded to [865,279,877,285] to contain visible blur previously clipped at top/right",
        "match172 frame18: excluded because apparent evidence overlaps a bright line and remains ambiguous",
        "All reviewer flags explicitly record agent visual review; no independent human review claim",
    ]
    document["visual_evidence"] = "outputs/yolo26_ball_pilot_diagnostic_v1/label_pixel_review.png"
    document["annotations"] = [a for a in document["annotations"]
                               if not (a["source_id"] == "match172" and a["frame_index"] == 18)]
    for annotation in document["annotations"]:
        annotation.update(human_reviewed=False, manual_reviewed=True,
                          reviewer_type="AGENT_VISUAL_REVIEW", independent_human_reviewed=False,
                          annotation_provenance="AGENT_VISUAL_MANUAL")
        if annotation["source_id"] == "match172" and annotation["frame_index"] == 13:
            annotation["instances"][0]["bbox_xyxy_px"] = [865, 279, 877, 285]
            annotation["instances"][0]["blur_severity"] = "visible_motion_blur"
            annotation["review_method"] = "Second agent pixel review of native 20x nearest-neighbor crop; expanded top/right edges to contain observed blur"
        annotation["reviewed_on"] = "2026-10-09"
    annotation_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    manifest = prepare(document, output, pilot=True)
    print(json.dumps({"manifest": str(manifest), "sha256": sha256_file(manifest)}))


if __name__ == "__main__":
    main()
