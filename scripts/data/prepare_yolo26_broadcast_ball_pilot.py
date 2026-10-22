"""Convert explicitly reviewed publisher centers into a grouped ball pilot.

The rectangles below were selected from the native-pixel crop sheets by agent
visual review. They are box annotations by the agent, not publisher boxes or
independent human truth. Ambiguous crops are excluded rather than negatives.
Run once to render review sheets, inspect them, then use --convert.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.data.prepare_yolo26_data import prepare
from src.detection.yolo26_bundle import REPO_ROOT, sha256_file

cv2.setNumThreads(1)
REVIEW = REPO_ROOT / "data/yolo26/broadcast_ball_review_v1"
DOCUMENT = REPO_ROOT / "data/yolo26/broadcast_ball_annotations_v1.json"
OUTPUT = REPO_ROOT / "data/yolo26/pilot_broadcast_v1"

# x1,y1,x2,y2 relative to the explicitly recorded 33x33 native crop origin.
# None means evidence was ambiguous/occluded/background-like: exclude it.
RECTANGLES = {
    "match173": [None, [13,10,22,24], [13,13,20,20], [14,12,22,21], None,
                 [13,14,21,21], [13,13,20,20], [12,10,19,23], [14,15,22,23],
                 [13,16,19,22], [14,8,20,27], [11,15,20,22]],
    "match176": [[11,11,21,21], [12,15,19,22], [10,10,23,25], [12,12,21,20], None,
                 [9,11,25,26], [12,12,21,20], [12,12,20,20], [13,14,19,21],
                 [12,12,22,20], [13,17,20,23], [13,11,21,22]],
    "match179": [None, [11,9,29,24], [12,12,22,22], [13,12,21,18], [11,8,19,26],
                 [13,14,23,21], [13,14,21,20], [12,10,23,20], [10,10,19,26],
                 [11,11,24,24], None, [10,9,21,20]],
    "match180": [[14,12,18,20], None, [12,12,21,20], [12,12,19,19], [13,13,20,20],
                 [13,10,21,21], [13,12,20,22], [13,13,20,20], [13,12,20,22],
                 [10,11,21,22], [12,12,20,19], [13,14,20,23]],
    "match182": [None, [12,13,20,19], [13,13,21,18], [13,12,20,19], [12,11,20,22],
                 [11,11,22,20], [11,11,22,20], [12,13,19,20], [12,12,20,20],
                 [13,10,18,24], [13,13,20,20], None],
    "match183": [[12,12,23,20], [12,12,21,21], [14,14,18,19], [12,9,21,27],
                 [12,10,22,22], [15,14,18,17], [15,14,18,20], [13,13,20,20],
                 None, [13,13,21,20], None, [12,14,20,18]],
    "match185": [[12,13,21,20], [13,13,19,20], None, [11,11,22,22], [14,11,19,22],
                 None, [10,12,22,29], [13,13,20,20], [13,12,20,22], [12,12,21,22],
                 [13,13,20,20], None],
    "match188": [[9,14,27,18], [12,13,18,20], [8,10,23,27], [10,9,24,22],
                 [13,14,20,20], [9,8,26,29], None, None, [12,9,20,25],
                 [10,12,24,22], [15,15,19,21], [10,12,22,31]],
}
CONTEXTS = {
    "match173": ("shanghai", "sinner_medvedev", "hard", "outdoor", "train"),
    "match176": ("barcelona", "de_minaur_nadal", "clay", "outdoor", "train"),
    "match179": ("queens_cinch", "murray_popyrin", "grass", "outdoor", "train"),
    "match180": ("indian_wells", "murray_goffin", "hard", "outdoor", "train"),
    "match182": ("st_petersburg", "norrie_kecmanovic", "hard", "indoor", "val"),
    "match183": ("barcelona", "medvedev_arnaldi", "clay", "outdoor", "train"),
    "match185": ("barcelona", "dimitrov_rune", "clay", "outdoor", "train"),
    "match188": ("torino_atp_finals", "medvedev_ruud", "hard", "indoor", "train"),
}


def reviewed_document():
    inventory_path = REVIEW / "review_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    document = {"schema_version": "1.0", "purpose": "GROUPED_DEVELOPMENT_BROADCAST_BALL_LEARNING_PILOT",
                "source_independence_verified": False, "independent_test_reviewed": False,
                "review_inventory_sha256": sha256_file(inventory_path),
                "box_annotation_source": "AGENT_MANUAL_NATIVE_PIXEL_CROP_REVIEW_AROUND_PUBLISHER_HUMAN_CENTERS",
                "reviewed_on": "2026-10-09", "sources": [], "annotations": [], "excluded": [],
                "split_policy": "All Barcelona pairings share one conservative session group; St Petersburg is development validation only, never final test",
                "final_test_used": False, "qualification_evidence": False}
    for original in inventory["sources"]:
        source = deepcopy(original)
        event, pair, surface, setting, split = CONTEXTS[source["source_id"]]
        source.update(match_id=f"{event}_{pair}_year_unverified", session_id=f"{event}_season_unverified_conservative_group",
                      original_source_video_id=f"publisher_{source['source_id']}_{source['publisher_rally_id']}",
                      event_identity_from_visible_court=event, player_pair_from_visible_scoreboard=pair,
                      original_date_and_round="UNVERIFIED", grouping_status="CONSERVATIVE_VISIBLE_EVENT_AND_PAIR_NOT_FULL_INDEPENDENCE",
                      owner="Linfeng Dong et al. / linfeng302, publisher; original broadcaster not separately verified",
                      rights_evidence="https://huggingface.co/datasets/linfeng302/RacketVision (MIT dataset card and explicit training instructions, checked 2026-10-09)",
                      training_allowed=True, use_scope="PRIVATE_LOCAL_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT_TRAINING",
                      public_redistribution_allowed=False, underlying_broadcast_permission="NOT_SEPARATELY_VERIFIED_NO_PUBLIC_RAW_DERIVATIVES",
                      surface=surface, lighting_setting=setting, view="elevated_end_court_singles", camera_reviewed=True,
                      reviewer_type="AGENT_VISUAL_REVIEW", acquired_on="HISTORIC_PINNED_LOCAL_ACQUISITION_MANIFEST_PRESERVED",
                      reviewed_on="2026-10-09", development_split=split,
                      camera_evidence="data/yolo26/broadcast_ball_review_v1/scope_00.jpg and native selected-frame review sheets")
        document["sources"].append(source)
    thumbnails = []
    for source_id, rectangles in RECTANGLES.items():
        candidates = [c for c in inventory["candidates"] if c["source_id"] == source_id]
        if len(candidates) != len(rectangles):
            raise ValueError("Candidate order/count changed; re-review native crops")
        for candidate, rectangle in zip(candidates, rectangles):
            image_path = REPO_ROOT / candidate["image"]
            if sha256_file(image_path) != candidate["image_sha256"]:
                raise ValueError("Native reviewed image changed")
            image = cv2.imread(str(image_path))
            text = f"{source_id} f{candidate['frame_index']}"
            if rectangle is None:
                document["excluded"].append({"source_id": source_id, "frame_index": candidate["frame_index"],
                                             "reason": "AMBIGUOUS_BACKGROUND_LINE_NET_OR_PLAYER_OVERLAP_NO_VISIBLE_BOX_ACCEPTED"})
                text += " EXCLUDED"
            else:
                x, y = candidate["crop_origin_xy_px"]
                left, top, right, bottom = rectangle
                box = [x + left, y + top, x + right, y + bottom]
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
                document["annotations"].append({
                    "role": "ball", **candidate, "human_reviewed": False, "manual_reviewed": True,
                    "reviewer_type": "AGENT_VISUAL_REVIEW", "independent_human_reviewed": False,
                    "annotation_provenance": "AGENT_VISUAL_MANUAL", "visibility": "visible",
                    "blur_severity": "MANUALLY_INSPECTED_VISIBLE_BLUR_RETAINED_IN_BOX_WHERE_PRESENT",
                    "review_method": "Publisher human center; agent selected tight visible/blurred native rectangle from 8x nearest-neighbor crop; ambiguous candidates excluded; full-frame view checked separately",
                    "instances": [{"bbox_xyxy_px": box, "visibility": "visible"}],
                })
            thumb = cv2.resize(image, (320, 180))
            caption = np.full((24, 320, 3), 245, np.uint8)
            cv2.putText(caption, text, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (15, 15, 15), 1)
            thumbnails.append(np.vstack([thumb, caption]))
    for start in range(0, len(thumbnails), 24):
        page = thumbnails[start:start + 24]
        while len(page) % 4:
            page.append(np.full_like(page[0], 245))
        canvas = np.vstack([np.hstack(page[i:i + 4]) for i in range(0, len(page), 4)])
        cv2.imwrite(str(REVIEW / f"full_frame_boxes_{start // 24:02d}.jpg"), canvas)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convert", action="store_true", help="After agent visual inspection of the rendered native-frame sheets")
    args = parser.parse_args()
    if DOCUMENT.exists() or OUTPUT.exists():
        raise FileExistsError("Prepared corpus exists; preserve it and inspect receipts")
    document = reviewed_document()
    if args.convert:
        document["full_frame_camera_reviewed"] = True
        DOCUMENT.write_text(json.dumps(document, indent=2), encoding="utf-8")
        path = prepare(document, OUTPUT, pilot=True)
        print(json.dumps({"manifest": str(path), "sha256": sha256_file(path)}))
    else:
        print(json.dumps({"reviewed_positive_boxes": len(document["annotations"]),
                          "excluded_ambiguous": len(document["excluded"]), "status": "FULL_FRAME_CAMERA_REVIEW_PENDING"}))


if __name__ == "__main__":
    main()
