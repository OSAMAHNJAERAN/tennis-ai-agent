import os
import cv2
import json

os.makedirs("artifacts/validation/phase6_4_diagnostic_review", exist_ok=True)
with open("data/benchmarks/cross_match_final_holdout/videos.json") as vf:
    vids = json.load(vf)["videos"]
with open("data/benchmarks/cross_match_final_holdout/ground_truth_events.json") as ef:
    evs = json.load(ef)["events"]

review_data = {}
for vid_id, elist in evs.items():
    vpath = vids[vid_id]["path"]
    cap = cv2.VideoCapture(vpath)
    hits = [e for e in elist if e["event_type"] in ("SERVE_CONTACT", "PLAYER_HIT")]
    review_data[vid_id] = []
    for h in hits:
        f_idx = h["frame_best"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            img_name = f"{vid_id}_hit_f{f_idx}_{h['event_type']}_P{h['player_id']}.jpg"
            img_path = os.path.join("artifacts/validation/phase6_4_diagnostic_review", img_name)
            cv2.imwrite(img_path, frame)
            review_data[vid_id].append({
                "event_id": h["event_id"],
                "event_type": h["event_type"],
                "frame_best": f_idx,
                "image": img_name,
                "status": "MANUAL_FROM_RAW_VIDEO"
            })
    cap.release()

with open("artifacts/validation/phase6_4_diagnostic_review/manifest.json", "w") as mf:
    json.dump(review_data, mf, indent=2)
print(f"Diagnostic review pack generated with {sum(len(v) for v in review_data.values())} hits.")
