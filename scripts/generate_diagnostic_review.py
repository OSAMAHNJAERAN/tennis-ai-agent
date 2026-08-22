"""Generate prediction-free t-3..t+3 diagnostic GT contact sheets."""

import json
import os

import cv2


OUTPUT_DIR = "artifacts/validation/phase6_4_diagnostic_review"
OFFSETS = tuple(range(-3, 4))


def _read_frame(cap, frame_index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    return frame if ok else None


def _panel(frame, frame_index, fps, target_height=240):
    scale = target_height / frame.shape[0]
    panel = cv2.resize(frame, (round(frame.shape[1] * scale), target_height))
    label = f"RAW frame={frame_index}  t={frame_index / fps:.3f}s"
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return panel


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open("data/benchmarks/cross_match_final_holdout/videos.json", encoding="utf-8") as stream:
        videos = json.load(stream)["videos"]
    with open("data/benchmarks/cross_match_final_holdout/ground_truth_events.json", encoding="utf-8") as stream:
        events = json.load(stream)["events"]

    manifest = {
        "schema_version": "1.0",
        "scientific_split": "CROSS_MATCH_DIAGNOSTIC",
        "annotation_source": "MANUAL_FROM_RAW_VIDEO",
        "prediction_overlay": False,
        "videos": {},
    }
    for video_id, video_events in events.items():
        metadata = videos[video_id]
        fps = float(metadata["fps"])
        cap = cv2.VideoCapture(metadata["path"])
        manifest["videos"][video_id] = []
        for event in video_events:
            if event["event_type"] not in ("SERVE_CONTACT", "PLAYER_HIT"):
                continue
            center = int(event["frame_best"])
            panels = []
            reviewed_frames = []
            for offset in OFFSETS:
                frame_index = center + offset
                if frame_index < 0:
                    continue
                frame = _read_frame(cap, frame_index)
                if frame is None:
                    continue
                panels.append(_panel(frame, frame_index, fps))
                reviewed_frames.append(frame_index)
            if len(panels) != len(OFFSETS):
                raise RuntimeError(f"Incomplete review window for {video_id} frame {center}")
            sheet = cv2.hconcat(panels)
            filename = (
                f"{video_id}_event_{event['event_id']:02d}_f{center}_"
                f"{event['event_type']}_P{event['player_id']}_contact_sheet.jpg"
            )
            path = os.path.join(OUTPUT_DIR, filename)
            if not cv2.imwrite(path, sheet):
                raise RuntimeError(f"Failed to write {path}")
            manifest["videos"][video_id].append({
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "frame_best": center,
                "reviewed_frames": reviewed_frames,
                "review_asset": filename,
                "status": "MANUAL_FROM_RAW_VIDEO",
            })
        cap.release()

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    count = sum(len(items) for items in manifest["videos"].values())
    print(f"Generated {count} prediction-free seven-frame contact sheets.")


if __name__ == "__main__":
    main()
