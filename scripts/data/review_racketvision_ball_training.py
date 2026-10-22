"""Extract publisher-positive native frames/PTS for a bounded visual review.

No box, model prediction, interpolated label or final-test sample is created.
All historic training clips remain development only because original broadcast
match/session mapping is unresolved. Native decoded images are preserved.
"""
import argparse
import csv
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parents[2]


def native_timeline(video):
    executable = os.getenv("TENNIS_FFPROBE") or shutil.which("ffprobe")
    if not executable:
        local = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Links/ffprobe.exe"
        executable = str(local) if local.is_file() else None
    if not executable:
        raise RuntimeError("Existing FFprobe required; never manufacture PTS from nominal FPS")
    value = json.loads(subprocess.check_output([
        executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=pts:stream=width,height,avg_frame_rate,time_base", "-of", "json", str(video)
    ], text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    return value["streams"][0], [int(frame["pts"]) for frame in value["frames"]]


def digest(path):
    import hashlib
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", help="Comma-separated publisher match IDs; otherwise scope overview only")
    parser.add_argument("--frames-per-clip", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new immutable review directory")
    if not 1 <= args.frames_per_clip <= 20:
        raise ValueError("Bound review to 1–20 explicit publisher-positive frames per clip")
    selected = set(args.selected.split(",")) if args.selected else None
    args.output.mkdir(parents=True)
    sources, candidates, thumbnails, crops = [], [], [], []
    for cache in ("racketvision_training20", "racketvision_training_additional20", "racketvision_training_third20"):
        directory = ROOT / "data/external" / cache
        acquisition = json.loads((directory / "manifest.json").read_text())
        if acquisition["split"] != "TRAINING_ONLY":
            raise ValueError("Never read final-test or validation media for training")
        entries = {item["path"]: item for item in acquisition["files"]}
        for match, rally in acquisition["selected_clips"]:
            if selected is not None and match not in selected:
                continue
            video_relative = f"tennis/videos/{match}_{rally}.mp4"
            csv_relative = f"tennis/all/{match}/csv/{rally}_ball.csv"
            video, label = directory / video_relative, directory / csv_relative
            if digest(video) != entries[video_relative]["sha256"] or digest(label) != entries[csv_relative]["sha256"]:
                raise ValueError("Historic acquisition checksum changed")
            rows = list(csv.DictReader(label.open(newline="")))
            positive = [row for row in rows if int(row["Visibility"]) == 1]
            indices = sorted(set(int(value) for value in np.linspace(0, len(positive) - 1, min(args.frames_per_clip, len(positive)))))
            chosen = {int(positive[i]["Frame"]): positive[i] for i in indices} if selected else {}
            stream, ticks = native_timeline(video)
            capture = cv2.VideoCapture(str(video))
            try:
                source = {"source_id": match, "publisher_rally_id": rally,
                          "source_path": str(video.relative_to(ROOT)), "source_url": entries[video_relative]["url"],
                          "sha256": entries[video_relative]["sha256"], "annotation_path": str(label.relative_to(ROOT)),
                          "annotation_url": entries[csv_relative]["url"], "annotation_sha256": entries[csv_relative]["sha256"],
                          "revision": acquisition["revision"], "dataset_card_sha256": acquisition["dataset_card_sha256"],
                          "fps": float(Fraction(stream["avg_frame_rate"])), "average_rate": stream["avg_frame_rate"],
                          "time_base": stream["time_base"], "resolution": [stream["width"], stream["height"]],
                          "publisher_visible_rows": len(positive), "publisher_invisible_rows": len(rows) - len(positive),
                          "view_reviewed": False, "match_independence_verified": False,
                          "qualification_evidence": False}
                if stream["width"] != 1920 or stream["height"] != 1080:
                    raise ValueError("Publisher points assume 1920x1080; explicit transform review required")
                count = 0
                for frame_index, tick in enumerate(ticks):
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError("Decoded frames differ from native PTS timeline")
                    count += 1
                    if frame_index == 0 or frame_index in chosen:
                        if frame_index == 0:
                            thumb = cv2.resize(image, (320, 180))
                            caption = np.full((30, 320, 3), 245, np.uint8)
                            cv2.putText(caption, f"{match} {source['average_rate']}fps", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 20, 20), 1)
                            thumbnails.append(np.vstack([thumb, caption]))
                        if frame_index in chosen:
                            row = chosen[frame_index]
                            x, y = int(row["X"]), int(row["Y"])
                            if not 0 <= x < stream["width"] or not 0 <= y < stream["height"]:
                                raise ValueError("Invalid publisher center or missing native PTS")
                            path = args.output / f"{match}_{rally}_f{frame_index:06d}.png"
                            cv2.imwrite(str(path), image)
                            origin = [max(0, x - 16), max(0, y - 16)]
                            crop = image[origin[1]:min(y + 17, stream["height"]), origin[0]:min(x + 17, stream["width"])]
                            zoom = cv2.resize(crop, (264, 264), interpolation=cv2.INTER_NEAREST)
                            caption = np.full((42, 264, 3), 245, np.uint8)
                            cv2.putText(caption, f"{match} f{frame_index} ({x},{y})", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (15, 15, 15), 1)
                            cv2.putText(caption, f"origin({origin[0]},{origin[1]}) 8x", (3, 34), cv2.FONT_HERSHEY_SIMPLEX, .42, (15, 15, 15), 1)
                            crops.append(np.vstack([zoom, caption]))
                            candidates.append({"source_id": match, "frame_index": frame_index,
                                               "publisher_center_xy_px": [x, y], "publisher_visibility": 1,
                                               "image": str(path.resolve().relative_to(ROOT)), "image_sha256": digest(path),
                                               "source_pts": tick, "source_time_base": stream["time_base"],
                                               "timestamp_seconds": float(Fraction(tick) * Fraction(stream["time_base"])),
                                               "crop_origin_xy_px": origin, "manual_reviewed": False})
                    if selected is None:
                        break
                source["decoded_frame_count"] = count
                source["native_timeline_frame_count"] = len(ticks)
                source["clip_duration_seconds"] = float((ticks[-1] - ticks[0]) * Fraction(stream["time_base"]) + 1 / Fraction(stream["avg_frame_rate"]))
                sources.append(source)
                print(f"Reviewed candidate source {match}: {len(chosen)} native positive frames", flush=True)
            finally:
                capture.release()
    if selected is not None and selected != {s["source_id"] for s in sources}:
        raise ValueError("Requested source is absent from historic training acquisition")
    def sheets(items, prefix, columns, page_size):
        for start in range(0, len(items), page_size):
            page = items[start:start + page_size]
            while len(page) % columns:
                page.append(np.full_like(page[0], 245))
            canvas = np.vstack([np.hstack(page[i:i + columns]) for i in range(0, len(page), columns)])
            cv2.imwrite(str(args.output / f"{prefix}_{start // page_size:02d}.jpg"), canvas)
    sheets(thumbnails, "scope", 5, 30)
    if crops:
        sheets(crops, "ball_crops", 4, 24)
    receipt = {"schema_version": "1.0", "status": "PUBLISHER_LABEL_CANDIDATES_REQUIRE_VISUAL_REVIEW",
               "sources": sources, "candidates": candidates, "final_test_used": False,
               "match_independence_verified": False, "qualification_evidence": False,
               "rights_basis": "Publisher MIT dataset card/training instructions; private local academic use, no raw public redistribution"}
    (args.output / "review_inventory.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"sources": len(sources), "candidates": len(candidates), "receipt": str(args.output / "review_inventory.json")}))


if __name__ == "__main__":
    main()
