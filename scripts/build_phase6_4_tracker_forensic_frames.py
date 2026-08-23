"""Build post-hoc visual evidence for the remaining Phase 6.4 tracker losses."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = "video_10"
FRAMES = (110, 122, 218, 232)


def main() -> None:
    videos = json.loads(
        (ROOT / "data/benchmarks/cross_match_final_holdout/videos.json").read_text(encoding="utf-8")
    )["videos"]
    raw = json.loads(
        (ROOT / f"artifacts/validation/raw_candidates/{VIDEO_ID}_candidates.json").read_text(encoding="utf-8")
    )
    output = ROOT / "artifacts/validation/phase6_4_tracker_forensic_frames"
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(ROOT / videos[VIDEO_ID]["path"]))
    if not capture.isOpened():
        raise RuntimeError("Could not open diagnostic video")
    try:
        for frame_index in FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index}")
            for candidate_index, candidate in enumerate(raw["frames"][frame_index], 1):
                center = (round(float(candidate["x_px"])), round(float(candidate["y_px"])))
                cv2.circle(image, center, 18, (0, 0, 255), 3)
                cv2.putText(
                    image,
                    f"C{candidate_index} {float(candidate['confidence']):.3f}",
                    (center[0] + 20, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                image,
                f"POST-HOC FORENSIC ONLY | {VIDEO_ID} frame {frame_index}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            path = output / f"{VIDEO_ID}_frame_{frame_index:04d}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not write {path}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
