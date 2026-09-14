"""Generate raw-video-only contact sheets for annotation coverage review.

This utility deliberately reads only the media manifest and raw video frames.
It does not read predictions, tracks, event outputs, or FP labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "benchmarks" / "cross_match_final_holdout" / "videos.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "validation" / "phase6_4_coverage_raw_review"


def generate_sheet(
    video_id: str, metadata: dict, output_dir: Path, sample_interval_s: float
) -> dict:
    source = ROOT / metadata["path"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open raw video: {source}")
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    sample_step = max(1, round(fps * sample_interval_s))
    thumbnail_width, thumbnail_height = 320, 180
    columns = 5
    sampled = []
    for frame_index in range(0, frame_count, sample_step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Unable to read {video_id} frame {frame_index}")
        frame = cv2.resize(frame, (thumbnail_width, thumbnail_height))
        label = f"{video_id} f={frame_index} t={frame_index / fps:.2f}s"
        cv2.rectangle(frame, (0, 0), (thumbnail_width, 24), (0, 0, 0), -1)
        cv2.putText(frame, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sampled.append((frame_index, frame))
    capture.release()

    page_size = 20
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for page_number, offset in enumerate(range(0, len(sampled), page_size), start=1):
        page_frames = sampled[offset : offset + page_size]
        rows = (len(page_frames) + columns - 1) // columns
        canvas = cv2.copyMakeBorder(
            page_frames[0][1], 0, thumbnail_height * rows - thumbnail_height,
            0, thumbnail_width * columns - thumbnail_width, cv2.BORDER_CONSTANT, value=(32, 32, 32)
        )
        for index, (_, frame) in enumerate(page_frames):
            row, column = divmod(index, columns)
            canvas[
                row * thumbnail_height : (row + 1) * thumbnail_height,
                column * thumbnail_width : (column + 1) * thumbnail_width,
            ] = frame
        interval_tag = f"{sample_interval_s:g}s".replace(".", "p")
        path = output_dir / f"{video_id}_raw_{interval_tag}_page_{page_number:02d}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError(f"Unable to write {path}")
        pages.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "start_frame": page_frames[0][0],
                "end_frame": page_frames[-1][0],
            }
        )
    return {
        "video_id": video_id,
        "source_path": metadata["path"],
        "source_sha256": metadata["sha256"],
        "sampling_basis": "RAW_VIDEO_ONLY",
        "sample_interval_seconds": sample_step / fps,
        "sampled_frame_count": len(sampled),
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    args = parser.parse_args()
    if not args.manifest.is_absolute():
        args.manifest = ROOT / args.manifest
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = [
        generate_sheet(video_id, metadata, args.output_dir, args.sample_interval_s)
        for video_id, metadata in manifest["videos"].items()
    ]
    review_manifest = {
        "schema_version": "1.0",
        "purpose": "RAW_VIDEO_ANNOTATION_COVERAGE_REVIEW",
        "prediction_inputs_used": False,
        "records": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(review_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
