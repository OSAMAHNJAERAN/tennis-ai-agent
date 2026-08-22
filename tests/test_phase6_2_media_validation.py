"""
Unit tests for Phase 6.2 Real-Media Validation, Physical Video Integrity,
and Anti-Circularity Guarantees.
"""

import os
import json
import hashlib
import pytest
import cv2


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def test_physical_video_files_exist_and_hash_match():
    """Verify all 5 benchmark videos exist on disk and match SHA256 checksums."""
    videos_file = "data/benchmarks/shot_classification_real_v2/videos.json"
    assert os.path.exists(videos_file), f"Missing {videos_file}"

    with open(videos_file) as f:
        videos_data = json.load(f)["videos"]

    assert len(videos_data) == 5, f"Expected 5 videos, found {len(videos_data)}"

    for vid, vinfo in videos_data.items():
        vpath = vinfo["path"]
        assert os.path.exists(vpath), f"Video file {vpath} does not exist!"
        assert os.path.getsize(vpath) > 100_000, f"Video {vpath} suspiciously small!"

        # Verify SHA256
        actual_sha = compute_sha256(vpath)
        assert actual_sha == vinfo["sha256"], f"SHA mismatch for {vid}: {actual_sha} vs {vinfo['sha256']}"

        # Verify OpenCV readability
        cap = cv2.VideoCapture(vpath)
        assert cap.isOpened(), f"Could not open {vpath} with cv2"
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        assert frame_count == vinfo["frame_count"], f"Frame count mismatch for {vid}: {frame_count} vs {vinfo['frame_count']}"
        assert f"{w}x{h}" == vinfo["resolution"], f"Resolution mismatch for {vid}: {w}x{h} vs {vinfo['resolution']}"


def test_video_source_splits_strict_disjointness():
    """Verify 0 video ID overlap and 0 SHA256 overlap across Dev, Val, Test."""
    splits_file = "data/benchmarks/shot_classification_real_v2/splits.json"
    videos_file = "data/benchmarks/shot_classification_real_v2/videos.json"

    with open(splits_file) as f:
        splits = json.load(f)["splits"]
    with open(videos_file) as f:
        videos = json.load(f)["videos"]

    dev_vids = set(splits["development"]["video_ids"])
    val_vids = set(splits["validation"]["video_ids"])
    test_vids = set(splits["held_out_test"]["video_ids"])

    # Disjointness
    assert dev_vids.isdisjoint(val_vids), "Dev and Val share videos!"
    assert dev_vids.isdisjoint(test_vids), "Dev and Test share videos!"
    assert val_vids.isdisjoint(test_vids), "Val and Test share videos!"

    # SHA disjointness
    dev_shas = {videos[v]["sha256"] for v in dev_vids}
    val_shas = {videos[v]["sha256"] for v in val_vids}
    test_shas = {videos[v]["sha256"] for v in test_vids}

    assert dev_shas.isdisjoint(val_shas), "Dev and Val share video content hashes!"
    assert dev_shas.isdisjoint(test_shas), "Dev and Test share video content hashes!"
    assert val_shas.isdisjoint(test_shas), "Val and Test share video content hashes!"


def test_ground_truth_count_reconciliation():
    """Verify sum(class_counts) == total_strokes and sum(split_counts) == total_strokes."""
    gt_file = "data/benchmarks/shot_classification_real_v2/ground_truth.json"
    splits_file = "data/benchmarks/shot_classification_real_v2/splits.json"

    with open(gt_file) as f:
        strokes = json.load(f)["strokes"]
    with open(splits_file) as f:
        splits = json.load(f)["splits"]

    total_strokes = len(strokes)
    assert total_strokes == 26, f"Expected 26 total strokes, got {total_strokes}"

    # Class counts
    class_counts = {}
    for s in strokes:
        st = s["shot_type"]
        class_counts[st] = class_counts.get(st, 0) + 1

    assert sum(class_counts.values()) == total_strokes
    assert class_counts["FOREHAND"] == 11
    assert class_counts["BACKHAND"] == 8
    assert class_counts["SERVE"] == 5
    assert class_counts["UNKNOWN"] == 2

    # Split counts
    split_stroke_sum = sum(len(sdata["stroke_ids"]) for sdata in splits.values())
    assert split_stroke_sum == total_strokes
    assert len(splits["development"]["stroke_ids"]) == 3
    assert len(splits["validation"]["stroke_ids"]) == 10
    assert len(splits["held_out_test"]["stroke_ids"]) == 13


def test_ground_truth_frame_bounds():
    """Verify all stroke frame_hit values are strictly within physical video frame limits."""
    gt_file = "data/benchmarks/shot_classification_real_v2/ground_truth.json"
    videos_file = "data/benchmarks/shot_classification_real_v2/videos.json"

    with open(gt_file) as f:
        strokes = json.load(f)["strokes"]
    with open(videos_file) as f:
        videos = json.load(f)["videos"]

    for s in strokes:
        vid = s["video_id"]
        assert vid in videos, f"Unknown video_id {vid} in ground truth!"
        v_frames = videos[vid]["frame_count"]
        f_hit = s["frame_hit"]
        assert 0 <= f_hit < v_frames, f"Stroke {s['stroke_id']} frame_hit {f_hit} out of bounds for {vid} (0..{v_frames-1})"


def test_evaluator_zero_feature_injection():
    """Verify Phase6Pipeline contract does not accept ground-truth hints."""
    from src.pipeline.phase6_pipeline import Phase6Pipeline
    pipeline = Phase6Pipeline()

    import inspect
    sig = inspect.signature(pipeline.run)
    param_names = list(sig.parameters.keys())

    # Only video_path, output_dir, max_frames, save_video are allowed
    allowed = {"input_video_path", "output_dir", "max_frames", "save_video"}
    for p in param_names:
        assert p in allowed, f"Pipeline.run exposes unauthorized feature injection parameter: {p}"
