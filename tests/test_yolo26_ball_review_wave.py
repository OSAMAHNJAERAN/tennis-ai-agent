"""Review preparation must preserve groups, actual timing, and native pixels."""
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.data.prepare_yolo26_ball_review_wave import (
    GROUPS, exact_context, image_record, native_to_patch_box, nearest_pts_index,
    patch_origin, validate_group_split, validate_groups, verify_timeline,
)


def test_context_uses_actual_vfr_pts_including_nonzero_origin_and_exact_ties():
    ticks = [100, 140, 180, 220, 301]
    context = exact_context(ticks, "1/1000", 2)
    assert [item["frame_index"] for item in context] == [0, 2, 4]
    assert context[0]["offset_fraction_seconds"] == "-2/25"
    assert context[2]["timestamp_fraction_seconds"] == "301/1000"
    assert nearest_pts_index(ticks, Fraction(200)) == 2
    assert exact_context(ticks, "1/1000", 0)[0]["frame_index"] == 0


@pytest.mark.parametrize("ticks,time_base", [([], "1/25"), ([0, 0], "1/25"), ([2, 1], "1/25"), ([0, 1], "0"), ([0, 1.5], "1/25")])
def test_invalid_or_manufactured_timelines_are_rejected(ticks, time_base):
    with pytest.raises(ValueError):
        verify_timeline(ticks, time_base)


def test_clamped_native_patch_origin_preserves_box_round_trip_at_image_edges():
    origin = patch_origin([1918, 1077])
    assert origin == [1824, 984]
    native = [1910, 1069, 1920, 1080]
    local = native_to_patch_box(native, origin)
    assert local == [86, 85, 96, 96]
    assert [local[i] + origin[i % 2] for i in range(4)] == native
    assert patch_origin([1, 2]) == [0, 0]
    with pytest.raises(ValueError):
        patch_origin([1920, 3])


def test_event_groups_cover_omitted_source_once_and_never_mix_train_dev():
    all_ids = {source for members in GROUPS.values() for source in members}
    assert len(all_ids) == 60
    assert "match210" in GROUPS["unresolved_event_quarantine"]
    validate_groups(GROUPS, all_ids)
    rows = [{"clip": "match177_000", "frame_index": 9, "conservative_event_group": "washington", "proposed_split": "development"},
            {"clip": "match215_000", "frame_index": 20, "conservative_event_group": "washington", "proposed_split": "train"}]
    with pytest.raises(ValueError, match="spans training and development"):
        validate_group_split(rows)
    with pytest.raises(ValueError):
        validate_groups({"first": ["same"], "second": ["same"]}, {"same"})


def test_source_frame_duplicates_are_rejected_before_review():
    row = {"clip": "match172_000", "frame_index": 9, "conservative_event_group": "beijing", "proposed_split": "train"}
    with pytest.raises(ValueError, match="Duplicate source frame"):
        validate_group_split([row, dict(row)])


def test_saved_patch_png_retains_exact_pixels_and_shape(tmp_path):
    import cv2
    pixels = np.zeros((96, 96, 3), dtype=np.uint8)
    pixels[1:4, 40:59] = [16, 232, 211]
    record = image_record(tmp_path / "patch.png", pixels)
    assert record["shape"] == [96, 96, 3]
    assert np.array_equal(cv2.imread(record["path"]), pixels)
    assert len(record["sha256"]) == len(record["pixel_sha256"]) == 64


def test_metadata_entry_imports_no_image_or_model_stack():
    result = subprocess.check_output([sys.executable, "-c", "import sys,json; import scripts.data.prepare_yolo26_ball_review_wave; print(json.dumps([name for name in ('cv2','numpy','torch','ultralytics') if name in sys.modules]))"], text=True)
    assert json.loads(result) == []


def test_pure_sampler_retains_prior_sampling_without_importing_image_helpers():
    from scripts.data.yolo26_ball_review_metadata import select_positive
    from scripts.data.prepare_ball_review_expansion import select_positive as retained
    pool = [{"clip": f"match{i // 20}", "frame_index": i,
             "publisher_center_xy_px": [100, (i * 7) % 1080],
             "labeled_image_motion_proxy_px_per_second": float((i * 13) % 97)} for i in range(100)]
    assert select_positive(pool, 37) == retained(pool, 37)


def test_card_reconciliation_accepts_only_exact_original_text_with_windows_newlines(tmp_path):
    import hashlib
    from scripts.data.yolo26_ball_review_metadata import verify_dataset_card
    original = b"---\nlicense: mit\n---\nPublisher text\n"
    expected = hashlib.sha256(original).hexdigest()
    path = tmp_path / "card.md"
    path.write_bytes(original.replace(b"\n", b"\r\n"))
    audit = verify_dataset_card(path, expected)
    assert audit["strict_crlf_to_lf_sha256"] == expected
    assert audit["local_raw_card_sha256"] != expected
    path.write_bytes(original.replace(b"mit", b"other"))
    with pytest.raises(ValueError, match="text changed"):
        verify_dataset_card(path, expected)
