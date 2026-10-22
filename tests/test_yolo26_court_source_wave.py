"""Selection and archive-boundary invariants for metadata-only court repair."""

import copy
import math

import pytest

from scripts.data.plan_yolo26_court_source_wave import bounded_member, inspect_points, parse_id, select_rows


def row(video, ordinal=0):
    return {"id": f"{video}_{ordinal}", "kps": [[100 + index, 200] for index in range(14)]}


def cached(sample):
    return {"id": sample["id"], "source_video_id": sample["id"].rsplit("_", 1)[0],
            "publisher_keypoints_xy_px": sample["kps"]}


def test_existing_video_is_excluded_from_both_original_splits():
    old = row("existing001", 1)
    documents = {"data_train.json": [old, row("newvideo001", 3)],
                 "data_val.json": [row("existing001", 99), row("newvideo001", 2), row("newvideo002", 4)]}
    selected, candidates, counts = select_rows(documents, [cached(old)])
    assert len(selected) == 2
    assert {item["source_video_id"] for item in selected} == {"newvideo001", "newvideo002"}
    assert counts["publisher_cross_split_video_ids"] == 2
    assert len(candidates) == 2


def test_hash_selection_is_stable_against_document_and_row_order():
    documents = {"data_train.json": [row(f"video{index:06}", 5) for index in range(80)],
                 "data_val.json": [row(f"video{index:06}", 10) for index in range(80)]}
    first = select_rows(documents, [])
    reversed_documents = {key: list(reversed(value)) for key, value in reversed(list(documents.items()))}
    second = select_rows(reversed_documents, [])
    assert [item["id"] for item in first[0]] == [item["id"] for item in second[0]]
    assert first[1] == second[1]
    assert len(first[0]) == 64
    assert len({item["source_video_id"] for item in first[0]}) == 64


def test_outside_and_null_slots_are_preserved_without_clamping():
    sample = row("newvideo001")
    sample["kps"][0] = [-3, 9]
    sample["kps"][1] = [1280, 719]
    sample["kps"][2] = None
    before = copy.deepcopy(sample)
    selected, _, counts = select_rows({"train": [sample], "val": []}, [])
    assert selected[0]["publisher_keypoints_xy_px"] == before["kps"]
    assert selected[0]["out_of_image_keypoint_indices"] == [0, 1]
    assert selected[0]["publisher_null_keypoint_indices"] == [2]
    assert counts["selected_with_out_of_image_points"] == 1
    assert counts["selected_null_keypoint_slots"] == 1
    assert sample == before


@pytest.mark.parametrize("count", [0, 65, -1, True, 1.5])
def test_fixed_wave_limit_cannot_be_bypassed(count):
    with pytest.raises(ValueError):
        select_rows({"train": [], "val": []}, [], count)


def test_duplicate_publisher_image_across_splits_fails_closed():
    sample = row("newvideo001")
    with pytest.raises(ValueError, match="Duplicate"):
        select_rows({"train": [sample], "val": [sample]}, [])


def test_existing_inventory_must_still_match_original_proposals():
    sample = row("existing001")
    old = cached(sample)
    old["publisher_keypoints_xy_px"] = copy.deepcopy(sample["kps"])
    old["publisher_keypoints_xy_px"][0] = [13, 14]
    with pytest.raises(ValueError, match="proposal differs"):
        select_rows({"train": [sample], "val": []}, [old])


@pytest.mark.parametrize("identifier", ["bad_2", "newvideo001_-1", "newvideo001_0.png", "../newvideo001_0", None])
def test_identifier_must_be_canonical_publisher_handle(identifier):
    with pytest.raises(ValueError):
        parse_id(identifier)


@pytest.mark.parametrize("point", [[True, 4], [math.nan, 4], [math.inf, 4], [1], "1,2"])
def test_invalid_point_slots_are_not_normalized_into_labels(point):
    points = row("newvideo001")["kps"]
    points[0] = point
    with pytest.raises(ValueError):
        inspect_points(points)


def entry():
    return {"name": "data/images/newvideo001_0.png", "bytes": 900_000,
            "compressed_bytes": 850_000, "header_offset": 10_000,
            "crc32": 0x12345678, "compression": 8}


def test_member_budget_includes_maximum_reviewed_local_header_allowance():
    member = entry()
    bound = bounded_member(member, "newvideo001_0", 7_255_696_316)
    assert bound["planned_range_bytes_upper_bound"] == 850_000 + 30 + 1024 + len(member["name"])
    assert bound["planned_range_start"] == 10_000


@pytest.mark.parametrize("field,value", [("compression", 0), ("compressed_bytes", 5_000_001),
                                          ("bytes", 0), ("crc32", -1), ("crc32", 2**32),
                                          ("header_offset", -1), ("compressed_bytes", True)])
def test_invalid_or_oversized_member_fails_before_network(field, value):
    member = entry()
    member[field] = value
    with pytest.raises(ValueError):
        bounded_member(member, "newvideo001_0", 7_255_696_316)


def test_range_cannot_leave_pinned_archive():
    with pytest.raises(ValueError, match="leaves"):
        bounded_member(entry(), "newvideo001_0", 860_000)
