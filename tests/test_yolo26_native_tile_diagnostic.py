import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import diagnose_yolo26_native_tiles as diagnostic


def proposal(identifier, box, confidence=.1):
    return {"candidate_id": identifier, "global_box_xyxy_px": box,
            "confidence": confidence, "eligible_for_scoring": True}


def test_fixed_edge_grid_covers_all_native_pixels_without_target_input():
    regions = diagnostic.tile_grid(1920, 1080)
    assert regions == [(x, y, x + 640, y + 640) for y in (0, 440) for x in (0, 512, 1024, 1280)]
    coverage = np.zeros((1080, 1920), dtype=bool)
    for left, top, right, bottom in regions:
        coverage[top:bottom, left:right] = True
        assert (right - left, bottom - top) == (640, 640)
    assert coverage.all()
    with pytest.raises(ValueError, match="native1920x1080"):
        diagnostic.tile_grid(1280, 720)


def test_mapping_adds_origin_once_and_retains_raw_clipped_and_invalid_rows():
    rows = diagnostic.map_candidates([[10, 20, 18, 26], [-10, 0, 700, 700], [np.nan, 1, 2, 3]],
                                     [.2, .1, .3], [0, 0, 0], (1280, 440, 1920, 1080), "t7")
    assert len(rows) == 3
    assert rows[0]["global_box_xyxy_px"] == [1290., 460., 1298., 466.]
    assert rows[0]["tile_origin_xy_px"] == [1280, 440]
    assert rows[1]["raw_global_box_xyxy_px"] == [1270., 440., 1980., 1140.]
    assert rows[1]["global_box_xyxy_px"] == [1270., 440., 1920., 1080.]
    assert rows[2]["eligible_for_scoring"] is False
    assert rows[2]["raw_global_box_xyxy_px"][0] is None
    json.dumps(rows, allow_nan=False)


def test_duplicate_associations_use_frozen_iou_or_center_rule_and_keep_raw():
    raw = [proposal("strong", [10, 10, 20, 20], .3),
           proposal("iou", [11, 10, 21, 20], .2),
           proposal("center", [14, 14, 16, 16], .1),
           proposal("separate", [40, 40, 42, 42], .05)]
    accepted, associations = diagnostic.deduplicate(raw)
    assert [item["candidate_id"] for item in accepted] == ["strong", "separate"]
    assert [item["candidate_id"] for item in associations] == ["iou", "center"]
    assert associations[0]["iou_rule_triggered"] is True
    assert associations[1]["iou_rule_triggered"] is False
    assert associations[1]["center_rule_triggered"] is True
    assert all(item["representative_candidate_id"] == "strong" for item in associations)
    assert len(raw) == 4


def test_duplicate_comparison_is_against_accepted_representatives_with_stable_ties():
    raw = [proposal("first", [0, 0, 2, 2]), proposal("middle", [2, 0, 4, 2]),
           proposal("last", [4, 0, 6, 2])]
    accepted, associations = diagnostic.deduplicate(raw)
    assert [item["candidate_id"] for item in accepted] == ["first", "last"]
    assert associations[0]["representative_candidate_id"] == "first"


def test_one_target_matches_only_one_candidate_and_reports_native_center_error():
    rows = [proposal("exact", [10, 20, 18, 26], .2), proposal("duplicate", [10, 20, 18, 26], .1),
            proposal("far", [100, 100, 108, 106], .9)]
    scored = diagnostic.score_candidates(rows, [10, 20, 18, 26], [14, 23])
    assert scored["matched_target_at_iou05"] is True
    assert scored["matched_candidate_id"] == "exact"
    assert scored["unmatched_or_duplicate_candidates"] == 2
    assert sum(item["one_to_one_match"] for item in scored["candidate_target_assessments"]) == 1
    assert scored["matched_candidate_center_error_native_px"] == 0
    assert scored["nearest_candidate_center_error_native_px"] == 0


def test_near_center_with_low_iou_is_still_a_miss_and_empty_returns_no_error():
    scored = diagnostic.score_candidates([proposal("small", [13, 22, 15, 24])], [10, 20, 18, 26], [14, 23])
    assert scored["nearest_candidate_center_error_native_px"] == 0
    assert scored["matched_target_at_iou05"] is False
    assert scored["unmatched_or_duplicate_candidates"] == 1
    empty = diagnostic.score_candidates([], [10, 20, 18, 26], [14, 23])
    assert empty["nearest_candidate_center_error_native_px"] is None
    aggregate = diagnostic.aggregate([{"native_tiles": empty}])
    assert aggregate["missed_targets_conf001_iou05"] == 1
    assert aggregate["median_nearest_candidate_center_error_native_px"] is None


def test_inference_resource_guard_requires_cuda_initialized_headroom():
    resources = {"free_disk_bytes": 2 * 1024 ** 3, "free_ram_bytes": 2 * 1024 ** 3,
                 "free_vram_bytes": 3 * 1024 ** 3}
    diagnostic.check_resources(resources)
    resources["free_ram_bytes"] = 1024 ** 3 - 1
    with pytest.raises(ValueError, match="after CUDA initialization"):
        diagnostic.check_resources(resources)
    resources["free_ram_bytes"] = 2 * 1024 ** 3
    resources["free_vram_bytes"] = None
    with pytest.raises(ValueError, match="2 GiB free VRAM"):
        diagnostic.check_resources(resources)


def test_prepare_does_not_import_torch_or_run_inference_and_refuses_protocol_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "OUTPUT", tmp_path)
    monkeypatch.setattr(diagnostic, "prepare_protocol", lambda: {"samples": [1], "frozen": True})
    monkeypatch.setattr(diagnostic, "execute", lambda _: pytest.fail("Preparation must not infer"))
    before = sys.modules.get("torch")
    diagnostic.main(["--prepare"])
    assert sys.modules.get("torch") is before
    assert diagnostic.read_json(tmp_path / "protocol.json")["frozen"] is True
    monkeypatch.setattr(diagnostic, "prepare_protocol", lambda: {"samples": [2], "frozen": False})
    with pytest.raises(ValueError, match="do not silently replace"):
        diagnostic.main(["--prepare"])
    with pytest.raises(ValueError, match="inputs or script changed"):
        diagnostic.main(["--execute"])


def test_resource_rejection_is_persisted_before_model_construction(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "OUTPUT", tmp_path)
    diagnostic.write_json(tmp_path / "protocol.json", {"frozen": True})
    fake_cuda = SimpleNamespace(is_available=lambda: True, init=lambda: None,
                                mem_get_info=lambda _: (3 * 1024 ** 3, 6 * 1024 ** 3))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=1024 ** 3 - 1)))
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=lambda *_, **__: pytest.fail("Rejected resource guard must not construct a model")))
    monkeypatch.setattr(diagnostic.shutil, "disk_usage", lambda _: SimpleNamespace(free=2 * 1024 ** 3))
    protocol = {"resource_guard_floors_bytes": {"free_ram_bytes": 1024 ** 3}, "limitations": ["Positive only"]}
    diagnostic.execute(protocol)
    receipt = diagnostic.read_json(tmp_path / "diagnostic.json")
    assert receipt["status"] == "RESOURCE_GUARD_BLOCKED_NOT_INFERRED"
    assert receipt["model_constructed"] is False
    assert receipt["inference_performed"] is False
    assert receipt["resource_guard"]["free_ram_bytes"] == 1024 ** 3 - 1
    assert receipt["qualification_evidence"] is False


def test_cli_output_is_workspace_contained_and_cannot_overwrite_terminal_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "OUTPUT", tmp_path / "default-output")
    monkeypatch.setattr(diagnostic, "prepare_protocol", lambda: {"samples": [1], "frozen": True})
    output = tmp_path / "fresh-v2"
    diagnostic.main(["--prepare", "--output", str(output)])
    assert diagnostic.read_json(output / "protocol.json")["frozen"] is True
    diagnostic.write_json(output / "diagnostic.json", {"status": "RESOURCE_GUARD_BLOCKED_NOT_INFERRED"})
    before = (output / "diagnostic.json").read_bytes()
    for action in ("--prepare", "--execute"):
        with pytest.raises(ValueError, match="Preserve existing terminal"):
            diagnostic.main([action, "--output", str(output)])
    assert (output / "diagnostic.json").read_bytes() == before
    with pytest.raises(ValueError, match="inside the private workspace"):
        diagnostic.main(["--prepare", "--output", str(tmp_path.parent / "outside-workspace")])
