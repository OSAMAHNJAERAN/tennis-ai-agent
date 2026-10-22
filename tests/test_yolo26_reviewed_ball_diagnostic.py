"""Protocol/source-binding checks; these tests perform no model inference."""
import builtins
import json

import pytest

from scripts.evaluate import diagnose_yolo26_reviewed_ball as diagnostic


def test_real_pretraining_protocol_keeps_every_dev_target_and_one_complete_absence_crop(monkeypatch):
    original_import = builtins.__import__
    attempted = []
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics", "cv2", "PIL", "numpy", "psutil"}:
            attempted.append(name)
            raise AssertionError("Metadata preflight imported image/model runtime")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = diagnostic.prepare_protocol(diagnostic.ROOT / "outputs/yolo26_ball_reviewed_development_run1", "2026-10-15T09:00:00+00:00")
    assert attempted == []
    assert len(result["positive_samples"]) == result["all_target_denominator"] == 20
    assert len({(row["clip"], row["frame_index"]) for row in result["positive_samples"]}) == 20
    assert len({row["component"] for row in result["positive_samples"]}) == 3
    assert [row["id"] for row in result["negative_samples"]] == ["tile_review_0113"]
    assert result["actual_image_forward_count"] == 161
    assert result["qualification_evidence"] is result["final_test_used"] is False
    assert result["training_authorized"] is False
    assert "UNADJUDICATED" in result["unmatched_full_frame_candidates"]


def test_grid_covers_all_native_pixels_and_does_not_depend_on_target_boxes():
    for y in range(1080):
        intervals = sorted((left, right) for left, top, right, bottom in diagnostic.GRID if top <= y < bottom)
        covered = 0
        for left, right in intervals:
            assert left <= covered
            covered = max(covered, right)
        assert covered == 1920
    assert all(right-left == bottom-top == 640 for left, top, right, bottom in diagnostic.GRID)
    assert diagnostic.INFERENCE["conf"] == .01
    assert diagnostic.INFERENCE["rect"] is False
    assert diagnostic.INFERENCE["end2end"] is True
    assert diagnostic.INFERENCE["nms"] is False


def test_preparation_is_deterministic_for_a_retained_freeze_time():
    path = diagnostic.ROOT / "outputs/yolo26_ball_reviewed_development_run1"
    first = diagnostic.prepare_protocol(path, "2026-10-15T09:00:00+00:00")
    assert first == diagnostic.prepare_protocol(path, first["frozen_at_utc"])


def test_changed_native_source_is_rejected_before_model_import(monkeypatch):
    original = diagnostic.binding
    def changed(path, expected=None):
        if "match177_000_f000349.png" in str(path):
            raise ValueError("Frozen diagnostic input changed")
        return original(path, expected)
    monkeypatch.setattr(diagnostic, "binding", changed)
    with pytest.raises(ValueError, match="changed"):
        diagnostic.prepare_protocol(diagnostic.ROOT / "outputs/new_run")


def test_paths_outside_private_workspace_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="workspace"):
        diagnostic.workspace_path(diagnostic.ROOT.parent / "outside_workspace")


@pytest.fixture
def completed_run(tmp_path):
    run = tmp_path / "training_run"
    checkpoint = run / "training/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"a small fake checkpoint; never loaded")
    metadata = {"role": "ball", "mode": "REVIEWED_NATIVE_BALL_DEVELOPMENT", "status": "TRAINED_UNQUALIFIED",
                "qualification_evidence": False, "final_test_used": False, "started_utc": "2026-10-15T09:01:00+00:00",
                "checkpoints": [{"path": str(checkpoint), "sha256": diagnostic.digest(checkpoint)}]}
    protocol = {"training_run_directory": str(run), "frozen_at_utc": "2026-10-15T09:00:00+00:00"}
    def save():
        (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return protocol
    return metadata, checkpoint, save


@pytest.mark.parametrize("field,value", [("role", "court"), ("mode", "GENERIC_PILOT"),
    ("status", "TRAINING_STARTED_UNQUALIFIED"), ("qualification_evidence", True), ("final_test_used", True)])
def test_incomplete_or_wrong_training_run_cannot_supply_checkpoint(completed_run, field, value):
    metadata, _, save = completed_run
    metadata[field] = value
    with pytest.raises(ValueError, match="completed private"):
        diagnostic.validate_checkpoint(save())


@pytest.mark.parametrize("start", ["2026-10-15T08:59:00+00:00", "2026-10-15T09:00:00+00:00"])
def test_posthoc_or_equal_training_protocol_freeze_is_refused(completed_run, start):
    metadata, _, save = completed_run
    metadata["started_utc"] = start
    with pytest.raises(ValueError, match="before.*training"):
        diagnostic.validate_checkpoint(save())


def test_checkpoint_substitution_is_rejected_before_loading(completed_run):
    _, checkpoint, save = completed_run
    protocol = save()
    checkpoint.write_bytes(b"a different checkpoint")
    with pytest.raises(ValueError, match="registered"):
        diagnostic.validate_checkpoint(protocol)


def test_duplicate_registered_best_is_rejected(completed_run):
    metadata, _, save = completed_run
    metadata["checkpoints"] *= 2
    with pytest.raises(ValueError, match="registered"):
        diagnostic.validate_checkpoint(save())


def test_successful_checkpoint_binding_requires_completed_export_verification(completed_run, tmp_path, monkeypatch):
    from scripts.data import export_yolo26_reviewed_ball_wave as exporter
    metadata, checkpoint, save = completed_run
    output = tmp_path / "export"
    output.mkdir()
    manifest = output / "source_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    metadata["data_audit"] = {"manifest_sha256": diagnostic.digest(manifest)}
    protocol = save()
    protocol["bindings"] = {"export_manifest": diagnostic.binding(manifest)}
    monkeypatch.setattr(diagnostic, "EXPORT", output)
    calls = []
    monkeypatch.setattr(exporter, "verify_export", lambda path: calls.append(path))
    actual_checkpoint, bindings = diagnostic.validate_checkpoint(protocol)
    assert actual_checkpoint == checkpoint
    assert calls == [output]
    assert bindings["checkpoint"]["sha256"] == diagnostic.digest(checkpoint)
    assert bindings["export_manifest"] == protocol["bindings"]["export_manifest"]


def test_changed_export_cannot_reuse_a_pretraining_protocol(completed_run, tmp_path, monkeypatch):
    metadata, _, save = completed_run
    output = tmp_path / "export"
    output.mkdir()
    manifest = output / "source_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    metadata["data_audit"] = {"manifest_sha256": diagnostic.digest(manifest)}
    protocol = save()
    protocol["bindings"] = {"export_manifest": diagnostic.binding(manifest)}
    monkeypatch.setattr(diagnostic, "EXPORT", output)
    manifest.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="data binding"):
        diagnostic.validate_checkpoint(protocol)


def test_manual_visible_center_is_never_labeled_as_publisher_truth():
    from scripts.data import diagnose_yolo26_native_tiles as scoring
    candidate = {"candidate_id": "x", "global_box_xyxy_px": [10, 20, 14, 24], "confidence": .5}
    result = diagnostic.score_reviewed_target(scoring, [candidate], {"target_box": [10, 20, 14, 24], "visible_center": [12, 22]})
    assert result["matched_target_at_iou05"] is True
    assert result["candidate_target_assessments"][0]["visible_extent_center_error_native_px"] == 0
    assert "publisher_center_error_native_px" not in result["candidate_target_assessments"][0]


def test_missing_target_remains_a_miss_in_aggregate():
    from scripts.data import diagnose_yolo26_native_tiles as scoring
    result = diagnostic.score_reviewed_target(scoring, [], {"target_box": [10, 20, 14, 24], "visible_center": [12, 22]})
    aggregate = scoring.aggregate([{"native_tiles": result}])
    assert aggregate["reviewed_targets"] == aggregate["missed_targets_conf001_iou05"] == 1
    assert aggregate["matched_targets_conf001_iou05"] == 0
    assert aggregate["median_matched_candidate_center_error_native_px"] is None
