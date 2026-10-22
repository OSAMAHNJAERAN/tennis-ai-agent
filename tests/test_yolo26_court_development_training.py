"""CPU-only contract checks for the reviewed static-court development route."""

import builtins
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.train import train_yolo26_tennis as training


MANIFEST = training.REPO_ROOT / "data/yolo26/court_reviewed_development_v1/source_manifest.json"


@pytest.fixture
def reviewed_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture
def semantic_manifest(tmp_path, monkeypatch, reviewed_manifest):
    """Bypass only the document pin to exercise deeper checks without editing data."""
    path = tmp_path / "source_manifest.json"
    original_sha = training.sha256_file
    pinned_sha = original_sha(MANIFEST)

    def document_pin_only(candidate):
        if Path(candidate).resolve() == path.resolve():
            return pinned_sha
        return original_sha(candidate)

    monkeypatch.setattr(training, "sha256_file", document_pin_only)

    def save(document):
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    return copy.deepcopy(reviewed_manifest), save


def test_exact_reviewed_export_audits_as_development_without_qualification():
    manifest, records, audit = training.audit_manifest(MANIFEST, "court", development=True)
    assert manifest["frozen"] is False
    assert len(records) == 120
    assert {record["split"] for record in records} == {"train", "val"}
    assert audit["split_image_counts"] == {"train": 100, "val": 20, "test": 0}
    assert audit["labeled_instances"] == 120
    assert audit["final_test_used"] is False
    assert audit["independent_validation_available"] is False
    assert manifest["qualification_evidence"] is False
    assert manifest["source_independence_verified"] is False


def test_static_export_cannot_enter_frozen_full_training_route():
    with pytest.raises(ValueError, match="frozen"):
        training.audit_manifest(MANIFEST, "court")


def test_changed_development_document_fails_the_exact_export_pin(tmp_path, reviewed_manifest):
    reviewed_manifest["sources"][0]["source_type"] = "VIDEO"
    path = tmp_path / "source_manifest.json"
    path.write_text(json.dumps(reviewed_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)(hash|sha256|checksum|pinned|exact)"):
        training.audit_manifest(path, "court", development=True)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("sources", "source_type", "ORIGINAL_VIDEO"),
        ("sources", "training_allowed", False),
        ("sources", "fps", 30),
        ("sources", "match_id", "invented-match"),
        ("sources", "source_pts", 100),
        ("sources", "original_video_bytes_acquired", True),
        ("samples", "timestamp_seconds", 3.0),
        ("samples", "source_time_base", "1/30"),
        ("samples", "independent_human_reviewed", True),
        ("samples", "split", "test"),
    ],
)
def test_development_cannot_gain_unreviewed_source_or_timing_claims(semantic_manifest, section, field, value):
    manifest, save = semantic_manifest
    manifest[section][0][field] = value
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


def test_private_training_rights_cannot_be_widened(semantic_manifest):
    manifest, save = semantic_manifest
    manifest["sources"][0]["rights_evidence"]["allowed_scope"] = "PUBLIC_COMMERCIAL_REDISTRIBUTION"
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


@pytest.mark.parametrize("field", ["qualification_evidence", "source_independence_verified", "final_test_used"])
def test_development_cannot_be_relabelled_as_qualified(semantic_manifest, field):
    manifest, save = semantic_manifest
    manifest[field] = True
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


def test_pinned_lineage_cannot_be_replaced(semantic_manifest):
    manifest, save = semantic_manifest
    manifest["lineage"]["decisions_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


def test_review_decision_file_is_rehashed_before_training(monkeypatch):
    relative_path, _ = training.COURT_DEVELOPMENT_LINEAGE["decisions_sha256"]
    decisions_path = (training.REPO_ROOT / relative_path).resolve()
    original_sha = training.sha256_file

    def altered_decisions(candidate):
        return "0" * 64 if Path(candidate).resolve() == decisions_path else original_sha(candidate)

    monkeypatch.setattr(training, "sha256_file", altered_decisions)
    with pytest.raises(ValueError, match="decisions_sha256"):
        training.audit_manifest(MANIFEST, "court", development=True)


def test_unknown_semantic_slot_cannot_gain_inferred_visibility(semantic_manifest):
    manifest, save = semantic_manifest
    sample = next(sample for sample in manifest["samples"] if None in sample["keypoints_px"])
    index = sample["keypoints_px"].index(None)
    sample["keypoints_px"][index] = [640, 360, 2]
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


def test_original_source_video_cannot_cross_development_splits(semantic_manifest):
    manifest, save = semantic_manifest
    train = next(source for source in manifest["sources"] if source["development_split"] == "train")
    val = next(source for source in manifest["sources"] if source["development_split"] == "val")
    val["original_source_video_id"] = train["original_source_video_id"]
    with pytest.raises(ValueError):
        training.audit_manifest(save(manifest), "court", development=True)


def test_native_image_digest_is_checked_without_decoding(monkeypatch, reviewed_manifest):
    image = (training.REPO_ROOT / reviewed_manifest["samples"][0]["image"]).resolve()
    original_sha = training.sha256_file

    def altered_image(candidate):
        return "0" * 64 if Path(candidate).resolve() == image else original_sha(candidate)

    monkeypatch.setattr(training, "sha256_file", altered_image)
    with pytest.raises(ValueError, match="(?i)(image|native|checksum|digest|bytes)"):
        training.audit_manifest(MANIFEST, "court", development=True)


def test_development_dataset_has_real_validation_and_no_test_key(tmp_path):
    records = [{"split": "train", "image": "reviewed_train.png"}, {"split": "val", "image": "reviewed_val.png"}]
    path = training.prepare_dataset(records, "court", tmp_path, development=True)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["train"] == "train.txt"
    assert data["val"] == "val.txt"
    assert "test" not in data
    assert not (tmp_path / "test.txt").exists()
    assert (tmp_path / "train.txt").read_text().splitlines() == ["reviewed_train.png"]
    assert (tmp_path / "val.txt").read_text().splitlines() == ["reviewed_val.png"]
    assert data["kpt_shape"] == [14, 3]
    assert data["flip_idx"] == [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


@pytest.mark.parametrize("records", [[], [{"split": "train", "image": "same_sample.png"}],
                                    [{"split": "val", "image": "val_only.png"}],
                                    [{"split": "train", "image": "train.png"},
                                     {"split": "val", "image": "val.png"},
                                     {"split": "test", "image": "final_test.png"}]])
def test_development_cannot_fall_back_to_overfit_validation_or_consume_test(tmp_path, records):
    with pytest.raises(ValueError):
        training.prepare_dataset(records, "court", tmp_path, development=True)


def test_court_development_resource_floors_are_separate_from_ball():
    resources = {"free_disk_bytes": 1024 ** 3, "free_ram_bytes": 1536 * 1024 ** 2,
                 "free_vram_bytes": 2 * 1024 ** 3}
    training.check_resource_guard(resources, gpu_requested=True, role="court", image_size=1280, development=True)
    for field, message in [("free_disk_bytes", "1 GiB"), ("free_ram_bytes", "1.5 GiB"),
                           ("free_vram_bytes", "2 GiB")]:
        insufficient = dict(resources)
        insufficient[field] -= 1
        with pytest.raises(ValueError, match=message):
            training.check_resource_guard(insufficient, gpu_requested=True, role="court", image_size=1280,
                                          development=True)


@pytest.mark.parametrize("extra", [
    ["--pilot", "--smoke", "--epochs", "1"], ["--smoke", "--epochs", "1"],
    ["--epochs", "0"], ["--epochs", "21"], ["--epochs", "101"], ["--batch", "2"],
    ["--image-size", "960"], ["--patience", "14"], ["--patience", "21"],
])
def test_invalid_development_run_is_rejected_before_data_or_runtime_access(tmp_path, monkeypatch, extra):
    def unexpected_audit(*args, **kwargs):
        pytest.fail("Invalid development arguments reached the data audit")

    monkeypatch.setattr(training, "audit_manifest", unexpected_audit)
    with pytest.raises(SystemExit):
        training.main(["--role", "court", "--output", str(tmp_path), "--development", *extra])


@pytest.mark.parametrize("role", ["ball", "players"])
def test_static_development_route_is_court_only(tmp_path, monkeypatch, role):
    def unexpected_audit(*args, **kwargs):
        pytest.fail("Non-court development request reached the data audit")

    monkeypatch.setattr(training, "audit_manifest", unexpected_audit)
    with pytest.raises(SystemExit):
        training.main(["--role", role, "--output", str(tmp_path), "--development"])


def test_development_preflight_never_imports_training_or_gpu_runtime(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def cpu_only_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "ultralytics", "psutil"}:
            pytest.fail(f"Data preflight imported the training runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", cpu_only_import)
    training.main(["--role", "court", "--manifest", str(MANIFEST), "--output", str(tmp_path), "--development"])
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "PREFLIGHT_PASSED_NOT_TRAINED"
    assert metadata["runtime_probe_performed"] is False
    assert metadata["resource_guard_pending"] is True
    assert metadata["training_arguments"]["epochs"] == 20
    assert metadata["training_arguments"]["patience"] == 20
    assert metadata["training_arguments"]["batch"] == 1
    assert metadata["training_arguments"]["imgsz"] == 1280
    assert metadata["training_arguments"]["workers"] == 0
    assert metadata["training_arguments"]["cache"] is False
    assert metadata["training_arguments"]["amp"] is False
    assert metadata["training_arguments"]["augmentations"] == []
    assert metadata["training_arguments"]["auto_augment"] is None
    for key in ("cutmix", "erasing", "bgr", "multi_scale"):
        assert metadata["training_arguments"][key] == 0
    assert metadata["training_arguments"]["rect"] is False
    assert metadata["data_audit"]["split_image_counts"] == {"train": 100, "val": 20, "test": 0}
    assert not list(tmp_path.rglob("*.pt"))


@pytest.fixture
def development_extension(tmp_path):
    """Synthetic retained files test review binding; these are not model evidence."""
    prior = tmp_path / "synthetic_prior_run"
    checkpoint = prior / "training" / "weights" / "development_epoch20.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint bytes; never load as a model")
    metadata = prior / "run_metadata.json"
    metadata.write_text(json.dumps({
        "role": "court", "mode": "REVIEWED_STATIC_COURT_DEVELOPMENT",
        "data_audit": {"manifest_sha256": training.COURT_DEVELOPMENT_MANIFEST_SHA256},
        "training_arguments": {"epochs": 20},
    }), encoding="utf-8")
    metrics = checkpoint.parents[1] / "results.csv"
    metrics.write_text("epoch,train/pose_loss,val/pose_loss\n1,12,10\n20,3,4\n", encoding="utf-8")
    evidence = {
        "schema": "tennis.yolo26.court-development-extension.v1",
        "status": "ROOT_REVIEWED_USEFUL_LEARNING_EXTENSION_APPROVED",
        "reviewer_type": "ROOT_AGENT_DEVELOPMENT_METRICS_REVIEW",
        "observed_useful_learning": "Synthetic fixture only: DEV pose loss decreased from 10 to 4 across epochs1 and20.",
        "review_note": "Synthetic fixture only: reviewed decreasing DEV loss in the retained CSV.",
        "qualification_evidence": False, "final_test_used": False, "requested_total_epochs": 40,
        "observed_epoch_ids": [1, 20], "checkpoint_epoch": 20, "improved_loss_columns": ["val/pose_loss"],
    }
    for key, path in [("prior_run_metadata", metadata), ("checkpoint", checkpoint), ("metrics_csv", metrics)]:
        evidence[key] = {"path": str(path), "sha256": training.sha256_file(path)}
    evidence_path = tmp_path / "synthetic_extension_review.json"

    def save():
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return evidence_path

    save()
    return evidence, save, checkpoint, metadata, metrics


def test_extension_binds_observed_retained_losses_and_exact_files(development_extension):
    evidence, save, checkpoint, metadata, _ = development_extension
    result = training.validate_development_extension(save(), checkpoint, metadata, 40)
    assert result["review"] == evidence
    assert result["sha256"] == training.sha256_file(save())


@pytest.mark.parametrize("changed", ["checkpoint", "metadata", "metrics"])
def test_extension_rejects_any_changed_retained_file(development_extension, changed):
    _, save, checkpoint, metadata, metrics = development_extension
    path = {"checkpoint": checkpoint, "metadata": metadata, "metrics": metrics}[changed]
    path.write_bytes(path.read_bytes() + b"changed after root review")
    with pytest.raises(ValueError):
        training.validate_development_extension(save(), checkpoint, metadata, 40)


@pytest.mark.parametrize("claim", [False, True, "true"])
def test_extension_needs_concrete_learning_observations_beyond_a_boolean(development_extension, claim):
    evidence, save, checkpoint, metadata, _ = development_extension
    evidence["observed_useful_learning"] = claim
    with pytest.raises(ValueError):
        training.validate_development_extension(save(), checkpoint, metadata, 40)


@pytest.mark.parametrize("csv_rows", ["1,12,10\n20,3,10\n", "1,12,10\n19,3,4\n"])
def test_extension_needs_real_loss_improvement_through_initial_epoch20(development_extension, csv_rows):
    evidence, save, checkpoint, metadata, metrics = development_extension
    metrics.write_text("epoch,train/pose_loss,val/pose_loss\n" + csv_rows, encoding="utf-8")
    evidence["metrics_csv"]["sha256"] = training.sha256_file(metrics)
    with pytest.raises(ValueError):
        training.validate_development_extension(save(), checkpoint, metadata, 40)


def test_reviewed_resume_overrides_total_and_output_before_runtime_initialization(tmp_path, monkeypatch):
    original_output = tmp_path / "immutable_prior_run" / "training"
    checkpoint = original_output / "weights" / "development_epoch20.pt"

    class FakePoseTrainer:
        def check_resume(self, overrides):
            self.args = SimpleNamespace(epochs=20, project=str(original_output.parent), name="training",
                                        save_dir=str(original_output), model=str(checkpoint), resume=str(checkpoint))
            self.resume = True

    original_import = builtins.__import__

    def stub_pose_import(name, *args, **kwargs):
        if name == "ultralytics.models.yolo.pose":
            return SimpleNamespace(PoseTrainer=FakePoseTrainer)
        if name.split(".", 1)[0] in {"torch", "ultralytics"}:
            pytest.fail(f"Resume contract test imported an actual runtime: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_pose_import)
    fresh_project = tmp_path / "fresh_extension_run"
    trainer_type = training._development_resume_trainer({"epochs": 40, "project": str(fresh_project),
                                                        "name": "training", "batch": 1})
    trainer = trainer_type()
    trainer.check_resume({"resume": str(checkpoint)})
    assert trainer.args.epochs == 40
    assert Path(trainer.args.save_dir) == fresh_project / "training"
    assert trainer.args.model == str(checkpoint)
    assert trainer.args.resume == str(checkpoint)
    assert trainer.resume is True
