import json
from pathlib import Path

import pytest

from scripts.train.train_yolo26_tennis import audit_manifest, prepare_dataset, validate_labels, check_resource_guard, main


def test_labels_keep_court_visibility_and_reject_wrong_head(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("0 .5 .5 1 1 " + " ".join([".1 .2 2"] * 14))
    assert validate_labels(path, "court") == 1
    with pytest.raises(ValueError, match="56 finite fields"):
        validate_labels(path, "players")
    path.write_text("0 .5 .5 1 1 " + " ".join([".1 .2 .9"] * 14))
    with pytest.raises(ValueError, match="visibility"):
        validate_labels(path, "court")


def test_empty_manifest_cannot_start_training(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"scope": "elevated_end_court_singles", "frozen": True, "sources": [], "samples": []}))
    with pytest.raises(ValueError, match="No authorized labeled"):
        audit_manifest(path, "ball")


def test_missing_training_rights_cannot_be_overridden_by_publisher_license(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"scope": "elevated_end_court_singles", "frozen": True,
                               "sources": [{"license": "MIT", "training_allowed": False}],
                               "samples": [{"role": "ball"}]}))
    with pytest.raises(ValueError, match="explicit training rights"):
        audit_manifest(path, "ball")


def test_court_flip_mapping_matches_established_semantics(tmp_path):
    path = prepare_dataset([], "court", tmp_path)
    import yaml
    config = yaml.safe_load(path.read_text())
    assert config["kpt_shape"] == [14, 3]
    assert config["flip_idx"] == [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


def test_pilot_overfit_data_does_not_consume_final_test(tmp_path):
    path = prepare_dataset([{"split": "train", "image": "reviewed-training-frame.png"},
                            {"split": "test", "image": "secret-final-frame.png"}], "ball", tmp_path, pilot=True)
    import yaml
    data = yaml.safe_load(path.read_text())
    assert "test" not in data
    assert data["val"] == "train.txt"


def test_resource_guard_does_not_start_when_gpu_or_disk_is_too_full():
    resources = {"free_disk_bytes": 2 * 1024 ** 3, "free_ram_bytes": 1024 ** 3, "free_vram_bytes": 1024 ** 3}
    with pytest.raises(ValueError, match="2 GiB free VRAM"):
        check_resource_guard(resources, gpu_requested=True)
    check_resource_guard(resources, gpu_requested=False)
    resources["free_disk_bytes"] = 0
    with pytest.raises(ValueError, match="1 GiB free disk"):
        check_resource_guard(resources, gpu_requested=False)


def test_amp_cannot_silently_load_a_fourth_checkpoint(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--role", "ball", "--output", str(tmp_path), "--amp"])
    assert "extra YOLO26n" in capsys.readouterr().err


def test_measured_ball_pilot_requires_ram_and_vram_headroom():
    resources = {"free_disk_bytes": 2 * 1024 ** 3, "free_ram_bytes": 1024 ** 3, "free_vram_bytes": 5_500_000_000}
    with pytest.raises(ValueError, match="1.5 GiB available RAM"):
        check_resource_guard(resources, gpu_requested=True, role="ball")
    resources["free_ram_bytes"] = 2 * 1024 ** 3
    resources["free_vram_bytes"] = 4_000_000_000
    with pytest.raises(ValueError, match="5 GB free VRAM"):
        check_resource_guard(resources, gpu_requested=True, role="ball")
    resources["free_ram_bytes"] = 900 * 1024 ** 2
    with pytest.raises(ValueError, match="1 GiB available RAM"):
        check_resource_guard(resources, gpu_requested=True, role="ball", image_size=640)


def test_reviewed_pilot_is_bounded_to_four_epochs(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--role", "ball", "--output", str(tmp_path), "--pilot", "--smoke", "--epochs", "5"])
    assert "four" in capsys.readouterr().err
