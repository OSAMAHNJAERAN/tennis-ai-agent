import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "artifacts" / "validation"
SCRIPT = ROOT / "scripts" / "replay_phase6_4_tracker_variants.py"
TRACKER = ROOT / "src" / "tracking" / "temporal_ball_tracker.py"
VIDEOS = ("video_08", "video_09", "video_10")


def load(name: str) -> dict:
    return json.loads((VALIDATION / name).read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_replay_manifest_uses_frozen_raw_candidate_files_and_hashes() -> None:
    manifest = load("phase6_4_tracker_replay_manifest.json")
    for video_id in VIDEOS:
        raw = VALIDATION / "raw_candidates" / f"{video_id}_candidates.json"
        assert manifest["provenance"]["raw_candidate_sha256"][video_id] == file_sha256(raw)
    assert manifest["provenance"]["videos"] == list(VIDEOS)
    assert manifest["provenance"]["pristine_video_11_plus_used"] is False


def test_replay_configs_are_distinguished_and_historical_does_not_fallback() -> None:
    manifest = load("phase6_4_tracker_replay_manifest.json")
    variants = manifest["variants"]
    assert variants["B_CURRENT_PRODUCTION_CONFIG"]["settings"] != variants["C_CURRENT_CLASS_DEFAULTS"]["settings"]
    assert variants["D_HISTORICAL_92_5_RECOVERY_CONFIG"]["status"] == "NOT_REPRODUCIBLE"
    assert variants["D_HISTORICAL_92_5_RECOVERY_CONFIG"]["settings"] is None


def test_every_executed_replay_is_deterministic() -> None:
    manifest = load("phase6_4_tracker_replay_manifest.json")
    for variant, by_video in manifest["trajectory_hashes"].items():
        for video_id, record in by_video.items():
            assert video_id in VIDEOS
            assert record["identical"] is True
            assert record["first_sha256"] == record["second_sha256"]


def test_gt_lineage_has_all_40_events_and_all_18_video10_events() -> None:
    lineage = load("phase6_4_tracker_replay_gt_lineage.json")
    assert lineage["record_count"] == 40
    assert len(lineage["records"]) == 40
    assert sum(row["video_id"] == "video_10" for row in lineage["records"]) == 18
    taxonomy = load("phase6_4_tracker_loss_taxonomy.json")
    assert len(taxonomy["video_10_forensic_records"]) == 18


def test_replay_declares_and_structurally_enforces_no_gt_tracking_input() -> None:
    manifest = load("phase6_4_tracker_replay_manifest.json")
    assert manifest["gt_loaded_after_all_trajectories"] is True
    assert manifest["tracker_received_ground_truth"] is False
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "track_video_candidates"
    ]
    assert calls
    for call in calls:
        rendered = ast.unparse(call)
        assert "gt" not in rendered.lower()
        assert "coverage" not in rendered.lower()


def test_tracker_has_no_video_identity_or_fixed_resolution_branch() -> None:
    source = TRACKER.read_text(encoding="utf-8")
    assert "video_08" not in source
    assert "video_09" not in source
    assert "video_10" not in source
    assert "video_11" not in source
    assert "frame_size" in source
    assert "enable_scale_norm" in source


def test_predicted_only_points_are_excluded_from_stage1() -> None:
    from scripts.reconstruct_phase6_4_evaluator_v2_1_artifacts import _stage_one_records
    from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint

    points = [
        TemporalBallPoint(0, 0.0, 10.0, 10.0, state=BallState.PREDICTED),
        TemporalBallPoint(1, 1 / 30, 11.0, 11.0, state=BallState.INTERPOLATED),
    ]
    records = _stage_one_records("video_test", points)
    assert [record["frame"] for record in records] == [1]


def test_interpolation_provenance_is_explicit_in_replays() -> None:
    found = False
    for path in (VALIDATION / "phase6_4_tracker_replay" / "e_reconciled_config").glob("*_trajectory.json"):
        points = json.loads(path.read_text(encoding="utf-8"))["points"]
        for point in points:
            if point["state"] == "INTERPOLATED":
                found = True
                assert point["source"] == "linear_interpolation"
                assert point["confidence"] is None
    assert found
