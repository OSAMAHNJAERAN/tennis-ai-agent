"""Fail-closed configuration and hash checks for the selected YOLO26 bundle."""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import yaml

from src.detection.yolo26_adapters import YOLO26BallDetector, YOLO26CourtPoseDetector, YOLO26PlayerPoseDetector
from src.evaluation.yolo26_qualification import validate_qualification_receipt


REPO_ROOT = Path(__file__).resolve().parents[2]
PINNED_ULTRALYTICS = "8.4.36"
ROLE_SPEC = {"ball": ("yolo26m", "detect", None), "players": ("yolo26s", "pose", [17, 3]),
             "court": ("yolo26s", "pose", [14, 3])}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_yolo26_config(config, *, check_assets=False, require_qualification=True, root=REPO_ROOT):
    if type(require_qualification) is not bool:
        raise TypeError('require_qualification must be an explicit boolean')
    system = config.get("model_system", {})
    if system.get("family") != "yolo26" or system.get("vision_model_count") != 3:
        raise ValueError("Select exactly three specialized YOLO26 checkpoints")
    if system.get("ultralytics_version") != PINNED_ULTRALYTICS:
        raise ValueError(f"Runtime was verified with Ultralytics {PINNED_ULTRALYTICS}")
    if system.get("inference_head") != "one_to_one" or system.get("nms") is not False or system.get("end2end") is not True:
        raise ValueError("Pin one_to_one, end2end: true and nms: false for this runtime")
    if config.get("ball_detection", {}).get("backend") != "yolo26":
        raise ValueError("YOLO26 production cannot fall back to a historical ball backend")
    pose = config.get("pose_estimation", {})
    if pose.get("source") != "player_model" or pose.get("model_path") not in (None, config["player_detection"]["model"]):
        raise ValueError("Player boxes and pose must use the same checkpoint")
    if config.get("racket_tracking", {}).get("enabled", False):
        raise ValueError("A fourth racket model is outside the selected production scope")
    ball = config["ball_detection"]
    if any(ball.get(key) for key in ("ensemble_checkpoint", "verifier_model", "candidate_verifier")):
        raise ValueError("No learned ball ensemble or verifier is allowed")
    tracker_path = Path(config["player_detection"]["tracker"])
    tracker_path = tracker_path if tracker_path.is_absolute() else Path(root) / tracker_path
    tracker = yaml.safe_load(tracker_path.read_text(encoding="utf-8"))
    if tracker.get("tracker_type") != "botsort" or tracker.get("with_reid") is not False:
        raise ValueError("The pinned player tracker must be BoT-SORT with with_reid: false")
    manifest_path = Path(system["artifact_manifest"])
    manifest_path = manifest_path if manifest_path.is_absolute() else Path(root) / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = manifest.get("models", {})
    if set(roles) != set(ROLE_SPEC):
        raise ValueError("Artifact manifest must contain exactly ball, players and court")
    expected_paths = {"ball": ball["model"], "players": config["player_detection"]["model"],
                      "court": config["court_detection"]["model_path"]}
    if len(set(expected_paths.values())) != 3:
        raise ValueError("Ball, human pose and court pose need separate specialized checkpoints")
    for role, (family, task, shape) in ROLE_SPEC.items():
        record = roles[role]
        if (record.get("family"), record.get("task"), record.get("kpt_shape")) != (family, task, shape):
            raise ValueError(f"Incorrect {role} model schema")
        if record.get("path") != expected_paths[role]:
            raise ValueError(f"Manifest and configuration disagree for {role}")
        if check_assets:
            path = Path(record["path"])
            path = path if path.is_absolute() else Path(root) / path
            if not path.is_file():
                raise FileNotFoundError(f"Selected {role} YOLO26 artifact is missing: {path}. No legacy fallback.")
            digest = record.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or sha256_file(path) != digest:
                raise ValueError(f"Missing or mismatched SHA256 for {role} artifact")
            if role in ("ball", "court") and record.get("training_provenance") is None:
                raise ValueError(f"Specialist {role} artifact needs training/data provenance")
    if check_assets:
        if require_qualification:
            if manifest.get('use_scope') == 'PRIVATE_OFFLINE_EVALUATION':
                raise ValueError('Private evaluation weights require independent tennis qualification before production')
            manifest = {**manifest, "verified_qualification": validate_qualification_receipt(manifest, config, root=root)}
        else:
            # Asset integrity alone is never a qualification result.
            manifest = {key: value for key, value in manifest.items() if key != 'verified_qualification'}
        if importlib.metadata.version("ultralytics") != PINNED_ULTRALYTICS:
            raise ValueError(f"Install the pinned Ultralytics {PINNED_ULTRALYTICS} runtime")
    return manifest


@dataclass
class YOLO26Bundle:
    court_detector: YOLO26CourtPoseDetector
    player_detector: YOLO26PlayerPoseDetector
    ball_detector: YOLO26BallDetector
    artifact_manifest: dict


def build_yolo26_bundle(config, *, root=REPO_ROOT):
    manifest = validate_yolo26_config(config, check_assets=True, root=root)
    return _construct_bundle(config, manifest, root)


def build_evaluation_bundle(config, *, root=REPO_ROOT):
    """Explicit private engineering inference; the worker never calls this API."""
    manifest = validate_evaluation_config(config, root=root)
    return _construct_bundle(config, manifest, root)


def validate_evaluation_config(config, *, root=REPO_ROOT):
    """Validate private assets without constructing or qualifying any model."""
    manifest = validate_yolo26_config(config, check_assets=True, require_qualification=False, root=root)
    if (manifest.get('use_scope') != 'PRIVATE_OFFLINE_EVALUATION'
            or manifest.get('evaluation_only') is not True
            or manifest.get('public_redistribution_allowed') is not False
            or not str(manifest.get('status', '')).startswith('UNQUALIFIED')
            or manifest.get('qualification_receipt') is not None):
        raise ValueError('Evaluation requires an explicit unqualified private-only artifact manifest')
    if config.get('event_detection', {}).get('authoritative_enabled') is not False:
        raise ValueError('Private evaluation must withhold event, scoring and metric authority')
    for role, record in manifest['models'].items():
        provenance = record.get('training_provenance') if role in {'ball', 'court'} else record.get('origin_provenance')
        if not isinstance(provenance, dict) or not isinstance(provenance.get('path'), str):
            raise ValueError(f'Private {role} weights require retained provenance path and SHA256')
        evidence_path = Path(provenance['path'])
        evidence_path = evidence_path if evidence_path.is_absolute() else Path(root) / evidence_path
        if not evidence_path.is_file() or sha256_file(evidence_path) != provenance.get('sha256'):
            raise ValueError(f'Private {role} provenance SHA256 mismatch')
        if record.get('independent_accuracy_validated') is not False:
            raise ValueError('Evaluation manifest cannot assert independent model accuracy')
        evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
        weight_path = Path(record['path'])
        weight_path = weight_path if weight_path.is_absolute() else Path(root) / weight_path
        if role in {'ball', 'court'}:
            if (evidence.get('role') != role or evidence.get('status') != 'TRAINED_UNQUALIFIED'
                    or evidence.get('data_audit', {}).get('final_test_used') is not False):
                raise ValueError(f'Private {role} provenance must identify completed unqualified development training')
            retained = evidence.get('checkpoints', [])
            key = 'path'
        else:
            retained, key = evidence.get('artifacts', []), 'file'
        def same_artifact(item):
            item_path = Path(item.get(key, ''))
            item_path = item_path if item_path.is_absolute() else Path(root) / item_path
            return item_path.resolve() == weight_path.resolve() and item.get('sha256') == record['sha256']
        if not any(same_artifact(item) for item in retained):
            raise ValueError(f'Private {role} checkpoint is not bound to its retained provenance')
    return manifest


def _construct_bundle(config, manifest, root):
    def path(value):
        return str(Path(value) if Path(value).is_absolute() else Path(root) / value)
    player, ball, court = config["player_detection"], config["ball_detection"], config["court_detection"]
    device = config["model_system"].get("device", "auto")
    for role, record in manifest['models'].items():
        print(f"YOLO26 artifact: role={role}, path={path(record['path'])}, sha256={record['sha256']}, "
              f"scope={manifest.get('use_scope', 'PRODUCTION_QUALIFICATION_REQUIRED')}")
    player_detector = YOLO26PlayerPoseDetector(
        path(player["model"]), confidence=player["confidence"], imgsz=player["imgsz"],
        tracker=path(player["tracker"]), min_pose_confidence=config["pose_estimation"]["min_pose_confidence"], device=device)
    ball_detector = YOLO26BallDetector(path(ball["model"]), imgsz=ball["imgsz"], high_conf=ball["high_conf"],
                                      low_conf=ball["low_conf"], class_id=ball["class_id"], tiles=ball["tiles"], device=device)
    if list(ball_detector.model.names.values()) != ["tennis-ball"]:
        raise ValueError("Specialist ball checkpoint must have the single tennis-ball class")
    court_detector = YOLO26CourtPoseDetector(path(court["model_path"]), imgsz=court["imgsz"],
                                            confidence=court["confidence"], keypoint_confidence=court["keypoint_confidence"], device=device)
    if list(court_detector.model.names.values()) != ["court"]:
        raise ValueError("Specialist court checkpoint must have the single court class")
    return YOLO26Bundle(court_detector, player_detector, ball_detector, manifest)
