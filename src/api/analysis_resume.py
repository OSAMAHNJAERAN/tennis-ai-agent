"""Identity and artifact inventory for completed YOLO26 inference stages.

This deliberately resumes only post-inference finalization. It neither restores
tracker/filter state nor authorizes an interrupted inference stage to continue.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


REQUIRED_PIPELINE_FILES = (
    'source.mp4', 'media_metadata.json', 'inference_config.yaml',
    'annotated.mp4', 'player_motion.json', 'racket_tracking.json',
    'shot_events.json', 'rallies.json', 'point_analytics.json',
    'match_analytics.json', 'match_state.json', 'scoring_events.json',
    'serve_executions.json', 'shot_play_scope.json', 'score_history.json',
    'line_calls.json', 'match_events.json', 'event_candidates.json',
    'event_verification_trace.json', 'player_metrics.json', 'ball_metrics.json',
    'court_geometry.json', 'detections.json', 'temporal_spacing_audit.json',
    'temporal_detour_audit.json', 'trajectories.json', 'metrics.json',
    'run_config.yaml', 'point_timeline_manifest.json', 'camera_registration.json',
    'camera_intervals.json', 'match_identity.json', 'image_people.json',
    'stationary_candidate_audit.json', 'pixel_motion_audit.json',
    'patch_persistence_audit.json', 'native_rule_replay.json',
)
OPTIONAL_PIPELINE_FILES = (
    'source_frame_mapping.jsonl', 'normalization_frame_stats.txt',
    'landing_zones.json', 'media_source.json',
    'pose_tracking.json',
)


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _runtime(binaries):
    versions = {}
    for name in ('torch', 'ultralytics', 'numpy', 'opencv-python',
                 'opencv-python-headless', 'PyYAML', 'filelock'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    resolved = {name: Path(shutil.which(str(path)) or path).resolve() for name, path in binaries.items()}
    return {'python': platform.python_version(), 'python_executable_sha256': _hash(sys.executable),
            'package_versions': versions,
            'binaries': {name: {'path': str(path), 'sha256': _hash(path)}
                         for name, path in sorted(resolved.items())}}


def resume_identity(job, config, original, *, root, binaries):
    """Bind current inputs/dependencies without importing any inference runtime.

    The worker must first run its ordinary asset/qualification validation.
    Versions are recorded here; this is not a full installed-package source
    audit and is never itself a model-qualification result.
    """
    if config.get('model_system', {}).get('family') != 'yolo26':
        raise ValueError('Completed-stage checkpoints are scoped to the selected YOLO26 pipeline')
    root, original = Path(root).resolve(), Path(original).resolve()
    source_rows = []
    for path in sorted((root / 'src').rglob('*.py')):
        source_rows.append([path.relative_to(root).as_posix(), _hash(path)])
    if not source_rows:
        raise ValueError('Current pipeline source identity is missing')
    paths = [config['player_detection']['model'], config['ball_detection']['model'],
             config['court_detection']['model_path'], config['player_detection']['tracker'],
             config['model_system']['artifact_manifest'], job['config_path']]
    for section, field in (('camera_intervals', 'reviews_path'), ('match_identity', 'anchors_path')):
        if config.get(section, {}).get(field):
            paths.append(config[section][field])
    manifest_path = Path(config['model_system']['artifact_manifest'])
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Selected artifact manifest must remain an object')
    references = [manifest.get('qualification_receipt')]
    references.extend(row.get('training_provenance') for row in manifest.get('models', {}).values())
    for reference in references:
        if isinstance(reference, dict) and reference.get('path'):
            paths.append(reference['path'])
    dependencies = {}
    for value in paths:
        path = Path(value)
        path = (path if path.is_absolute() else root / path).resolve()
        if not path.is_file():
            raise ValueError('A configured checkpoint dependency is missing')
        dependencies[str(path)] = _hash(path)
    return {'schema': 'tennis.completed-inference-identity.v1',
            'analysis_id': job['id'], 'owner_id': job.get('owner_id', 'local'),
            'original_sha256': _hash(original), 'original_bytes': original.stat().st_size,
            'effective_config_sha256': hashlib.sha256(_canonical(config)).hexdigest(),
            'source_files': len(source_rows),
            'source_tree_sha256': hashlib.sha256(_canonical(source_rows)).hexdigest(),
            'dependency_files': dependencies, 'runtime': _runtime(binaries)}
