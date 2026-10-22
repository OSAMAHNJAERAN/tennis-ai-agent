"""Actual dependency/source identity, without model/runtime imports."""
import json
from pathlib import Path

import pytest

from src.api import analysis_resume as module


@pytest.fixture
def identity_inputs(tmp_path, monkeypatch):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/worker.py').write_bytes(b'original pipeline code')
    for name in ('ball.pt', 'players.pt', 'court.pt', 'tracker.yaml', 'config.yaml', 'review.jsonl'):
        (tmp_path / name).write_bytes(b'synthetic fixture only')
    (tmp_path / 'manifest.json').write_text(json.dumps({'models': {}, 'qualification_receipt': None}))
    original = tmp_path / 'original.mp4'
    original.write_bytes(b'not a real video')
    job = {'id': 'match_synthetic', 'owner_id': 'owner_a', 'config_path': str(tmp_path / 'config.yaml')}
    config = {'model_system': {'family': 'yolo26', 'artifact_manifest': 'manifest.json'},
              'player_detection': {'model': 'players.pt', 'tracker': 'tracker.yaml'},
              'ball_detection': {'model': 'ball.pt'}, 'court_detection': {'model_path': 'court.pt'},
              'camera_intervals': {'reviews_path': 'review.jsonl'}}
    monkeypatch.setattr(module, '_runtime', lambda binaries: {'synthetic_runtime_fingerprint': 1})
    def build():
        return module.resume_identity(job, config, original, root=tmp_path, binaries={})
    return tmp_path, job, config, build


@pytest.mark.parametrize('changed', ['original.mp4', 'ball.pt', 'players.pt', 'court.pt', 'tracker.yaml',
                                    'manifest.json', 'config.yaml', 'review.jsonl', 'src/worker.py'])
def test_current_dependency_or_source_changes_invalidate_identity(identity_inputs, changed):
    root, job, config, build = identity_inputs
    before = build()
    if changed == 'manifest.json':
        (root / changed).write_text('{"models":{},"changed":true}')
    else:
        with (root / changed).open('ab') as stream: stream.write(b'changed')
    assert build() != before


def test_effective_config_participant_or_owner_changes_invalidate_identity(identity_inputs):
    root, job, config, build = identity_inputs
    before = build()
    job['owner_id'] = 'owner_b'
    assert build() != before
    job['owner_id'] = 'owner_a'
    config['match_configuration'] = {'match_id': 'match_changed'}
    assert build() != before


def test_adding_source_file_invalidates_identity(identity_inputs):
    root, job, config, build = identity_inputs
    before = build()
    (root / 'src/new.py').write_bytes(b'new code')
    assert build() != before


def test_missing_dependency_and_nonfinite_config_cannot_make_identity(identity_inputs):
    root, job, config, build = identity_inputs
    config['nonfinite'] = float('nan')
    with pytest.raises(ValueError): build()
    config.pop('nonfinite')
    (root / 'ball.pt').unlink()
    with pytest.raises(ValueError, match='missing'): build()


def test_registered_qualification_and_training_references_are_bound(identity_inputs):
    root, job, config, build = identity_inputs
    (root / 'qualification.json').write_bytes(b'no actual qualification')
    (root / 'training.json').write_bytes(b'no actual training')
    (root / 'manifest.json').write_text(json.dumps({'qualification_receipt': {'path': 'qualification.json'},
        'models': {'ball': {'training_provenance': {'path': 'training.json'}}}}))
    before = build()
    assert str(root / 'qualification.json') in before['dependency_files']
    assert str(root / 'training.json') in before['dependency_files']
    (root / 'qualification.json').write_bytes(b'changed referenced receipt')
    assert build() != before
