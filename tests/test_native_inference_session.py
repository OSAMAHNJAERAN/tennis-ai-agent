"""Actual retained synthetic assets/prefix/journal; no model or video inference."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import stat
from types import SimpleNamespace

import numpy as np
import pytest

from src.utils import native_inference_session as module
from src.utils.native_inference_session import (SOURCE_SEAL_NAME, NORMALIZED_FILES,
    seal_normalized_source, load_normalized_source, inference_identity,
    ball_journal_identity, validate_partial_inference)
from src.utils.ball_inference_journal import BallInferenceJournal, JOURNAL_NAME
from src.utils.native_inference_prefix import SEAL_NAME, DATABASE_NAME
from src.tracking.temporal_ball_tracker import BallObservation
from tests.test_native_inference_prefix import audits, write as write_prefix


BASE = {'analysis_id': 'synthetic-session', 'owner_id': 'owner-a', 'runtime': {'revision': 1},
        'source_sha256': 'a' * 64, 'config': {'qualified': False}}
BALL_CONFIG = {'backend': 'yolo26', 'imgsz': 1280, 'tiles': {'enabled': True, 'size_px': 960}}


def normalized(directory, *, mapped=3, optional=True):
    (directory / 'source.mp4').write_bytes(b'SYNTHETIC_BYTES_NOT_DECODABLE_VIDEO')
    metadata = {'scope': 'SYNTHETIC_SESSION_STORAGE_ONLY'}
    if mapped is not None:
        metadata['timeMapping'] = {'frames': mapped}
    (directory / 'media_metadata.json').write_text(json.dumps(metadata))
    (directory / 'inference_config.yaml').write_text('scope: SYNTHETIC_NO_INFERENCE\n')
    if optional:
        (directory / 'source_frame_mapping.jsonl').write_text(''.join(
            json.dumps({'frame_index': index, 'source_pts': str(index * 1001)}) + '\n' for index in range(3)))
        (directory / 'normalization_frame_stats.txt').write_text('SYNTHETIC_STATS\n')


def session(directory, *, prefix=True, committed=None, mapped=3, optional=True):
    normalized(directory, mapped=mapped, optional=optional)
    source = seal_normalized_source(directory, BASE)
    derived = inference_identity(source)
    if prefix:
        audits(directory)
        write_prefix(directory, identity=derived)
    if committed is not None:
        identity = ball_journal_identity(directory, derived, BALL_CONFIG, 'cuda:2')
        with BallInferenceJournal.create(directory, identity, 3) as journal:
            for index in range(committed):
                journal.append(index, [] if index % 2 == 0 else [BallObservation(index + .125, .375, .9)])
    return source, derived


def test_source_seal_is_exclusive_detached_and_binds_every_normalized_asset(tmp_path):
    normalized(tmp_path)
    source = seal_normalized_source(tmp_path, BASE)
    before = (tmp_path / SOURCE_SEAL_NAME).read_bytes()
    assert set(source['files']) == set(NORMALIZED_FILES)
    assert all(record['present'] for record in source['files'].values())
    first = inference_identity(source)
    assert 'timestamp' not in json.dumps(first).lower()
    source['base_identity']['owner_id'] = 'caller mutation'
    first['normalized_files']['source.mp4']['sha256'] = '0' * 64
    loaded = load_normalized_source(tmp_path, BASE)
    assert loaded['base_identity'] == BASE
    assert inference_identity(loaded) == inference_identity(load_normalized_source(tmp_path, deepcopy(BASE)))
    with pytest.raises(FileExistsError, match='retained seal'):
        seal_normalized_source(tmp_path, BASE)
    assert (tmp_path / SOURCE_SEAL_NAME).read_bytes() == before
    assert list(tmp_path.glob('.native_inference_source.*.tmp'))  # Retained exact staging link.


def test_absent_source_or_unsealed_player_prefix_is_not_reusable(tmp_path):
    assert load_normalized_source(tmp_path, BASE) is None
    assert validate_partial_inference(tmp_path, BASE, BALL_CONFIG) is None
    session(tmp_path, prefix=False)
    (tmp_path / DATABASE_NAME).write_bytes(b'unsealed player leftovers')
    assert validate_partial_inference(tmp_path, BASE, BALL_CONFIG) is None
    assert not (tmp_path / JOURNAL_NAME).exists()


@pytest.mark.parametrize('committed', [None, 0, 1, 2, 3])
@pytest.mark.parametrize('mapped', [None, 3])
def test_actual_prefix_and_raw_journal_validate_zero_partial_or_complete_frontier(tmp_path, committed, mapped):
    _, derived = session(tmp_path, committed=committed, mapped=mapped)
    before = {name: (tmp_path / name).read_bytes() for name in NORMALIZED_FILES}
    result = validate_partial_inference(tmp_path, BASE, BALL_CONFIG)
    assert result == {'inference_identity': derived, 'mapped_count': mapped,
        'next_frame': committed if committed is not None else 0, 'total_frames': 3, 'resolved_device': 'cuda:2'}
    assert (tmp_path / JOURNAL_NAME).exists() is (committed is not None)
    result['inference_identity']['base_identity']['owner_id'] = 'caller mutation'
    assert validate_partial_inference(tmp_path, BASE, BALL_CONFIG)['inference_identity'] == derived
    assert {name: (tmp_path / name).read_bytes() for name in NORMALIZED_FILES} == before


@pytest.mark.parametrize('name', NORMALIZED_FILES)
@pytest.mark.parametrize('operation', ['change', 'remove'])
def test_changed_or_missing_normalized_asset_rejects_existing_seal(tmp_path, name, operation):
    session(tmp_path)
    if operation == 'change':
        (tmp_path / name).write_bytes(b'changed normalized asset')
    else:
        (tmp_path / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_normalized_source(tmp_path, BASE)
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)


def test_absent_optional_presence_is_explicit_and_later_addition_is_rejected(tmp_path):
    source, _ = session(tmp_path, optional=False)
    assert source['files']['source_frame_mapping.jsonl'] == {'present': False}
    assert source['files']['normalization_frame_stats.txt'] == {'present': False}
    assert validate_partial_inference(tmp_path, BASE, BALL_CONFIG)['next_frame'] == 0
    (tmp_path / 'source_frame_mapping.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='file bindings'):
        load_normalized_source(tmp_path, BASE)


@pytest.mark.parametrize('identity', [{'analysis_id': 'foreign'},
    {**BASE, 'runtime': {'revision': True}}, {**BASE, 'runtime': {'revision': 1.}}])
def test_changed_base_identity_rejects_sealed_source(tmp_path, identity):
    session(tmp_path)
    with pytest.raises(ValueError, match='base identity'):
        validate_partial_inference(tmp_path, identity, BALL_CONFIG)


@pytest.mark.parametrize('identity', [{}, [], {'value': (1, 2)}, {1: 'key'},
    {'value': float('nan')}, {'value': float('inf')}, {'value': np.int64(1)}])
def test_nonplain_base_identity_is_rejected_without_a_seal(tmp_path, identity):
    normalized(tmp_path)
    with pytest.raises(ValueError):
        seal_normalized_source(tmp_path, identity)
    assert not (tmp_path / SOURCE_SEAL_NAME).exists()


def test_oversized_identity_and_circular_json_are_rejected(tmp_path):
    circular = {}
    circular['loop'] = circular
    for identity in (circular, {'value': 'x' * module.MAX_IDENTITY_BYTES}):
        with pytest.raises(ValueError):
            seal_normalized_source(tmp_path, identity)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('payload', [b'{', b'[]', b'{} trailing', b'{"x":1,"x":2}', b'{"x":NaN}',
    b'\xff', b' ' * (module.MAX_SOURCE_SEAL_BYTES + 1)],
    ids=['truncated', 'wrong_root', 'trailing', 'duplicate_field', 'nonfinite', 'invalid_utf8', 'oversized'])
def test_present_malformed_source_marker_raises_instead_of_becoming_absent(tmp_path, payload):
    (tmp_path / SOURCE_SEAL_NAME).write_bytes(payload)
    with pytest.raises(ValueError):
        load_normalized_source(tmp_path, BASE)


@pytest.mark.parametrize('mutation', ['bool_schema', 'stage', 'extra', 'missing_file',
                                    'bool_bytes', 'bad_hash', 'required_absent', 'optional_extra'])
def test_source_seal_schema_and_bindings_are_strict(tmp_path, mutation):
    normalized(tmp_path, optional=False)
    source = seal_normalized_source(tmp_path, BASE)
    if mutation == 'bool_schema':
        source['schema_version'] = True
    elif mutation == 'stage':
        source['stage'] = 'PHASE6_COMPLETED'
    elif mutation == 'extra':
        source['created_at'] = 1
    elif mutation == 'missing_file':
        del source['files']['source.mp4']
    elif mutation == 'bool_bytes':
        source['files']['source.mp4']['bytes'] = True
    elif mutation == 'bad_hash':
        source['files']['source.mp4']['sha256'] = 'A' * 64
    elif mutation == 'required_absent':
        source['files']['source.mp4'] = {'present': False}
    else:
        source['files']['source_frame_mapping.jsonl']['bytes'] = 0
    (tmp_path / SOURCE_SEAL_NAME).write_text(json.dumps(source))
    with pytest.raises(ValueError):
        load_normalized_source(tmp_path, BASE)


def test_journal_identity_binds_exact_prefix_marker_bytes_config_and_device(tmp_path):
    _, derived = session(tmp_path)
    identity = ball_journal_identity(tmp_path, derived, BALL_CONFIG, 'cuda:2')
    assert identity['resolved_device'] == 'cuda:2'
    identity['ball_config']['imgsz'] = 32
    original = ball_journal_identity(tmp_path, derived, BALL_CONFIG, 'cuda:2')
    assert original['ball_config']['imgsz'] == 1280
    marker = tmp_path / SEAL_NAME
    with marker.open('ab') as stream:
        stream.write(b'\n')
    assert ball_journal_identity(tmp_path, derived, BALL_CONFIG, 'cuda:2') != original
    assert ball_journal_identity(tmp_path, derived, BALL_CONFIG, 'cpu') != original
    assert ball_journal_identity(tmp_path, derived, {**BALL_CONFIG, 'imgsz': 640}, 'cuda:2') != original


@pytest.mark.parametrize('device', ['auto', 'cuda', 'cuda:-1', 'cuda:01', '0', 0, True, None, ' cpu'])
def test_journal_identity_requires_an_explicit_canonical_device(tmp_path, device):
    _, derived = session(tmp_path)
    with pytest.raises(ValueError, match='explicit'):
        ball_journal_identity(tmp_path, derived, BALL_CONFIG, device)


@pytest.mark.parametrize('config', [{}, [], {'imgsz': np.int64(1280)}, {'conf': float('nan')}, {'tiles': (1, 2)}])
def test_ball_operation_config_requires_nonempty_finite_plain_json(tmp_path, config):
    _, derived = session(tmp_path)
    with pytest.raises(ValueError):
        ball_journal_identity(tmp_path, derived, config, 'cuda:2')
    with pytest.raises(ValueError):
        validate_partial_inference(tmp_path, BASE, config)


@pytest.mark.parametrize('change', ['prefix_bytes', 'config', 'device', 'payload_hash'])
def test_existing_journal_rejects_changed_prefix_operation_or_corrupt_raw_row(tmp_path, change):
    _, derived = session(tmp_path, committed=2)
    config = BALL_CONFIG
    if change == 'prefix_bytes':
        with (tmp_path / SEAL_NAME).open('ab') as stream:
            stream.write(b'\n')
    elif change == 'config':
        config = {**BALL_CONFIG, 'imgsz': 640}
    elif change == 'device':
        # Real valid prefix uses cuda:2. Retain the prior journal separately,
        # then create a valid cpu-bound journal under the fixed journal name.
        (tmp_path / JOURNAL_NAME).rename(tmp_path / 'retained_prior_ball.sqlite3')
        identity = ball_journal_identity(tmp_path, derived, config, 'cpu')
        with BallInferenceJournal.create(tmp_path, identity, 3) as journal:
            journal.append(0, [])
    else:
        with sqlite3.connect(tmp_path / JOURNAL_NAME) as connection:
            connection.execute("UPDATE raw_frames SET payload='[]' WHERE frame_index=1")
    with pytest.raises(ValueError):
        validate_partial_inference(tmp_path, BASE, config)


@pytest.mark.parametrize('corrupt', ['prefix_marker', 'prefix_database', 'journal'])
def test_corrupt_existing_partial_state_is_retained_and_never_overwritten(tmp_path, corrupt):
    session(tmp_path, committed=1)
    name = SEAL_NAME if corrupt == 'prefix_marker' else DATABASE_NAME if corrupt == 'prefix_database' else JOURNAL_NAME
    path = tmp_path / name
    path.write_bytes(b'corrupt existing retained state')
    before = path.read_bytes()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)
    assert path.read_bytes() == before


@pytest.mark.parametrize('frames', [True, 3., 0, -1, 2, '3'])
def test_saved_mapping_count_must_be_a_positive_integer_matching_the_prefix(tmp_path, frames):
    session(tmp_path, mapped=frames)
    with pytest.raises(ValueError, match='mapping|frames'):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)


def test_oversized_media_metadata_is_rejected_before_json_decode(tmp_path, monkeypatch):
    session(tmp_path)
    monkeypatch.setattr(module, 'MAX_MEDIA_METADATA_BYTES', 2)
    with pytest.raises(ValueError, match='size bound'):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)


def test_zero_byte_required_normalization_file_cannot_be_sealed(tmp_path):
    normalized(tmp_path)
    (tmp_path / 'inference_config.yaml').write_bytes(b'')
    with pytest.raises(ValueError, match='file binding'):
        seal_normalized_source(tmp_path, BASE)
    assert not (tmp_path / SOURCE_SEAL_NAME).exists()


def test_valid_prefix_with_orphaned_ball_sidecar_is_not_a_fresh_frontier(tmp_path):
    session(tmp_path)
    sidecar = tmp_path / (JOURNAL_NAME + '-journal')
    sidecar.write_bytes(b'orphaned retained state')
    with pytest.raises(ValueError, match='without its journal'):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)
    assert sidecar.read_bytes() == b'orphaned retained state'


@pytest.mark.parametrize('where', ['directory', 'source', 'source_marker', 'prefix_marker', 'journal'])
@pytest.mark.parametrize('kind', ['symlink', 'reparse'])
def test_redirected_artifacts_or_ancestor_are_rejected(tmp_path, monkeypatch, where, kind):
    session(tmp_path, committed=1)
    selected = tmp_path if where == 'directory' else tmp_path / {
        'source': 'source.mp4', 'source_marker': SOURCE_SEAL_NAME,
        'prefix_marker': SEAL_NAME, 'journal': JOURNAL_NAME}[where]
    native_lstat = Path.lstat

    def redirected(path):
        value = native_lstat(path)
        if path == selected:
            return SimpleNamespace(st_mode=stat.S_IFLNK if kind == 'symlink' else value.st_mode,
                                   st_file_attributes=0x400 if kind == 'reparse' else 0)
        return value

    monkeypatch.setattr(Path, 'lstat', redirected)
    with pytest.raises(ValueError, match='links|regular files|reparse'):
        validate_partial_inference(tmp_path, BASE, BALL_CONFIG)


def test_parent_traversal_is_rejected_before_any_state_changes(tmp_path):
    with pytest.raises(ValueError, match='parent traversal'):
        seal_normalized_source(tmp_path / 'nested' / '..', BASE)
    assert list(tmp_path.iterdir()) == []


def test_interrupted_marker_publication_retains_staging_and_no_valid_marker(tmp_path, monkeypatch):
    normalized(tmp_path)
    primary = KeyboardInterrupt('source seal publication interrupted')
    monkeypatch.setattr(module.os, 'link', lambda *args: (_ for _ in ()).throw(primary))
    with pytest.raises(KeyboardInterrupt) as caught:
        seal_normalized_source(tmp_path, BASE)
    assert caught.value is primary
    assert not (tmp_path / SOURCE_SEAL_NAME).exists()
    assert list(tmp_path.glob('.native_inference_source.*.tmp'))
    assert load_normalized_source(tmp_path, BASE) is None


def test_directory_sync_cleanup_preserves_primary_interruption(tmp_path, monkeypatch):
    primary = KeyboardInterrupt('source directory fsync interrupted')
    descriptor = 525252
    events = []
    monkeypatch.setattr(module, '_DIRECTORY_SYNC_SUPPORTED', True)
    monkeypatch.setattr(module.os, 'open', lambda *args: descriptor)

    def sync(value):
        assert value == descriptor
        events.append('fsync')
        raise primary

    def close(value):
        assert value == descriptor
        events.append('close')
        raise ValueError('source directory close failed')

    monkeypatch.setattr(module.os, 'fsync', sync)
    monkeypatch.setattr(module.os, 'close', close)
    with pytest.raises(KeyboardInterrupt) as caught:
        module._sync_directory(tmp_path)
    assert caught.value is primary and events == ['fsync', 'close']
    assert primary.__notes__ == ['Native inference directory cleanup also failed: source directory close failed']


def test_file_read_failure_and_close_failure_keep_original_identity():
    primary = KeyboardInterrupt('source file read interrupted')

    class Stream:
        def close(self):
            raise ValueError('source stream close failed')

    path = SimpleNamespace(open=lambda mode: Stream())
    with pytest.raises(KeyboardInterrupt) as caught:
        with module._stream(path, 'rb'):
            raise primary
    assert caught.value is primary
    assert primary.__notes__ == ['Native inference file cleanup also failed: source stream close failed']
