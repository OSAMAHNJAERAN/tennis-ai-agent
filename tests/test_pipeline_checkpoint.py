"""Pure filesystem evidence for completed-stage reuse, never model resume."""
import json
import os
from pathlib import Path

import pytest

from src.api import pipeline_checkpoint as checkpoints


IDENTITY = {'analysisId': 'match_test', 'sourceSha256': 'a' * 64,
            'config': {'mode': 'selected_yolo26', 'revision': 0, 'qualified': False}}
REQUIRED = ('source.mp4', 'inference_config.yaml')
OPTIONAL = ('source_frame_mapping.jsonl', 'landing_zones.json')


def files(directory):
    (directory / 'source.mp4').write_bytes(b'SYNTHETIC_TEST_BYTES_NOT_PLAYABLE_MEDIA')
    (directory / 'inference_config.yaml').write_text('model_system: synthetic_test\n', encoding='utf-8')
    return directory


def write(directory, identity=IDENTITY, **kwargs):
    return checkpoints.write_pipeline_checkpoint(directory, identity,
        required_files=kwargs.get('required_files', REQUIRED), optional_files=kwargs.get('optional_files', OPTIONAL))


def load(directory, identity=IDENTITY, **kwargs):
    return checkpoints.load_pipeline_checkpoint(directory, identity,
        required_files=kwargs.get('required_files', REQUIRED), optional_files=kwargs.get('optional_files', OPTIONAL))


def marker(directory):
    return directory / checkpoints.CHECKPOINT_NAME


def owned_temps(directory):
    return list(directory.glob('.pipeline_checkpoint.*.tmp'))


def test_absent_marker_is_the_only_no_checkpoint_result(tmp_path):
    assert load(tmp_path) is None
    marker(tmp_path).write_bytes(b'{')
    with pytest.raises(ValueError):
        load(tmp_path)


def test_complete_roundtrip_detaches_identity_and_covers_absent_optional_outputs(tmp_path):
    files(tmp_path)
    identity = json.loads(json.dumps(IDENTITY))
    document = write(tmp_path, identity)
    assert document['stage'] == 'PHASE6_COMPLETED'
    assert set(document['artifacts']) == set(REQUIRED + OPTIONAL)
    assert document['artifacts']['landing_zones.json'] == {'present': False}
    identity['config']['revision'] = 5
    assert load(tmp_path) == document
    document['identity']['config']['revision'] = 9
    assert load(tmp_path)['identity'] == IDENTITY
    assert owned_temps(tmp_path) == []


@pytest.mark.parametrize('operation', ['change_required', 'remove_required', 'add_optional', 'remove_optional', 'change_optional'])
def test_any_artifact_or_optional_presence_drift_rejects_resume(tmp_path, operation):
    files(tmp_path)
    optional = tmp_path / 'source_frame_mapping.jsonl'
    optional.write_bytes(b'{"frame_index":0}\n')
    write(tmp_path)
    if operation == 'change_required':
        path = tmp_path / 'source.mp4'
        old = path.stat()
        data = path.read_bytes()
        path.write_bytes(b'X' + data[1:])
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    elif operation == 'remove_required':
        (tmp_path / 'source.mp4').unlink()
    elif operation == 'add_optional':
        (tmp_path / 'landing_zones.json').write_bytes(b'{}')
    elif operation == 'remove_optional':
        optional.unlink()
    else:
        optional.write_bytes(b'{"frame_index":1}\n')
    with pytest.raises(ValueError):
        load(tmp_path)


def test_exact_identity_and_inventory_are_required(tmp_path):
    files(tmp_path)
    write(tmp_path)
    changed = json.loads(json.dumps(IDENTITY))
    changed['config']['qualified'] = 0  # Python equality must not alias false and zero.
    with pytest.raises(ValueError):
        load(tmp_path, changed)
    with pytest.raises(ValueError):
        load(tmp_path, optional_files=('source_frame_mapping.jsonl',))
    with pytest.raises(ValueError):
        load(tmp_path, required_files=REQUIRED + ('new.json',))
    assert load(tmp_path, required_files=tuple(reversed(REQUIRED)), optional_files=tuple(reversed(OPTIONAL)))


@pytest.mark.parametrize('identity', [{}, [], {'x': float('nan')}, {'x': float('inf')}, {1: 'x'}, {'x': (1, 2)}, {'x': Path('source.mp4')}])
def test_invalid_identity_is_rejected_before_creating_any_files(tmp_path, identity):
    with pytest.raises(ValueError):
        write(tmp_path, identity)
    assert list(tmp_path.iterdir()) == []


def test_circular_or_oversized_identity_is_rejected(tmp_path):
    circular = {}
    circular['self'] = circular
    for value in (circular, {'text': 'x' * checkpoints.MAX_CHECKPOINT_BYTES}):
        with pytest.raises(ValueError):
            write(tmp_path, value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('required,optional', [
    ((), ()), ('source.mp4', ()), (['../source.mp4'], ()), (['dir/file.json'], ()),
    (['C:source.mp4'], ()), (['CON.json'], ()), (['trailing.'], ()),
    (['source.mp4'], ['source.mp4']), (['source.mp4', 'source.mp4'], ()),
    (['source.mp4'], ['SOURCE.mp4']),
    (['source.mp4'], [checkpoints.CHECKPOINT_NAME]),
])
def test_unsafe_or_ambiguous_inventories_are_rejected(tmp_path, required, optional):
    with pytest.raises(ValueError):
        write(tmp_path, required_files=required, optional_files=optional)
    assert list(tmp_path.iterdir()) == []


def test_parent_traversal_and_non_file_artifacts_are_rejected(tmp_path):
    files(tmp_path)
    with pytest.raises(ValueError):
        write(tmp_path / '..' / tmp_path.name)
    (tmp_path / 'source.mp4').unlink()
    (tmp_path / 'source.mp4').mkdir()
    with pytest.raises(ValueError):
        write(tmp_path)
    assert not marker(tmp_path).exists()


def test_symlink_artifact_and_marker_are_rejected(tmp_path):
    files(tmp_path)
    source = tmp_path / 'source.mp4'
    moved = tmp_path / 'real_source.mp4'
    source.rename(moved)
    try:
        source.symlink_to(moved)
    except OSError:
        pytest.skip('The current Windows account cannot create symbolic links')
    with pytest.raises(ValueError):
        write(tmp_path)
    source.unlink()
    moved.rename(source)
    write(tmp_path)
    real_marker = tmp_path / 'real_marker.json'
    marker(tmp_path).rename(real_marker)
    marker(tmp_path).symlink_to(real_marker)
    with pytest.raises(ValueError):
        load(tmp_path)


@pytest.mark.parametrize('payload', [b'{', b'{} trailing', b'[]', b'{"x":1,"x":2}',
    b'{"schemaVersion":1,"schemaVersion":1}', b'{"x":NaN}', b'\xff', b' ' * (checkpoints.MAX_CHECKPOINT_BYTES + 1)],
    ids=['truncated', 'trailing', 'array', 'duplicate', 'duplicate_schema', 'nan', 'utf8', 'oversized'])
def test_malformed_or_unbounded_marker_is_rejected(tmp_path, payload):
    marker(tmp_path).write_bytes(payload)
    with pytest.raises(ValueError):
        load(tmp_path)


@pytest.mark.parametrize('change', ['schema_bool', 'stage', 'extra', 'missing', 'bytes_bool', 'bad_digest', 'required_absent', 'optional_record_extra'])
def test_marker_schema_and_artifact_records_are_exact(tmp_path, change):
    files(tmp_path)
    document = write(tmp_path)
    if change == 'schema_bool':
        document['schemaVersion'] = True
    elif change == 'stage':
        document['stage'] = 'PARTIAL'
    elif change == 'extra':
        document['resumeFrame'] = 10
    elif change == 'missing':
        del document['artifacts']['source.mp4']
    elif change == 'bytes_bool':
        document['artifacts']['source.mp4']['bytes'] = True
    elif change == 'bad_digest':
        document['artifacts']['source.mp4']['sha256'] = 'A' * 64
    elif change == 'required_absent':
        document['artifacts']['source.mp4'] = {'present': False}
    else:
        document['artifacts']['landing_zones.json']['bytes'] = 0
    marker(tmp_path).write_text(json.dumps(document), encoding='utf-8')
    with pytest.raises(ValueError):
        load(tmp_path)


def test_committed_marker_is_never_overwritten_or_deleted(tmp_path, monkeypatch):
    files(tmp_path)
    write(tmp_path)
    previous = marker(tmp_path).read_bytes()
    monkeypatch.setattr(checkpoints, '_file_record', lambda *args, **kwargs: pytest.fail('Existing seal should be checked before artifacts'))
    with pytest.raises(FileExistsError):
        write(tmp_path)
    assert marker(tmp_path).read_bytes() == previous
    assert owned_temps(tmp_path) == []


@pytest.mark.parametrize('error', [OSError('disk full'), KeyboardInterrupt()])
def test_manifest_fsync_failure_or_cancellation_removes_only_its_owned_temp(tmp_path, monkeypatch, error):
    files(tmp_path)
    unrelated = tmp_path / '.pipeline_checkpoint.unrelated.tmp'
    unrelated.write_bytes(b'keep')
    actual_sync = checkpoints.os.fsync
    calls = 0

    def sync(descriptor):
        nonlocal calls
        calls += 1
        if calls == len(REQUIRED) + 1:
            raise error
        return actual_sync(descriptor)

    monkeypatch.setattr(checkpoints.os, 'fsync', sync)
    with pytest.raises(type(error)):
        write(tmp_path)
    assert not marker(tmp_path).exists()
    assert owned_temps(tmp_path) == [unrelated]
    assert unrelated.read_bytes() == b'keep'


@pytest.mark.parametrize('error', [OSError('publish denied'), KeyboardInterrupt()])
def test_atomic_publication_failure_or_cancellation_leaves_no_marker_or_owned_temp(tmp_path, monkeypatch, error):
    files(tmp_path)

    def publish(*args, **kwargs):
        raise error

    monkeypatch.setattr(checkpoints.os, 'link', publish)
    with pytest.raises(type(error)):
        write(tmp_path)
    assert not marker(tmp_path).exists()
    assert owned_temps(tmp_path) == []


def test_racing_writer_cannot_replace_the_other_committed_marker(tmp_path, monkeypatch):
    files(tmp_path)
    actual_link = checkpoints.os.link
    prior = b'other writer committed first\n'

    def competing_link(source, destination):
        destination.write_bytes(prior)
        actual_link(source, destination)

    monkeypatch.setattr(checkpoints.os, 'link', competing_link)
    with pytest.raises(FileExistsError):
        write(tmp_path)
    assert marker(tmp_path).read_bytes() == prior
    assert owned_temps(tmp_path) == []


def test_publication_lost_acknowledgement_preserves_a_complete_valid_marker(tmp_path, monkeypatch):
    files(tmp_path)
    actual_link = checkpoints.os.link

    def lost_acknowledgement(source, destination):
        actual_link(source, destination)
        raise KeyboardInterrupt()

    monkeypatch.setattr(checkpoints.os, 'link', lost_acknowledgement)
    with pytest.raises(KeyboardInterrupt):
        write(tmp_path)
    assert load(tmp_path)['identity'] == IDENTITY
    assert owned_temps(tmp_path) == []


def test_artifact_mutation_during_hashing_cannot_publish_a_checkpoint(tmp_path, monkeypatch):
    files(tmp_path)
    real_open = Path.open

    class MutatingReader:
        def __init__(self, stream, path):
            self.stream, self.path, self.changed = stream, path, False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            result = self.stream.read(count)
            if not self.changed:
                self.changed = True
                with real_open(self.path, 'ab') as writer:
                    writer.write(b'changed while hashing')
            return result

    def mutating_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return MutatingReader(stream, path) if path.name == 'source.mp4' and args[0] == 'r+b' else stream

    monkeypatch.setattr(Path, 'open', mutating_open)
    with pytest.raises(ValueError, match='changed during hashing'):
        write(tmp_path)
    assert not marker(tmp_path).exists()
    assert owned_temps(tmp_path) == []


def test_large_artifacts_use_bounded_reads_and_are_not_json_decoded(tmp_path, monkeypatch):
    files(tmp_path)
    (tmp_path / 'source.mp4').write_bytes(b'\x00' * (2 * checkpoints.HASH_CHUNK_BYTES + 17))
    real_open = Path.open
    read_sizes = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, count=-1):
            assert 0 < count <= checkpoints.HASH_CHUNK_BYTES
            read_sizes.append(count)
            return self.stream.read(count)

    def bounded_open(path, *args, **kwargs):
        stream = real_open(path, *args, **kwargs)
        return BoundedReader(stream) if path.name == 'source.mp4' and args and args[0] in ('rb', 'r+b') else stream

    monkeypatch.setattr(Path, 'open', bounded_open)
    document = write(tmp_path)
    assert document['artifacts']['source.mp4']['bytes'] == 2 * checkpoints.HASH_CHUNK_BYTES + 17
    assert load(tmp_path) == document
    assert len(read_sizes) == 8  # Three data chunks plus EOF for each of write/load.
