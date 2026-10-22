"""Immutable publication uses synthetic bytes; no decoding or inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from threading import Barrier
import time

import pytest

from src.api import analysis_generation as generations
from src.api.result_storage import ResultStorage


def publish(directory, *, label='First', files=None, before_commit=None, revision=0,
            expected_generation=generations.AUTO, analysis_id='match_synthetic', result_extra=None):
    directory.mkdir(exist_ok=True)
    root = directory / '.result'
    root.mkdir(exist_ok=True)
    with ResultStorage(root) as storage:
        result = {'summary': {'id': analysis_id, 'frames': 129, 'label': label},
                  'records': storage.series({'frame': index, 'value': label} for index in range(129)),
                  'accuracyValidated': False}
        if result_extra is not None:
            result.update(result_extra)
        return generations.publish_analysis_generation(storage, result, directory,
            files={} if files is None else files, analysis_revision=revision, before_commit=before_commit,
            expected_generation=expected_generation)


def source(directory, name='working.mp4', data=b'SYNTHETIC_NOT_PLAYABLE_MEDIA'):
    path = directory / name
    path.write_bytes(data)
    return path


def current_bytes(directory):
    return (directory / '.result/current.store').read_bytes()


def assert_no_temps(directory):
    assert not list(directory.glob('.*.tmp'))
    assert not list((directory / '.result').glob('*.tmp'))


def test_full_generation_roundtrip_and_lazy_database_are_hash_bound(tmp_path):
    working = source(tmp_path)
    token = publish(tmp_path, files={'source.mp4': working, 'media_metadata.json': source(tmp_path, 'metadata.json', b'{}')})
    manifest_path = tmp_path / '.result' / (token + '.generation')
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == token
    assert json.loads(current_bytes(tmp_path)) == {'schemaVersion': 2, 'generation': token}
    snapshot = generations.resolve_generation(tmp_path)
    assert snapshot.generation == token and snapshot.analysis_id == 'match_synthetic'
    assert snapshot.analysis_revision == 0 and snapshot.frames == 129
    assert snapshot.resolve_file('source.mp4').read_bytes() == working.read_bytes()
    assert snapshot.file_sha256('source.mp4') == hashlib.sha256(working.read_bytes()).hexdigest()
    assert snapshot.resolve_file('absent.json') is None
    assert snapshot.file_sha256('absent.json') is None
    assert json.loads(snapshot.resolve_file('summary.json').read_bytes())['label'] == 'First'
    assert json.loads(snapshot.resolve_file('result.json').read_bytes())['records'][128]['frame'] == 128
    with ResultStorage.open(snapshot.database) as storage:
        result = storage.root()
        assert result['summary']['label'] == 'First' and result['records'][128]['frame'] == 128
    assert (tmp_path / 'result.json').read_bytes() == snapshot.resolve_file('result.json').read_bytes()
    assert (tmp_path / 'summary.json').read_bytes() == snapshot.resolve_file('summary.json').read_bytes()
    assert_no_temps(tmp_path)


def test_summary_resolution_does_not_hash_database_or_media(tmp_path, monkeypatch):
    token = publish(tmp_path, files={'source.mp4': source(tmp_path)})
    actual = generations._cached_hash
    inspected = []

    def selected(path, signature):
        inspected.append(Path(path).name)
        assert Path(path).name == 'summary.json'
        return actual(path, signature)

    monkeypatch.setattr(generations, '_cached_hash', selected)
    snapshot = generations.resolve_generation(tmp_path, token)
    assert snapshot.resolve_file('summary.json').is_file()
    assert inspected == ['summary.json']


def test_old_snapshots_and_historical_tokens_keep_original_files_after_working_copy_changes(tmp_path):
    working = source(tmp_path, data=b'first synthetic video')
    first_token = publish(tmp_path, files={'source.mp4': working})
    first = generations.resolve_generation(tmp_path)
    working.write_bytes(b'second synthetic video')
    assert first.resolve_file('source.mp4').read_bytes() == b'first synthetic video'
    second_token = publish(tmp_path, label='Second', files={'source.mp4': working}, revision=1)
    second = generations.resolve_generation(tmp_path)
    assert second.generation == second_token != first_token
    assert second.analysis_revision == 1
    assert first.resolve_file('summary.json').read_bytes() != second.resolve_file('summary.json').read_bytes()
    assert generations.resolve_generation(tmp_path, first_token).resolve_file('source.mp4').read_bytes() == b'first synthetic video'
    assert second.resolve_file('source.mp4').read_bytes() == b'second synthetic video'
    assert first.database.is_file() and second.database.is_file()


@pytest.mark.parametrize('name', ['source.mp4', 'summary.json', 'database'])
def test_same_size_corruption_is_rejected_even_with_restored_modification_time(tmp_path, name):
    publish(tmp_path, files={'source.mp4': source(tmp_path)})
    snapshot = generations.resolve_generation(tmp_path)
    path = snapshot.database if name == 'database' else snapshot.resolve_file(name)
    previous = path.stat()
    content = path.read_bytes()
    time.sleep(.003)  # NTFS change-time resolution; this is not a live-process wait.
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match='artifact bytes differ'):
        snapshot.database if name == 'database' else snapshot.resolve_file(name)


def test_repeated_large_file_validation_uses_bounded_hash_cache(tmp_path):
    publish(tmp_path, files={'source.mp4': source(tmp_path)})
    snapshot = generations.resolve_generation(tmp_path)
    generations._cached_hash.cache_clear()
    snapshot.resolve_file('source.mp4')
    first = generations._cached_hash.cache_info()
    snapshot.resolve_file('source.mp4')
    second = generations._cached_hash.cache_info()
    assert first.misses == second.misses == 1 and second.hits == first.hits + 1
    assert second.maxsize == 128


@pytest.mark.parametrize('error', [OSError('callback rejected'), KeyboardInterrupt()])
def test_callback_failure_preserves_previous_pointer_aliases_and_snapshot(tmp_path, error):
    first = publish(tmp_path)
    old_pointer = current_bytes(tmp_path)
    aliases = {name: (tmp_path / name).read_bytes() for name in ('result.json', 'summary.json')}

    def rejected():
        assert current_bytes(tmp_path) == old_pointer
        assert json.loads((tmp_path / 'summary.json').read_bytes())['label'] == 'Second'
        raise error

    with pytest.raises(type(error)):
        publish(tmp_path, label='Second', before_commit=rejected)
    assert current_bytes(tmp_path) == old_pointer
    assert all((tmp_path / name).read_bytes() == old for name, old in aliases.items())
    assert generations.resolve_generation(tmp_path).generation == first
    rejected = {path.stem for path in (tmp_path / '.result').glob('*.generation')} - {first}
    assert len(rejected) == 1
    with pytest.raises(ValueError, match='never committed'):
        generations.resolve_generation(tmp_path, rejected.pop())
    assert not (tmp_path / '.result' / (first + '.committed')).exists()
    assert_no_temps(tmp_path)


def test_callback_failure_on_first_publication_leaves_no_authoritative_result(tmp_path):
    def rejected():
        raise OSError('do not publish')

    with pytest.raises(OSError):
        publish(tmp_path, before_commit=rejected)
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)
    assert not (tmp_path / 'result.json').exists() and not (tmp_path / 'summary.json').exists()
    assert_no_temps(tmp_path)


@pytest.mark.parametrize('boundary', ['summary_alias', 'current_pointer'])
def test_atomic_replace_failure_rolls_back_aliases_and_keeps_previous_generation(tmp_path, monkeypatch, boundary):
    first = publish(tmp_path)
    old_pointer = current_bytes(tmp_path)
    aliases = {name: (tmp_path / name).read_bytes() for name in ('result.json', 'summary.json')}
    actual = generations.os.replace
    injected = False

    def replace(source, destination):
        nonlocal injected
        target = Path(destination)
        wanted = tmp_path / ('summary.json' if boundary == 'summary_alias' else '.result/current.store')
        if target == wanted and not injected:
            injected = True
            raise OSError('synthetic replacement failure')
        return actual(source, destination)

    monkeypatch.setattr(generations.os, 'replace', replace)
    with pytest.raises(OSError, match='replacement failure'):
        publish(tmp_path, label='Second')
    assert injected and current_bytes(tmp_path) == old_pointer
    assert generations.resolve_generation(tmp_path).generation == first
    assert all((tmp_path / name).read_bytes() == old for name, old in aliases.items())
    assert_no_temps(tmp_path)


def test_lost_pointer_acknowledgement_retains_complete_new_generation(tmp_path, monkeypatch):
    first = publish(tmp_path)
    actual = generations.os.replace

    def replace(source, destination):
        result = actual(source, destination)
        if Path(destination).name == 'current.store':
            raise KeyboardInterrupt('commit acknowledgement lost')
        return result

    monkeypatch.setattr(generations.os, 'replace', replace)
    with pytest.raises(KeyboardInterrupt):
        publish(tmp_path, label='Second')
    snapshot = generations.resolve_generation(tmp_path)
    assert snapshot.generation != first
    assert json.loads(snapshot.resolve_file('summary.json').read_bytes())['label'] == 'Second'
    assert json.loads((tmp_path / 'summary.json').read_bytes())['label'] == 'Second'
    assert generations.resolve_generation(tmp_path, first).database.is_file()
    assert_no_temps(tmp_path)


def test_two_concurrent_publishers_use_compare_and_swap_without_mixing_aliases(tmp_path, monkeypatch):
    first = publish(tmp_path)
    working = source(tmp_path)
    actual = generations._copy
    barrier = Barrier(2)

    def copy(source, destination, **kwargs):
        if destination.name == 'source.mp4':
            barrier.wait(timeout=10)
        return actual(source, destination, **kwargs)

    monkeypatch.setattr(generations, '_copy', copy)

    def compete(label):
        try:
            return label, publish(tmp_path, label=label, files={'source.mp4': working})
        except ValueError as error:
            assert 'current generation changed' in str(error)
            return label, None

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(compete, ('A', 'B')))
    winners = [(label, token) for label, token in outcomes if token is not None]
    assert len(winners) == 1
    label, token = winners[0]
    snapshot = generations.resolve_generation(tmp_path)
    assert snapshot.generation == token
    assert json.loads(snapshot.resolve_file('summary.json').read_bytes())['label'] == label
    assert json.loads((tmp_path / 'summary.json').read_bytes())['label'] == label
    rejected = {path.stem for path in (tmp_path / '.result').glob('*.generation')} - {first, token}
    assert len(rejected) == 1
    with pytest.raises(ValueError, match='never committed'):
        generations.resolve_generation(tmp_path, rejected.pop())
    assert_no_temps(tmp_path)


def test_copy_fsync_failure_keeps_existing_current_pointer(tmp_path, monkeypatch):
    first = publish(tmp_path)
    pointer = current_bytes(tmp_path)
    working = source(tmp_path)

    def fsync(descriptor):
        raise OSError('synthetic fsync failure')

    monkeypatch.setattr(generations.os, 'fsync', fsync)
    with pytest.raises(OSError, match='fsync failure'):
        publish(tmp_path, files={'source.mp4': working})
    assert current_bytes(tmp_path) == pointer
    assert generations.resolve_generation(tmp_path).generation == first


def test_database_publication_failure_preserves_previous_pointer_and_aliases(tmp_path, monkeypatch):
    first = publish(tmp_path)
    pointer = current_bytes(tmp_path)
    summary = (tmp_path / 'summary.json').read_bytes()

    def rejected(*args, **kwargs):
        raise OSError('database publication denied')

    monkeypatch.setattr(ResultStorage, 'publish', rejected)
    with pytest.raises(OSError, match='database publication'):
        publish(tmp_path, label='Uncommitted')
    assert current_bytes(tmp_path) == pointer and (tmp_path / 'summary.json').read_bytes() == summary
    assert generations.resolve_generation(tmp_path).generation == first


def test_source_mutation_during_copy_is_rejected_before_publication(tmp_path, monkeypatch):
    first = publish(tmp_path)
    working = source(tmp_path)
    actual_open = Path.open

    class MutatingReader:
        def __init__(self, stream): self.stream, self.changed = stream, False
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, count):
            value = self.stream.read(count)
            if not self.changed:
                self.changed = True
                with actual_open(working, 'ab') as writer:
                    writer.write(b'mutated during snapshot')
            return value

    def opened(path, *args, **kwargs):
        stream = actual_open(path, *args, **kwargs)
        return MutatingReader(stream) if path == working and args and args[0] == 'rb' else stream

    monkeypatch.setattr(Path, 'open', opened)
    with pytest.raises(ValueError, match='source changed'):
        publish(tmp_path, files={'source.mp4': working})
    assert generations.resolve_generation(tmp_path).generation == first


@pytest.mark.parametrize('files', [[], {'../escape.mp4': 'working.mp4'}, {'bad/file.mp4': 'working.mp4'},
    {'bad:stream': 'working.mp4'}, {'CON.txt': 'working.mp4'}, {'trailing.': 'working.mp4'},
    {'SOURCE.mp4': 'working.mp4', 'source.mp4': 'working.mp4'}, {'result.json': 'working.mp4'},
    {'SUMMARY.json': 'working.mp4'}], ids=['not_mapping', 'parent', 'slash', 'stream', 'device', 'trailing', 'case_collision', 'result_reserved', 'summary_reserved'])
def test_unsafe_filename_mappings_cannot_publish(tmp_path, files):
    source(tmp_path)
    with pytest.raises(ValueError):
        publish(tmp_path, files=files)
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)


def test_file_count_and_manifest_size_are_bounded(tmp_path, monkeypatch):
    working = source(tmp_path)
    with pytest.raises(ValueError, match='file count'):
        publish(tmp_path, files={f'file{index}.json': working for index in range(generations.MAX_FILES - 1)})
    monkeypatch.setattr(generations, 'MAX_MANIFEST_BYTES', 256)
    with pytest.raises(ValueError, match='manifest exceeds'):
        publish(tmp_path)
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)


def test_missing_input_file_cannot_publish(tmp_path):
    with pytest.raises(FileNotFoundError):
        publish(tmp_path, files={'source.mp4': tmp_path / 'missing.mp4'})
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)


def test_missing_marker_and_legacy_v1_pointer_return_none_without_migration(tmp_path):
    assert generations.resolve_generation(tmp_path) is None
    root = tmp_path / '.result'
    root.mkdir()
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)
    identifier = 'a' * 32
    with ResultStorage(root) as storage:
        storage.set_root({'summary': {'id': 'legacy', 'frames': 1}})
        database = storage.publish(root / (identifier + '.sqlite3'))
    pointer = {'schemaVersion': 1, 'database': database.name, 'databaseBytes': database.stat().st_size,
               'databaseSha256': hashlib.sha256(database.read_bytes()).hexdigest(), 'analysisId': 'legacy', 'frames': 1}
    marker = root / 'current.store'
    marker.write_text(json.dumps(pointer), encoding='utf-8')
    previous = marker.read_bytes()
    assert generations.resolve_generation(tmp_path) is None
    assert marker.read_bytes() == previous and database.is_file()
    token = publish(tmp_path)
    assert generations.resolve_generation(tmp_path).generation == token
    assert database.is_file()


def test_explicit_expected_generation_rejects_wrong_stale_and_absent_bases(tmp_path):
    first = publish(tmp_path, expected_generation=None)
    second = publish(tmp_path, label='Second', expected_generation=first)
    pointer = current_bytes(tmp_path)
    for expected in (first, None, 'f' * 64):
        with pytest.raises(ValueError, match='expected current'):
            publish(tmp_path, label='Stale', expected_generation=expected)
        assert current_bytes(tmp_path) == pointer
    assert generations.resolve_generation(tmp_path).generation == second


def test_revision_rollback_and_different_analysis_are_rejected(tmp_path):
    token = publish(tmp_path, revision=4)
    for kwargs in ({'revision': 3}, {'revision': 5, 'analysis_id': 'different_analysis'}):
        with pytest.raises(ValueError, match='identity changed or revision'):
            publish(tmp_path, **kwargs)
    assert generations.resolve_generation(tmp_path).generation == token
    assert publish(tmp_path, label='Equal revision retry', revision=4)


def test_retained_committed_generation_can_resolve_when_current_is_missing_or_corrupt(tmp_path):
    first = publish(tmp_path)
    second = publish(tmp_path, label='Second')
    receipt = tmp_path / '.result' / (first + '.committed')
    assert receipt.read_bytes() == generations._receipt_bytes(first)
    assert not (tmp_path / '.result' / (second + '.committed')).exists()
    current = tmp_path / '.result/current.store'
    current.unlink()
    assert generations.resolve_generation(tmp_path, first).generation == first
    with pytest.raises(ValueError, match='marker is missing'):
        generations.resolve_generation(tmp_path)
    current.write_bytes(b'{corrupt')
    assert generations.resolve_generation(tmp_path, first).generation == first
    with pytest.raises(ValueError):
        generations.resolve_generation(tmp_path)
    with pytest.raises(ValueError):
        generations.resolve_generation(tmp_path, second)


def test_corrupt_committed_receipt_fails_closed_and_is_never_overwritten(tmp_path):
    first = publish(tmp_path)
    second = publish(tmp_path, label='Second')
    old_receipt = tmp_path / '.result' / (first + '.committed')
    old_receipt.write_bytes(b'corrupt receipt')
    with pytest.raises(ValueError, match='receipt differs'):
        generations.resolve_generation(tmp_path, first)
    current_receipt = tmp_path / '.result' / (second + '.committed')
    current_receipt.write_bytes(b'preexisting receipt must not be replaced')
    pointer = current_bytes(tmp_path)
    summary = (tmp_path / 'summary.json').read_bytes()
    with pytest.raises(ValueError, match='receipt differs'):
        publish(tmp_path, label='Third')
    assert current_bytes(tmp_path) == pointer and (tmp_path / 'summary.json').read_bytes() == summary
    assert current_receipt.read_bytes() == b'preexisting receipt must not be replaced'
    assert old_receipt.read_bytes() == b'corrupt receipt'
    assert_no_temps(tmp_path)


@pytest.mark.parametrize('extra', [
    {'analysisRevision': 0}, {'analysisRevision': True}, {'analysisRevision': 1.0},
    {'pointAnalysis': {'analysisRevision': 0}}, {'pointAnalysis': {'analysisRevision': True}},
    {'pointAnalysis': {}}, {'analysisRevision': 1, 'pointAnalysis': {'analysisRevision': 2}},
], ids=['top_wrong', 'top_bool', 'top_float', 'point_wrong', 'point_bool', 'point_missing', 'point_disagrees'])
def test_present_raw_analysis_revisions_must_match_publication_revision(tmp_path, extra):
    with pytest.raises(ValueError, match='revision differs'):
        publish(tmp_path, revision=1, result_extra=extra)
    assert not (tmp_path / '.result/current.store').exists()


def test_matching_present_revisions_are_retained_without_injecting_absent_fields(tmp_path):
    plain_token = publish(tmp_path, revision=1)
    snapshot = generations.resolve_generation(tmp_path, plain_token)
    assert 'analysisRevision' not in json.loads(snapshot.resolve_file('result.json').read_bytes())
    token = publish(tmp_path, revision=2, result_extra={'analysisRevision': 2, 'pointAnalysis': {'analysisRevision': 2}})
    result = json.loads(generations.resolve_generation(tmp_path, token).resolve_file('result.json').read_bytes())
    assert result['analysisRevision'] == result['pointAnalysis']['analysisRevision'] == 2


@pytest.mark.parametrize('payload', [b'{', b'{} trailing', b'{"schemaVersion":2,"generation":"x"}',
    b'{"schemaVersion":2,"schemaVersion":2}', b'{"schemaVersion":true}', b'{"x":NaN}', b'\xff'])
def test_invalid_current_pointer_fails_closed(tmp_path, payload):
    root = tmp_path / '.result'
    root.mkdir()
    (root / 'current.store').write_bytes(payload)
    with pytest.raises(ValueError):
        generations.resolve_generation(tmp_path)
    with pytest.raises(ValueError):
        publish(tmp_path)
    assert (root / 'current.store').read_bytes() == payload


def test_changed_manifest_digest_wrong_tokens_and_correlated_paths_are_rejected(tmp_path):
    token = publish(tmp_path)
    path = tmp_path / '.result' / (token + '.generation')
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='manifest digest'):
        generations.resolve_generation(tmp_path)
    path.write_bytes(original)
    for invalid in ('../escape', 'A' * 64, 'a' * 63):
        with pytest.raises(ValueError):
            generations.resolve_generation(tmp_path, invalid)
    value = json.loads(original)
    value['filesDirectory'] = 'b' * 32 + '.files'
    changed = json.dumps(value).encode('utf-8')
    other = hashlib.sha256(changed).hexdigest()
    (path.parent / (other + '.generation')).write_bytes(changed)
    (path.parent / 'current.store').write_text(json.dumps({'schemaVersion': 2, 'generation': other}), encoding='utf-8')
    with pytest.raises(ValueError, match='do not correlate'):
        generations.resolve_generation(tmp_path)


def test_symlink_inputs_and_generation_paths_are_rejected(tmp_path):
    working = source(tmp_path)
    link = tmp_path / 'linked.mp4'
    try:
        link.symlink_to(working)
    except OSError:
        pytest.skip('Current Windows account cannot create symbolic links')
    with pytest.raises(ValueError, match='symbolic'):
        publish(tmp_path, files={'source.mp4': link})
    token = publish(tmp_path, files={'source.mp4': working})
    snapshot = generations.resolve_generation(tmp_path, token)
    path = snapshot.resolve_file('source.mp4')
    path.unlink()
    path.symlink_to(working)
    with pytest.raises(ValueError, match='symbolic'):
        snapshot.resolve_file('source.mp4')


def test_copies_use_bounded_reads_instead_of_linking_or_loading_media(tmp_path, monkeypatch):
    working = source(tmp_path, data=b'\x00' * (2 * generations.COPY_CHUNK_BYTES + 19))
    actual_open = Path.open
    reads = []

    class Reader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, count=-1):
            assert 0 < count <= generations.COPY_CHUNK_BYTES
            reads.append(count)
            return self.stream.read(count)

    def open_file(path, *args, **kwargs):
        stream = actual_open(path, *args, **kwargs)
        return Reader(stream) if path == working and args and args[0] == 'rb' else stream

    monkeypatch.setattr(Path, 'open', open_file)
    monkeypatch.setattr(os, 'link', lambda *args, **kwargs: pytest.fail('Media must never be hard-linked'))
    token = publish(tmp_path, files={'source.mp4': working})
    snapshot = generations.resolve_generation(tmp_path, token)
    assert snapshot.resolve_file('source.mp4').stat().st_size == 2 * generations.COPY_CHUNK_BYTES + 19
    assert len(reads) == 4
