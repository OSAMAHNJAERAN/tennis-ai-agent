"""Immutable, hash-bound analysis files committed by one small pointer.

Media are copied in bounded chunks, never linked to mutable worker files.
Windows file fsync and atomic replacement support process-crash recovery;
this module does not promise recovery from storage or power loss.
"""
from dataclasses import dataclass
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
import uuid

from filelock import FileLock


MAX_MANIFEST_BYTES = 64 * 1024
MAX_FILES = 256
COPY_CHUNK_BYTES = 1024 * 1024
AUTO = object()
_TOKEN = re.compile(r'[0-9a-f]{64}\Z')
_UUID = re.compile(r'[0-9a-f]{32}\Z')
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z')
_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'{base}{n}' for base in ('COM', 'LPT') for n in range(1, 10))}


def _require(condition, message):
    if not condition:
        raise ValueError('Analysis generation: ' + message)


def _name(value):
    _require(type(value) is str and _NAME.fullmatch(value) and not value.endswith('.')
             and value.split('.')[0].upper() not in _RESERVED, 'unsafe logical filename')
    return value


def _safe_path(path):
    path = Path(path)
    _require('..' not in path.parts, 'parent traversal is forbidden')
    path = path.absolute()
    for part in (path, *path.parents):
        _require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
                 'symbolic links and junctions are forbidden')
    return path


def _directory(path):
    path = _safe_path(path)
    _require(path.is_dir(), 'directory is missing or invalid')
    return path.resolve()


def _core(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _signature(details):
    return (*_core(details), details.st_ctime_ns)


def _path_stat(path):
    _safe_path(path)
    details = path.lstat()
    _require(stat.S_ISREG(details.st_mode), 'artifact is not a regular file: ' + path.name)
    return details


def _opened_signature(path):
    before = _path_stat(path)
    with path.open('rb') as stream:
        actual = os.fstat(stream.fileno())
        _require(_core(before) == _core(actual), 'artifact changed while opening: ' + path.name)
    _require(_core(_path_stat(path)) == _core(actual), 'artifact changed while closing: ' + path.name)
    # NTFS lstat can briefly report a cached ctime. The open handle's exact
    # ctime is used in cache keys; path/handle identity checks use stable fields.
    return _signature(actual)


@lru_cache(maxsize=128)
def _cached_hash(path_text, signature):
    path = Path(path_text)
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        _require(_signature(os.fstat(stream.fileno())) == signature, 'artifact changed before hashing')
        for block in iter(lambda: stream.read(COPY_CHUNK_BYTES), b''):
            digest.update(block)
            count += len(block)
        _require(_signature(os.fstat(stream.fileno())) == signature, 'artifact changed during hashing')
    _require(count == signature[2] and _core(_path_stat(path)) == signature[:4],
             'artifact changed after hashing')
    return digest.hexdigest()


def _file_record(path):
    signature = _opened_signature(path)
    digest = _cached_hash(str(path), signature)
    _require(_opened_signature(path) == signature, 'artifact changed while validating')
    return {'bytes': signature[2], 'sha256': digest}


def _validate_file(path, expected):
    _require(_file_record(path) == expected, 'artifact bytes differ: ' + path.name)
    return path


def _fsync_directory(path):
    if os.name != 'nt':
        descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _copy(source, destination, *, temporary=False):
    """Copy and hash one regular source while retaining only a 1 MiB chunk."""
    source = _safe_path(source)
    before = _path_stat(source)
    created = False
    digest = hashlib.sha256()
    count = 0
    try:
        with source.open('rb') as reader:
            opened = os.fstat(reader.fileno())
            _require(_core(opened) == _core(before), 'copy source changed while opening')
            with destination.open('xb') as writer:
                created = True
                for block in iter(lambda: reader.read(COPY_CHUNK_BYTES), b''):
                    writer.write(block)
                    digest.update(block)
                    count += len(block)
                writer.flush()
                os.fsync(writer.fileno())
            _require(_signature(os.fstat(reader.fileno())) == _signature(opened),
                     'copy source changed during reading')
        _require(_core(_path_stat(source)) == _core(opened) and count == opened.st_size,
                 'copy source changed after reading')
    except BaseException as error:
        if temporary and created:
            try:
                destination.unlink(missing_ok=True)
            except BaseException as cleanup:
                error.add_note('Generation copy temporary cleanup also failed: ' + str(cleanup))
        raise
    return {'bytes': count, 'sha256': digest.hexdigest()}


def _pairs(values):
    result = {}
    for key, value in values:
        _require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Analysis generation: nonfinite JSON constant ' + value)


def _json(payload):
    try:
        return json.loads(payload.decode('utf-8'), object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('Analysis generation: invalid JSON') from error


def _read_small(path):
    before = _opened_signature(path)
    _require(before[2] <= MAX_MANIFEST_BYTES, 'manifest exceeds bounded size')
    with path.open('rb') as stream:
        _require(_signature(os.fstat(stream.fileno())) == before, 'manifest changed while opening')
        payload = stream.read(MAX_MANIFEST_BYTES + 1)
        _require(_signature(os.fstat(stream.fileno())) == before, 'manifest changed during reading')
    _require(len(payload) <= MAX_MANIFEST_BYTES and _opened_signature(path) == before,
             'manifest changed or exceeds bounded size')
    return payload


def _metadata(value):
    _require(type(value) is dict and set(value) == {'bytes', 'sha256'}
             and type(value['bytes']) is int and value['bytes'] >= 0
             and type(value['sha256']) is str and _TOKEN.fullmatch(value['sha256']),
             'invalid artifact metadata')
    return dict(value)


def _current(generations):
    marker = generations / 'current.store'
    try:
        marker.lstat()
    except FileNotFoundError:
        return None, None
    raw = _read_small(marker)
    value = _json(raw)
    _require(type(value) is dict and type(value.get('schemaVersion')) is int, 'invalid current pointer')
    if value['schemaVersion'] == 2:
        _require(set(value) == {'schemaVersion', 'generation'} and type(value['generation']) is str
                 and _TOKEN.fullmatch(value['generation']), 'invalid generation pointer')
    else:
        expected = {'schemaVersion', 'database', 'databaseBytes', 'databaseSha256', 'analysisId', 'frames'}
        _require(value['schemaVersion'] == 1 and set(value) == expected
                 and type(value['database']) is str and value['database'].endswith('.sqlite3')
                 and _UUID.fullmatch(value['database'][:-8])
                 and type(value['databaseBytes']) is int and value['databaseBytes'] >= 0
                 and type(value['databaseSha256']) is str and _TOKEN.fullmatch(value['databaseSha256'])
                 and type(value['analysisId']) is str and bool(value['analysisId'])
                 and type(value['frames']) is int and value['frames'] > 0, 'invalid legacy generation pointer')
        _require(_path_stat(generations / value['database']).st_size == value['databaseBytes'],
                 'legacy generation database differs')
    return raw, value


@dataclass(frozen=True)
class GenerationSnapshot:
    generation: str
    analysis_id: str
    analysis_revision: int
    frames: int
    _root: Path
    _database_name: str
    _database_metadata: dict
    _files_name: str
    _files: object

    @property
    def database(self):
        return _validate_file(self._root / self._database_name, self._database_metadata)

    @property
    def files_directory(self):
        return _directory(self._root / self._files_name)

    def resolve_file(self, name):
        _name(name)
        expected = self._files.get(name)
        if expected is None:
            return None
        return _validate_file(self.files_directory / name, expected)

    def file_sha256(self, name):
        path = self.resolve_file(name)
        return None if path is None else self._files[name]['sha256']


def _snapshot(generations, token):
    _require(type(token) is str and _TOKEN.fullmatch(token), 'invalid generation token')
    raw = _read_small(generations / (token + '.generation'))
    _require(hashlib.sha256(raw).hexdigest() == token, 'generation manifest digest differs')
    manifest = _json(raw)
    expected = {'schemaVersion', 'analysisId', 'analysisRevision', 'frames', 'database', 'filesDirectory', 'files'}
    _require(type(manifest) is dict and set(manifest) == expected
             and type(manifest['schemaVersion']) is int and manifest['schemaVersion'] == 1
             and type(manifest['analysisId']) is str and bool(manifest['analysisId'])
             and type(manifest['analysisRevision']) is int and manifest['analysisRevision'] >= 0
             and type(manifest['frames']) is int and manifest['frames'] > 0, 'invalid generation manifest')
    database = manifest['database']
    _require(type(database) is dict and set(database) == {'name', 'bytes', 'sha256'}
             and type(database['name']) is str and database['name'].endswith('.sqlite3')
             and _UUID.fullmatch(database['name'][:-8]), 'invalid database name')
    database_metadata = _metadata({key: database[key] for key in ('bytes', 'sha256')})
    identifier = database['name'][:-8]
    _require(manifest['filesDirectory'] == identifier + '.files', 'database and asset directories do not correlate')
    files = manifest['files']
    _require(type(files) is dict and 2 <= len(files) <= MAX_FILES
             and {'result.json', 'summary.json'} <= set(files), 'invalid generation file inventory')
    names = [_name(name) for name in files]
    _require(len({name.casefold() for name in names}) == len(names), 'ambiguous filename inventory')
    metadata = {name: MappingProxyType(_metadata(value)) for name, value in files.items()}
    _directory(generations / manifest['filesDirectory'])
    _require(_path_stat(generations / database['name']).st_size == database_metadata['bytes'],
             'generation database is missing or its size differs')  # Full hash validation is deliberately lazy.
    return GenerationSnapshot(token, manifest['analysisId'], manifest['analysisRevision'], manifest['frames'],
        generations, database['name'], MappingProxyType(database_metadata), manifest['filesDirectory'], MappingProxyType(metadata))


def _receipt_bytes(token):
    return json.dumps({'schemaVersion': 1, 'generation': token}, separators=(',', ':')).encode('utf-8') + b'\n'


def _has_committed_receipt(generations, token):
    path = generations / (token + '.committed')
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    _require(_read_small(path) == _receipt_bytes(token), 'committed generation receipt differs')
    return True


def _retain_current_receipt(generations, token, identifier):
    """Remember an already committed current token before replacing it.

    Only a tiny newly created immutable receipt is atomically linked here;
    worker inputs/media are always independently copied. Exclusive creation
    preserves any existing receipt, including a corrupt one that must fail.
    """
    if _has_committed_receipt(generations, token):
        return
    temporary = generations / (identifier + '.committed.tmp')
    _write_bytes(temporary, _receipt_bytes(token), temporary=True)
    primary = None
    try:
        os.link(temporary, generations / (token + '.committed'))
        _fsync_directory(generations)
    except BaseException as error:
        primary = error
        raise
    finally:
        _cleanup((temporary,), primary)


def resolve_generation(directory, generation=None):
    """Pin a current or retained immutable generation; v1 remains caller-owned."""
    directory = _directory(directory)
    generations = _safe_path(directory / '.result')
    if not generations.exists():
        _require(generation is None, 'requested generation directory is absent')
        return None
    generations = _directory(generations)
    if generation is not None:
        _require(type(generation) is str and _TOKEN.fullmatch(generation), 'invalid generation token')
        if not _has_committed_receipt(generations, generation):
            _raw, current = _current(generations)
            _require(current is not None and current['schemaVersion'] == 2
                     and current['generation'] == generation,
                     'requested generation was never committed')
        return _snapshot(generations, generation)
    _raw, current = _current(generations)
    _require(current is not None, 'current generation marker is missing')
    if current['schemaVersion'] == 1:
        return None
    return _snapshot(generations, current['generation'])


def _write_bytes(path, payload, *, temporary=False):
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        if temporary and created:
            try:
                path.unlink(missing_ok=True)
            except BaseException as cleanup:
                error.add_note('Generation marker temporary cleanup also failed: ' + str(cleanup))
        raise


def _cleanup(paths, primary):
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except BaseException as error:
            if primary is not None:
                primary.add_note('Generation temporary cleanup also failed: ' + str(error))
            else:
                raise


def publish_analysis_generation(storage, result, directory, *, files, analysis_revision=0,
                                before_commit=None, expected_generation=AUTO):
    """Copy/seal all files, then CAS the pointer under one publication lock.

    ``before_commit`` takes no arguments and may reject current identity or
    cancellation. It runs after all copies and seals, immediately before commit.
    A lost acknowledgement can leave a valid committed generation; never delete
    a prior or newly committed generation during error cleanup.
    """
    directory = _directory(directory)
    generations = _safe_path(directory / '.result')
    generations.mkdir(exist_ok=True)
    generations = _directory(generations)
    previous_raw, previous_pointer = _current(generations)
    if expected_generation is not AUTO:
        _require(expected_generation is None or type(expected_generation) is str
                 and _TOKEN.fullmatch(expected_generation), 'invalid expected generation')
        _require((previous_raw is None if expected_generation is None else
                  previous_pointer is not None and previous_pointer['schemaVersion'] == 2
                  and previous_pointer['generation'] == expected_generation),
                 'expected current generation does not match')
    previous_snapshot = None
    if previous_pointer is not None and previous_pointer['schemaVersion'] == 2:
        previous_snapshot = _snapshot(generations, previous_pointer['generation'])
    _require(type(analysis_revision) is int and analysis_revision >= 0, 'invalid analysis revision')
    _require(before_commit is None or callable(before_commit), 'before_commit must be callable')
    _require(isinstance(result, dict) and isinstance(result.get('summary'), dict), 'result summary is missing')
    summary = result['summary']
    _require(type(summary.get('id')) is str and bool(summary['id'])
             and type(summary.get('frames')) is int and summary['frames'] > 0, 'result identity is invalid')
    if 'analysisRevision' in result:
        _require(type(result['analysisRevision']) is int and result['analysisRevision'] == analysis_revision,
                 'result analysis revision differs from publication revision')
    point_analysis = result.get('pointAnalysis')
    if isinstance(point_analysis, dict):
        _require(type(point_analysis.get('analysisRevision')) is int
                 and point_analysis['analysisRevision'] == analysis_revision,
                 'point analysis revision differs from publication revision')
    if previous_snapshot is not None:
        _require(previous_snapshot.analysis_id == summary['id']
                 and analysis_revision >= previous_snapshot.analysis_revision,
                 'analysis identity changed or revision moved backwards')
    _require(isinstance(files, Mapping) and len(files) <= MAX_FILES - 2, 'invalid file mapping or file count')
    items = tuple(files.items())
    names = [name for name, _source in items]
    for name in names:
        _name(name)
    _require(len({name.casefold() for name in names + ['result.json', 'summary.json']}) == len(names) + 2,
             'duplicate or reserved generated filename')
    identifier = uuid.uuid4().hex
    assets = generations / (identifier + '.files')
    assets.mkdir()
    records = {}
    for name, source in items:
        source = Path(source)
        source = source if source.is_absolute() else directory / source
        records[name] = _copy(source, assets / name)
    for name in ('result.json', 'summary.json'):
        path = assets / name
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            if name == 'result.json':
                storage.dump(result, stream)
            else:
                json.dump(summary, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        records[name] = _file_record(path)
    storage.set_root(result)
    database = storage.publish(generations / (identifier + '.sqlite3'))
    with database.open('r+b') as stream:
        os.fsync(stream.fileno())
    database_record = _file_record(database)
    manifest = {'schemaVersion': 1, 'analysisId': summary['id'], 'analysisRevision': analysis_revision,
                'frames': summary['frames'], 'database': {'name': database.name, **database_record},
                'filesDirectory': assets.name, 'files': records}
    raw = json.dumps(manifest, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
    _require(len(raw) <= MAX_MANIFEST_BYTES, 'manifest exceeds bounded size')
    token = hashlib.sha256(raw).hexdigest()
    manifest_path = generations / (token + '.generation')
    _write_bytes(manifest_path, raw)
    _fsync_directory(assets)
    _fsync_directory(generations)
    # Copy compatibility JSON to owned unique temporaries. These files remain
    # diagnostic aliases; authoritative readers use only generation manifests.
    temporary = []
    replaced = []
    backups = {}
    primary = None
    try:
        for name in ('result.json', 'summary.json'):
            path = directory / ('.' + identifier + '.' + name + '.tmp')
            _copy(assets / name, path, temporary=True)
            temporary.append(path)
        pointer_temporary = generations / (identifier + '.store.tmp')
        _write_bytes(pointer_temporary, json.dumps({'schemaVersion': 2, 'generation': token},
                                                  separators=(',', ':')).encode('utf-8') + b'\n', temporary=True)
        temporary.append(pointer_temporary)
        lock_path = _safe_path(generations / 'publication.lock')
        if lock_path.exists():
            _path_stat(lock_path)
        with FileLock(str(lock_path), timeout=30):
            current_raw, _ = _current(generations)
            _require(current_raw == previous_raw, 'current generation changed during publication')
            for name in ('result.json', 'summary.json'):
                destination = directory / name
                try:
                    destination.lstat()
                except FileNotFoundError:
                    continue
                backup = generations / (identifier + '.' + name + '.previous')
                _copy(destination, backup)
                backups[destination] = backup
            try:
                for path, name in zip(temporary[:2], ('result.json', 'summary.json'), strict=True):
                    destination = directory / name
                    os.replace(path, destination)
                    replaced.append(destination)
                if before_commit is not None:
                    before_commit()
                _require(_current(generations)[0] == previous_raw, 'current generation changed before commit')
                if previous_pointer is not None and previous_pointer['schemaVersion'] == 2:
                    _retain_current_receipt(generations, previous_pointer['generation'], identifier)
                os.replace(pointer_temporary, generations / 'current.store')
                _fsync_directory(generations)
            except BaseException as error:
                # If replacement reached disk but acknowledgement was lost, the
                # new complete generation is authoritative. Do not roll it back.
                try:
                    committed = _read_small(generations / 'current.store') == json.dumps(
                        {'schemaVersion': 2, 'generation': token}, separators=(',', ':')).encode('utf-8') + b'\n'
                except (OSError, ValueError):
                    committed = False
                if not committed:
                    for destination in reversed(replaced):
                        try:
                            backup = backups.get(destination)
                            if backup is None:
                                destination.unlink(missing_ok=True)
                            else:
                                restore = directory / ('.' + identifier + '.' + destination.name + '.restore.tmp')
                                temporary.append(restore)
                                _copy(backup, restore, temporary=True)
                                os.replace(restore, destination)
                        except BaseException as cleanup:
                            error.add_note('Generation compatibility rollback also failed: ' + str(cleanup))
                raise
    except BaseException as error:
        primary = error
        raise
    finally:
        _cleanup(temporary, primary)
    return token
