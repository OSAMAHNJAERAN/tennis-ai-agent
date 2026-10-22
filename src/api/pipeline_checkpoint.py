"""Seal completed Phase6 exports for post-inference worker continuation.

The caller writes this only after ``Phase6Pipeline.run`` returns and its owned
writers close. This is not a tracker/filter checkpoint or a partial-run reader.
No checkpoint can give its contents event authority or accuracy qualification.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile


CHECKPOINT_NAME = '.pipeline_checkpoint.json'
MAX_CHECKPOINT_BYTES = 256 * 1024
MAX_ARTIFACTS = 256
HASH_CHUNK_BYTES = 1024 * 1024
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z')
_SHA256 = re.compile(r'[0-9a-f]{64}\Z')
_DEVICES = {'CON', 'PRN', 'AUX', 'NUL', *(f'{base}{n}' for base in ('COM', 'LPT') for n in range(1, 10))}


def _plain_json(value, active=None):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Checkpoint identity must contain finite JSON values')
        return
    if type(value) not in (dict, list):
        raise ValueError('Checkpoint identity must contain plain JSON values')
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('Checkpoint identity must not be circular')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError('Checkpoint identity keys must be strings')
            for item in value.values():
                _plain_json(item, active)
        else:
            for item in value:
                _plain_json(item, active)
    finally:
        active.remove(id(value))


def _canonical(value):
    try:
        _plain_json(value)
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    except (RecursionError, OverflowError) as error:
        raise ValueError('Checkpoint identity exceeds supported JSON nesting or number size') from error


def _identity(value):
    if type(value) is not dict or not value:
        raise ValueError('Checkpoint identity must be a nonempty plain JSON object')
    encoded = _canonical(value)
    if len(encoded) > MAX_CHECKPOINT_BYTES // 2:
        raise ValueError('Checkpoint identity is too large')
    return encoded


def _inventory(required_files, optional_files):
    groups = []
    for values in (required_files, optional_files):
        if type(values) not in (tuple, list) or len(values) > MAX_ARTIFACTS:
            raise ValueError('Checkpoint inventory must be bounded filename lists or tuples')
        for name in values:
            if (type(name) is not str or _NAME.fullmatch(name) is None or name.endswith('.')
                    or name.split('.')[0].upper() in _DEVICES):
                raise ValueError('Checkpoint artifacts require safe top-level filenames')
        groups.append(tuple(sorted(values)))
    required, optional = groups
    all_names = required + optional
    if (not required or len(all_names) > MAX_ARTIFACTS
            or len({name.casefold() for name in all_names}) != len(all_names)):
        raise ValueError('Checkpoint inventory requires unique filenames and a required artifact')
    return required, optional


def _directory(value):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Checkpoint directory must not contain parent traversal')
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError('Checkpoint paths must not use symbolic links')
    if not path.is_dir():
        raise ValueError('Checkpoint directory does not exist')
    return path.resolve()


def _signature(details):
    signature = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns)
    # On this Python/NTFS runtime lstat's cached change time can lag fstat by
    # one millisecond even when opening an unchanged file. File identity, size,
    # modification time and streamed content digest remain checked on Windows.
    return signature if os.name == 'nt' else (*signature, details.st_ctime_ns)


def _file_record(path, *, required, sync=False):
    try:
        before = path.lstat()
    except FileNotFoundError:
        if required:
            raise ValueError('Required checkpoint artifact is missing: ' + path.name)
        return {'present': False}
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError('Checkpoint artifact must be a regular file: ' + path.name)
    digest = hashlib.sha256()
    count = 0
    # Windows _commit/fsync requires a descriptor opened for writing. Opening
    # r+b does not truncate or alter bytes; readonly resume checks stay rb.
    with path.open('r+b' if sync else 'rb') as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError('Checkpoint artifact changed while opening: ' + path.name)
        if sync:
            os.fsync(stream.fileno())
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b''):
            count += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    if (not stat.S_ISREG(current.st_mode) or path.is_symlink() or count != before.st_size
            or _signature(before) != _signature(after) or _signature(before) != _signature(current)):
        raise ValueError('Checkpoint artifact changed during hashing: ' + path.name)
    return {'present': True, 'bytes': count, 'sha256': digest.hexdigest()}


def _reject_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key in pipeline checkpoint')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('Nonfinite value in pipeline checkpoint: ' + value)


def _sync_directory(directory):
    # Windows does not expose directory fsync through the standard library.
    # The completed files and marker are fsynced; atomic NTFS link publication
    # protects process-crash recovery. This does not promise power-loss recovery.
    if os.name != 'nt':
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def write_pipeline_checkpoint(directory, identity, *, required_files, optional_files):
    """Publish a new completed-stage marker; never replace an existing marker.

    Hashing uses bounded reads. An atomic same-volume hard link publishes the
    fsynced marker with exclusive creation, avoiding an exists/replace race.
    A filesystem without hard links fails closed rather than overwriting a seal.
    """
    identity_bytes = _identity(identity)
    required, optional = _inventory(required_files, optional_files)
    directory = _directory(directory)
    destination = directory / CHECKPOINT_NAME
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError('A committed pipeline checkpoint already exists')
    artifacts = {name: _file_record(directory / name, required=name in required, sync=True)
                 for name in sorted(required + optional)}
    document = {'schemaVersion': 1, 'stage': 'PHASE6_COMPLETED',
                'identity': json.loads(identity_bytes), 'requiredFiles': list(required),
                'optionalFiles': list(optional), 'artifacts': artifacts}
    encoded = _canonical(document) + b'\n'
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise ValueError('Pipeline checkpoint exceeds its bounded manifest size')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=directory, prefix='.pipeline_checkpoint.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
        os.link(temporary, destination)
        _sync_directory(directory)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                error.add_note('Pipeline checkpoint temporary cleanup also failed: ' + str(cleanup_error))
        raise
    temporary.unlink()
    return document


def load_pipeline_checkpoint(directory, identity, *, required_files, optional_files):
    """Verify a completed stage, returning None only when its marker is absent.

    Any malformed marker, changed input identity, changed inventory or changed
    artifact raises. The returned JSON tree is detached from the caller's tree.
    The worker must still validate media/results before marking its job complete.
    """
    identity_bytes = _identity(identity)
    required, optional = _inventory(required_files, optional_files)
    directory = _directory(directory)
    marker = directory / CHECKPOINT_NAME
    try:
        details = marker.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(details.st_mode) or marker.is_symlink() or details.st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError('Pipeline checkpoint must be a bounded regular file')
    try:
        with marker.open('rb') as stream:
            if _signature(os.fstat(stream.fileno())) != _signature(details):
                raise ValueError('Pipeline checkpoint changed while opening')
            payload = stream.read(MAX_CHECKPOINT_BYTES + 1)
            after = os.fstat(stream.fileno())
        if (len(payload) > MAX_CHECKPOINT_BYTES or _signature(details) != _signature(after)
                or _signature(details) != _signature(marker.lstat())):
            raise ValueError('Pipeline checkpoint changed during reading')
        document = json.loads(payload.decode('utf-8'), object_pairs_hook=_reject_pairs, parse_constant=_reject_constant)
    except (UnicodeError, RecursionError, OverflowError) as error:
        raise ValueError('Malformed pipeline checkpoint JSON') from error
    expected_keys = {'schemaVersion', 'stage', 'identity', 'requiredFiles', 'optionalFiles', 'artifacts'}
    if (type(document) is not dict or set(document) != expected_keys
            or type(document['schemaVersion']) is not int or document['schemaVersion'] != 1
            or document['stage'] != 'PHASE6_COMPLETED'
            or type(document['identity']) is not dict or _canonical(document['identity']) != identity_bytes
            or document['requiredFiles'] != list(required) or document['optionalFiles'] != list(optional)
            or type(document['artifacts']) is not dict
            or set(document['artifacts']) != set(required + optional)):
        raise ValueError('Pipeline checkpoint identity or inventory does not match')
    for name, record in document['artifacts'].items():
        if type(record) is not dict or type(record.get('present')) is not bool:
            raise ValueError('Malformed pipeline checkpoint artifact record')
        if record['present']:
            if (set(record) != {'present', 'bytes', 'sha256'} or type(record['bytes']) is not int
                    or record['bytes'] < 0 or type(record['sha256']) is not str
                    or _SHA256.fullmatch(record['sha256']) is None):
                raise ValueError('Malformed pipeline checkpoint artifact digest')
        elif set(record) != {'present'} or name in required:
            raise ValueError('Required pipeline checkpoint artifact is absent')
        if _file_record(directory / name, required=name in required) != record:
            raise ValueError('Pipeline checkpoint artifact changed: ' + name)
    return document
