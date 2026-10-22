"""Bind normalized input, completed player prefix and raw ball frontier.

These functions coordinate retained artifacts. They load no vision models and
do not run or resume a pipeline themselves. Recovery may open an existing ball
journal read/write so SQLite can roll back a hot journal after process loss;
it accepts no new candidate frame and is not advertised as strictly read-only.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile


SOURCE_SEAL_NAME = '.native_inference_source.json'
REQUIRED_NORMALIZED_FILES = ('source.mp4', 'media_metadata.json', 'inference_config.yaml')
OPTIONAL_NORMALIZED_FILES = ('source_frame_mapping.jsonl', 'normalization_frame_stats.txt')
NORMALIZED_FILES = REQUIRED_NORMALIZED_FILES + OPTIONAL_NORMALIZED_FILES
MAX_IDENTITY_BYTES = 64 * 1024
MAX_SOURCE_SEAL_BYTES = 128 * 1024
MAX_PREFIX_SEAL_BYTES = 128 * 1024
MAX_MEDIA_METADATA_BYTES = 1024 * 1024
_DIRECTORY_SYNC_SUPPORTED = os.name != 'nt'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_CUDA = re.compile(r'cuda:(0|[1-9][0-9]*)\Z')


def _plain(value, active=None, depth=0):
    if depth > 64:
        raise ValueError('Native inference identity exceeds its nesting bound')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Native inference identity requires finite JSON')
        return
    if type(value) not in (dict, list):
        raise ValueError('Native inference identity requires plain JSON types')
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('Native inference identity cannot be circular')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError('Native inference identity keys must be strings')
            items = value.values()
        else:
            items = value
        for item in items:
            _plain(item, active, depth + 1)
    finally:
        active.remove(id(value))


def _canonical(value):
    try:
        _plain(value)
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False)
    except (RecursionError, OverflowError) as error:
        raise ValueError('Native inference JSON exceeds its supported size or nesting') from error


def _identity(value):
    if type(value) is not dict or not value:
        raise ValueError('Native inference identity must be a nonempty plain object')
    encoded = _canonical(value)
    if len(encoded) > MAX_IDENTITY_BYTES:
        raise ValueError('Native inference identity exceeds 64 KiB')
    return encoded


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate field in native inference JSON')
        result[key] = value
    return result


def _loads(value):
    def reject(constant):
        raise ValueError('Nonfinite token in native inference JSON: ' + constant)
    try:
        return json.loads(value, object_pairs_hook=_unique, parse_constant=reject)
    except (RecursionError, OverflowError) as error:
        raise ValueError('Native inference JSON exceeds supported nesting or numbers') from error


def _directory(value):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Native inference directory cannot contain parent traversal')
    path = path.absolute()
    for ancestor in (path, *path.parents):
        details = ancestor.lstat()
        if (stat.S_ISLNK(details.st_mode)
                or (hasattr(ancestor, 'is_junction') and ancestor.is_junction())
                or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('Native inference paths cannot use links, junctions or reparse points')
    if not path.is_dir():
        raise ValueError('Native inference directory must already exist')
    return path


def _regular(path, *, required=True):
    try:
        details = path.lstat()
    except FileNotFoundError:
        if required:
            raise
        return None
    if (not stat.S_ISREG(details.st_mode)
            or (hasattr(path, 'is_junction') and path.is_junction())
            or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('Native inference artifacts must be regular files without redirects')
    return details


def _signature(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


@contextmanager
def _stream(path, mode):
    stream = path.open(mode)
    try:
        yield stream
    except BaseException as error:
        try:
            stream.close()
        except BaseException as cleanup_error:
            error.add_note('Native inference file cleanup also failed: ' + str(cleanup_error))
        raise
    else:
        stream.close()


def _file_record(path, *, required=True, sync=False):
    before = _regular(path, required=required)
    if before is None:
        return {'present': False}
    digest, count = hashlib.sha256(), 0
    # Windows fsync requires a writable descriptor. r+b neither truncates nor
    # changes source bytes; recovery checks use rb and do not request fsync.
    with _stream(path, 'r+b' if sync else 'rb') as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError('Native inference artifact changed while opening')
        if sync:
            os.fsync(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            count += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (count != before.st_size or _signature(after) != _signature(before)
            or _signature(_regular(path)) != _signature(before)):
        raise ValueError('Native inference artifact changed during hashing')
    return {'present': True, 'bytes': count, 'sha256': digest.hexdigest()}


def _inventory(directory, *, sync=False):
    return {name: _file_record(directory / name, required=name in REQUIRED_NORMALIZED_FILES, sync=sync)
            for name in NORMALIZED_FILES}


def _read_json(path, maximum):
    before = _regular(path)
    if before.st_size > maximum:
        raise ValueError('Native inference JSON exceeds its size bound')
    with _stream(path, 'rb') as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError('Native inference JSON changed while opening')
        encoded = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if (len(encoded) != before.st_size or len(encoded) > maximum
            or _signature(after) != _signature(before)
            or _signature(_regular(path)) != _signature(before)):
        raise ValueError('Native inference JSON changed during reading')
    return _loads(encoded), {'present': True, 'bytes': len(encoded),
                             'sha256': hashlib.sha256(encoded).hexdigest()}


def _source_document(value):
    if (type(value) is not dict or set(value) != {'schema_version', 'stage', 'base_identity', 'files'}
            or type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['stage'] != 'NATIVE_NORMALIZED_SOURCE_COMPLETED'):
        raise ValueError('Malformed native normalized source seal')
    _identity(value['base_identity'])
    records = value['files']
    if type(records) is not dict or set(records) != set(NORMALIZED_FILES):
        raise ValueError('Native source seal file inventory differs')
    for name, record in records.items():
        if type(record) is not dict or type(record.get('present')) is not bool:
            raise ValueError('Malformed native source file binding')
        if record['present']:
            if (set(record) != {'present', 'bytes', 'sha256'} or type(record['bytes']) is not int
                    or record['bytes'] < (1 if name in REQUIRED_NORMALIZED_FILES else 0)
                    or type(record['sha256']) is not str
                    or _SHA.fullmatch(record['sha256']) is None):
                raise ValueError('Malformed native source file binding')
        elif set(record) != {'present'} or name in REQUIRED_NORMALIZED_FILES:
            raise ValueError('Native source seal omits a required file')
    encoded = _canonical(value)
    if len(encoded) > MAX_SOURCE_SEAL_BYTES:
        raise ValueError('Native normalized source seal exceeds its bound')
    return _loads(encoded)


def _sync_directory(directory):
    if not _DIRECTORY_SYNC_SUPPORTED:
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(descriptor)
    except BaseException as error:
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            error.add_note('Native inference directory cleanup also failed: ' + str(cleanup_error))
        raise
    else:
        os.close(descriptor)


def seal_normalized_source(directory, base_identity):
    """Exclusively seal normalized assets; never overwrite or delete files."""
    identity_text = _identity(base_identity)
    directory = _directory(directory)
    marker = directory / SOURCE_SEAL_NAME
    try:
        marker.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError('Native normalized source already has a retained seal')
    document = _source_document({'schema_version': 1, 'stage': 'NATIVE_NORMALIZED_SOURCE_COMPLETED',
        'base_identity': _loads(identity_text), 'files': _inventory(directory, sync=True)})
    # Fail before publication if the downstream derived identity would exceed
    # the common prefix/journal identity size bound.
    inference_identity(document)
    encoded = (_canonical(document) + '\n').encode('ascii')
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.native_inference_source.',
                                     suffix='.tmp', delete=False) as stream:
        staging = Path(stream.name)
    with _stream(staging, 'wb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if _canonical(_inventory(directory)) != _canonical(document['files']):
        raise ValueError('Native normalized assets changed before seal publication')
    os.link(staging, marker)
    _sync_directory(directory)
    return _loads(encoded)


def load_normalized_source(directory, base_identity):
    """Return None only for an absent marker; present invalid state raises."""
    expected = _identity(base_identity)
    directory = _directory(directory)
    marker = directory / SOURCE_SEAL_NAME
    if _regular(marker, required=False) is None:
        return None
    document, marker_record = _read_json(marker, MAX_SOURCE_SEAL_BYTES)
    document = _source_document(document)
    if _identity(document['base_identity']) != expected:
        raise ValueError('Native normalized source base identity differs')
    if _canonical(_inventory(directory)) != _canonical(document['files']):
        raise ValueError('Native normalized source file bindings differ')
    if _file_record(marker) != marker_record:
        raise ValueError('Native normalized source seal changed during verification')
    inference_identity(document)
    return document


def inference_identity(source_seal):
    """Detached deterministic identity, with no acquisition or wall time."""
    document = _source_document(source_seal)
    identity = {'schema': 'tennis.native-inference-identity.v1',
                'base_identity': document['base_identity'], 'normalized_files': document['files']}
    return _loads(_identity(identity))


def _resolved_device(value):
    if type(value) is not str or not (value == 'cpu' or _CUDA.fullmatch(value)):
        raise ValueError('Ball journal needs explicit cpu or indexed cuda:N device')
    return value


def ball_journal_identity(directory, inference_identity, ball_config, resolved_device):
    """Bind raw ball rows to exact completed-prefix bytes and operation policy."""
    identity_text, ball_text = _identity(inference_identity), _identity(ball_config)
    device = _resolved_device(resolved_device)
    directory = _directory(directory)
    from src.utils.native_inference_prefix import SEAL_NAME
    prefix_seal, prefix_binding = _read_json(directory / SEAL_NAME, MAX_PREFIX_SEAL_BYTES)
    if (type(prefix_seal) is not dict or type(prefix_seal.get('schema_version')) is not int
            or prefix_seal['schema_version'] != 1
            or prefix_seal.get('stage') != 'NATIVE_PLAYER_PREFIX_COMPLETED'
            or _identity(prefix_seal.get('identity')) != identity_text):
        raise ValueError('Ball journal prefix seal identity differs')
    identity = {'schema': 'tennis.native-ball-journal-identity.v1',
        'inference_identity': _loads(identity_text), 'prefix_seal': prefix_binding,
        'ball_config': _loads(ball_text), 'resolved_device': device}
    return _loads(_identity(identity))


def _mapped_count(directory):
    metadata, _ = _read_json(directory / 'media_metadata.json', MAX_MEDIA_METADATA_BYTES)
    if type(metadata) is not dict:
        raise ValueError('Saved normalized media metadata must be an object')
    if 'timeMapping' not in metadata:
        return None
    timing = metadata['timeMapping']
    if type(timing) is not dict:
        raise ValueError('Saved normalized time mapping must be an object')
    count = timing.get('frames')
    if count is None:
        return None
    if type(count) is not int or count <= 0:
        raise ValueError('Saved normalized frame mapping count is invalid')
    return count


def validate_partial_inference(directory, base_identity, ball_config):
    """Validate reusable completed-player/raw-ball state without inferring.

    Missing prefix marker means unsealed player leftovers are ineligible. A
    valid prefix with no ball database is a zero-frame frontier. An existing
    ball database is opened read/write for SQLite crash recovery, then every
    committed raw row is read through its hash/codec checks. Nothing is added.
    """
    _identity(ball_config)
    directory = _directory(directory)
    source = load_normalized_source(directory, base_identity)
    if source is None:
        return None
    from src.utils.native_inference_prefix import SEAL_NAME, open_native_inference_prefix
    if _regular(directory / SEAL_NAME, required=False) is None:
        return None
    derived = inference_identity(source)
    mapped_count = _mapped_count(directory)
    with open_native_inference_prefix(directory, derived) as prefix:
        total_frames = prefix.metadata['frame_count']
        if mapped_count is not None and mapped_count != total_frames:
            raise ValueError('Saved source mapping and completed player frames differ')
        device = _resolved_device(prefix.crop_execution['resolved_device'])
        identity = ball_journal_identity(directory, derived, ball_config, device)
        from src.utils.ball_inference_journal import JOURNAL_NAME, BallInferenceJournal
        journal_path = directory / JOURNAL_NAME
        if _regular(journal_path, required=False) is None:
            # A redirected or orphaned sidecar is not a clean absent frontier.
            for suffix in ('-journal', '-wal', '-shm'):
                if _regular(Path(str(journal_path) + suffix), required=False) is not None:
                    raise ValueError('Ball inference sidecar exists without its journal')
            next_frame = 0
        else:
            with BallInferenceJournal.open(directory, identity, total_frames) as journal:
                next_frame = journal.next_frame
                for index in range(next_frame):
                    journal[index]
        # Detect ordinary concurrent mutation of any normalized asset or seal.
        if _canonical(load_normalized_source(directory, base_identity)) != _canonical(source):
            raise ValueError('Native normalized source changed during partial validation')
        if _canonical(ball_journal_identity(directory, derived, ball_config, device)) != _canonical(identity):
            raise ValueError('Native prefix seal changed during partial validation')
        prefix.metadata  # Recheck the immutable database's retained signature.
    return {'inference_identity': derived, 'mapped_count': mapped_count,
            'next_frame': next_frame, 'total_frames': total_frames, 'resolved_device': device}
