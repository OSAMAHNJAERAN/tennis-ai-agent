"""Sealed, typed completed-player observations for native inference recovery.

This owner preserves a completed whole-video player pass. It does not restore
an interrupted tracker, advance a ball frontier, or qualify any measurement.
Writers retain unsealed diagnostics; readers never modify the retained files.
"""
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile

import numpy as np

from src.court.calibration import CourtCalibration
from src.utils.frame_registration_sequence import FrameRegistration, _array_parts, _array_from_parts
from src.utils.identity_frame_sequence import _encode as _identity_encode, _decode as _identity_decode
from src.utils.player_detection_sequence import (_encode_frame as _players_encode,
    _decode_frame as _players_decode, _encode_value, _decode_value)


DATABASE_NAME = '.native_inference_prefix.sqlite3'
SEAL_NAME = '.native_inference_prefix.json'
AUDIT_NAMES = ('camera_registration.json', 'camera_intervals.json', 'match_identity.json', 'image_people.json')
MAX_IDENTITY_BYTES = 64 * 1024
MAX_SEAL_BYTES = 128 * 1024
MAX_HEADER_BYTES = 128 * 1024
MAX_BOX_BYTES = 128 * 1024
MAX_REGISTRATION_BYTES = 32 * 1024
MAX_CANDIDATE_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAME_COUNT = 10_000_000
_DIRECTORY_SYNC_SUPPORTED = os.name != 'nt'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_CUDA = re.compile(r'cuda:(0|[1-9][0-9]*)\Z')
_CALIBRATION_SCALARS = ('inlier_mask', 'reprojection_error_px', 'reprojection_error_m',
                        'inlier_reprojection_error_px', 'rejection_reason')
_FRAME_COLUMNS = ('frame_index', 'role_1', 'role_2', 'boxes', 'registration', 'image', 'anchor',
                  'candidate', 'row_sha256')


def _plain(value, active=None, depth=0):
    if depth > 64:
        raise ValueError('Prefix JSON nesting exceeds its bound')
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Prefix identity and metadata require finite JSON')
        return
    if type(value) not in (dict, list):
        raise ValueError('Prefix identity and metadata require plain JSON types')
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('Prefix JSON must not be circular')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError('Prefix JSON keys must be strings')
            values = value.values()
        else:
            values = value
        for item in values:
            _plain(item, active, depth + 1)
    finally:
        active.remove(id(value))


def _canonical(value):
    _plain(value)
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'), ensure_ascii=True)


def _identity(value):
    if type(value) is not dict or not value:
        raise ValueError('Prefix identity must be a nonempty plain object')
    encoded = _canonical(value)
    if len(encoded.encode('ascii')) > MAX_IDENTITY_BYTES:
        raise ValueError('Prefix identity exceeds 64 KiB')
    return encoded


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate field in native prefix JSON')
        value[key] = item
    return value


def _load_json(value):
    def reject(constant):
        raise ValueError('Nonfinite JSON token in typed native prefix: ' + constant)
    try:
        return json.loads(value, object_pairs_hook=_unique, parse_constant=reject)
    except (RecursionError, OverflowError) as error:
        raise ValueError('Native prefix JSON exceeds its nesting/number bound') from error


def _directory(value):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Native prefix paths cannot contain parent traversal')
    path = path.absolute()
    for ancestor in (path, *path.parents):
        details = ancestor.lstat()
        if (stat.S_ISLNK(details.st_mode)
                or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
            raise ValueError('Native prefix paths cannot use links, junctions or reparse points')
    if not path.is_dir():
        raise ValueError('Native prefix directory must already exist')
    return path


def _signature(details):
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _regular(path):
    details = path.lstat()
    if (not stat.S_ISREG(details.st_mode)
            or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
        raise ValueError('Native prefix artifacts must be regular files')
    return details


def _file_record(path):
    before = _regular(path)
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        if _signature(os.fstat(stream.fileno())) != _signature(before):
            raise ValueError('Native prefix artifact changed while opening')
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            count += len(block)
        after = os.fstat(stream.fileno())
    if (_signature(before) != _signature(after) or _signature(before) != _signature(_regular(path))
            or count != before.st_size):
        raise ValueError('Native prefix artifact changed while hashing')
    return {'bytes': count, 'sha256': digest.hexdigest()}


def _metadata(value):
    if type(value) is not dict or set(value) != {'frame_count', 'fps', 'width', 'height'}:
        raise ValueError('Prefix video metadata requires frame_count/fps/width/height')
    for name, maximum in (('frame_count', MAX_FRAME_COUNT), ('width', 16384), ('height', 16384)):
        if type(value[name]) is not int or not 1 <= value[name] <= maximum:
            raise ValueError('Invalid native prefix ' + name)
    if (type(value['fps']) not in (int, float) or not math.isfinite(value['fps'])
            or not 0 < value['fps'] <= 120):
        raise ValueError('Invalid native prefix fps')
    return _load_json(_canonical(value))


def _crop(value):
    from src.detection.yolo26_adapters import _native_crop_options
    required = {'schema_version', 'role', 'selected_model_path', 'tracking_policy', 'configured_device',
                'resolved_device', 'input_spec', 'prediction_calls', 'completed_prediction_calls', 'failed'}
    if type(value) is not dict or set(value) != required:
        raise ValueError('Malformed clean player crop record')
    device = value['resolved_device']
    if (type(value['schema_version']) is not int or value['schema_version'] != 1
            or value['role'] != 'players' or value['tracking_policy'] != 'DISABLED_BOUND_SOURCE_CROPS'
            or type(device) is not str or not (device == 'cpu' or _CUDA.fullmatch(device))
            or type(value['selected_model_path']) is not str or not 1 <= len(value['selected_model_path']) <= 4096
            or type(value['configured_device']) is not str or not 1 <= len(value['configured_device']) <= 128
            or type(value['prediction_calls']) is not int or value['prediction_calls'] != 0
            or type(value['completed_prediction_calls']) is not int or value['completed_prediction_calls'] != 0
            or value['failed'] is not False or type(value['input_spec']) is not dict):
        raise ValueError('Prefix requires an unused, successful, explicitly bound crop provider')
    options = value['input_spec']
    if 'imgsz' not in options or 'single_cls' not in options:
        raise ValueError('Prefix crop preprocessing is incomplete')
    expected = _native_crop_options({key: options[key] for key in ('imgsz', 'single_cls')}, device)
    if _canonical(expected) != _canonical(options):
        raise ValueError('Prefix crop policy differs from the native selected-model policy')
    return _load_json(_canonical(value))


def _array_metadata(metadata, payload, shape):
    if metadata is None:
        if payload is not None:
            raise ValueError('Null native prefix array has a buffer')
        return
    if (type(metadata) is not dict or set(metadata) != {'dtype', 'shape'}
            or metadata['shape'] != list(shape) or any(type(item) is not int for item in metadata['shape'])
            or type(metadata['dtype']) is not str or len(metadata['dtype']) > 16):
        raise ValueError('Native prefix array shape/metadata is invalid')
    dtype = np.dtype(metadata['dtype'])
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biufc' or not 1 <= dtype.itemsize <= 16:
        raise ValueError('Unsafe native prefix numeric dtype')
    if type(payload) is not bytes or len(payload) != math.prod(shape) * dtype.itemsize:
        raise ValueError('Native prefix array buffer size differs from its fixed shape')


def _encode_registration(record):
    if type(record) is not FrameRegistration or type(record.calibration) is not CourtCalibration:
        raise TypeError('Native prefix requires raw FrameRegistration/CourtCalibration')
    cal = record.calibration
    if (type(cal.inlier_mask) not in (tuple, list) or len(cal.inlier_mask) > 14
            or any(not isinstance(item, (bool, np.bool_)) for item in cal.inlier_mask)):
        raise ValueError('Invalid native prefix calibration mask')
    for name in _CALIBRATION_SCALARS[1:4]:
        value = getattr(cal, name)
        if value is not None and (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))):
            raise ValueError('Invalid native prefix calibration residual')
    if cal.rejection_reason is not None and (type(cal.rejection_reason) is not str or len(cal.rejection_reason) > 4096):
        raise ValueError('Invalid native prefix calibration rejection reason')
    image_meta, image = _array_parts(cal.image_to_court)
    anchor_meta, anchor = _array_parts(record.anchor_to_frame_px)
    _array_metadata(image_meta, image, (3, 3))
    _array_metadata(anchor_meta, anchor, (3, 3))
    # Scalar tags retain signed zero/NaN payloads and NumPy scalar types, unlike
    # sanitized/public calibration JSON. Matrix buffers retain byte order.
    metadata = {name: _encode_value(getattr(cal, name)) for name in _CALIBRATION_SCALARS}
    metadata.update(image_to_court=image_meta, anchor_to_frame_px=anchor_meta)
    payload = json.dumps(metadata, separators=(',', ':'), allow_nan=False)
    if len(payload.encode('utf-8')) > MAX_REGISTRATION_BYTES:
        raise ValueError('Native prefix registration exceeds its bound')
    return payload, image, anchor


def _decode_registration(payload, image, anchor):
    metadata = _load_json(payload)
    if type(metadata) is not dict or set(metadata) != set(_CALIBRATION_SCALARS) | {'image_to_court', 'anchor_to_frame_px'}:
        raise ValueError('Malformed native prefix registration')
    _array_metadata(metadata['image_to_court'], image, (3, 3))
    _array_metadata(metadata['anchor_to_frame_px'], anchor, (3, 3))
    values = {name: _decode_value(metadata[name]) for name in _CALIBRATION_SCALARS}
    record = FrameRegistration(CourtCalibration(
        _array_from_parts(metadata['image_to_court'], image), **values),
        _array_from_parts(metadata['anchor_to_frame_px'], anchor))
    _encode_registration(record)
    return record


def _validate_typed_tree(node):
    pending, count = [(node, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        if count > 50000 or depth > 64 or type(value) is not list or not value or type(value[0]) is not str:
            raise ValueError('Native candidate encoding exceeds its bounded typed schema')
        kind = value[0]
        if kind in ('array', 'scalar'):
            if (len(value) != 4 or type(value[1]) is not str or len(value[1]) > 16
                    or type(value[2]) is not list or len(value[2]) > 2
                    or any(type(item) is not int or item < 0 for item in value[2])
                    or (kind == 'scalar' and value[2]) or type(value[3]) is not str):
                raise ValueError('Unsafe native candidate array metadata')
            dtype = np.dtype(value[1])
            if (dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biufc'
                    or not 1 <= dtype.itemsize <= 16 or math.prod(value[2]) > 4096
                    or len(value[3]) != math.prod(value[2]) * dtype.itemsize * 2):
                raise ValueError('Unsafe native candidate array buffer')
        elif kind in ('list', 'tuple'):
            if len(value) != 2 or type(value[1]) is not list or len(value[1]) > 4096:
                raise ValueError('Native candidate container exceeds its bound')
            pending.extend((item, depth + 1) for item in value[1])
        elif kind == 'dict':
            if len(value) != 2 or type(value[1]) is not list or len(value[1]) > 4096:
                raise ValueError('Native candidate dictionary exceeds its bound')
            for pair in value[1]:
                if type(pair) is not list or len(pair) != 2:
                    raise ValueError('Malformed native candidate dictionary')
                pending.extend((item, depth + 1) for item in pair)
        elif kind not in ('null', 'bool', 'int', 'float', 'str'):
            raise ValueError('Unknown native candidate type tag')


def _candidate(value, index):
    if (type(value) is not dict or set(value) != {0, 1, 2} or any(type(key) is not int for key in value)
            or type(value[0]) is not dict or type(value[1]) is not dict
            or any(type(value[key].get('frame_index')) is not int or value[key]['frame_index'] != index for key in (0, 1))):
        raise ValueError('Native prefix candidate observations do not align with their frame')
    points = value[2]
    if (type(points) is not np.ndarray or points.shape != (14, 2) or points.dtype.hasobject
            or points.dtype.fields is not None or points.dtype.kind not in 'biuf' or points.dtype.itemsize > 16):
        raise ValueError('Native prefix requires fourteen two-dimensional landmarks')


def _digest(parts):
    result = hashlib.sha256()
    for value in parts:
        if value is None:
            raw = b'N'
        elif type(value) is int:
            raw = b'I' + str(value).encode('ascii')
        elif type(value) is str:
            raw = b'T' + value.encode('utf-8')
        elif type(value) is bytes:
            raw = b'B' + value
        else:
            raise ValueError('Unexpected SQLite type in native prefix frame')
        result.update(len(raw).to_bytes(8, 'big'))
        result.update(raw)
    return result.hexdigest()


@dataclass(frozen=True)
class NativePrefixFrame:
    player1: object
    player2: object
    source_indices: dict
    registration: FrameRegistration
    candidate_frame: dict


def _frame_parts(record, index):
    if type(record) is not NativePrefixFrame:
        raise TypeError('Native prefix producer must yield NativePrefixFrame records')
    indices = record.source_indices
    if (type(indices) is not dict or set(indices) != {1, 2} or any(type(key) is not int for key in indices)
            or any(value is not None and (type(value) is not int or not 0 <= value < 4096) for value in indices.values())
            or (indices[1] is not None and indices[1] == indices[2])):
        raise ValueError('Native prefix source proposal indices are invalid')
    boxes = []
    for role, box in ((1, record.player1), (2, record.player2)):
        if (box is None) != (indices[role] is None):
            raise ValueError('Native prefix selected box and source index disagree')
        if box is not None:
            boxes.append(box)
    player_payload = _players_encode(boxes, maximum=2)
    registration, image, anchor = _encode_registration(record.registration)
    _candidate(record.candidate_frame, index)
    node = _identity_encode(record.candidate_frame)
    _validate_typed_tree(node)
    candidate = json.dumps(node, separators=(',', ':'), allow_nan=False)
    for payload, maximum in ((player_payload, MAX_BOX_BYTES), (candidate, MAX_CANDIDATE_BYTES)):
        if len(payload.encode('utf-8')) > maximum:
            raise ValueError('Native prefix frame exceeds its encoded bound')
    parts = (index, indices[1], indices[2], player_payload, registration, image, anchor, candidate)
    return (*parts, _digest(parts))


def _audit_records(directory, count):
    from src.api.result_storage import ResultStorage
    records = {}
    for name in AUDIT_NAMES:
        path = directory / name
        before = _file_record(path)
        with ResultStorage(directory) as storage:
            document = storage.read_json(path, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError('Nonfinite public prefix audit: ' + value)))
            if not isinstance(document, Mapping) or 'frames' not in document:
                raise ValueError('Native prefix audit is missing its frame array: ' + name)
            frames = document['frames']
            if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or len(frames) != count:
                raise ValueError('Native prefix audit frame count differs: ' + name)
            for index, value in enumerate(frames):
                if not isinstance(value, Mapping) or type(value.get('frame_index')) is not int or value['frame_index'] != index:
                    raise ValueError('Native prefix audit frame indices differ: ' + name)
        if _file_record(path) != before:
            raise ValueError('Native prefix audit changed during verification')
        records[name] = {**before, 'frames': count}
    return records


def _connect(path, *, readonly=False):
    connection = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) if readonly else sqlite3.connect(path)
    try:
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_FRAME_BYTES)
        connection.execute('PRAGMA cache_size=-2048').close()
        connection.execute('PRAGMA mmap_size=0').close()
        connection.execute('PRAGMA temp_store=FILE').close()
        if readonly:
            connection.execute('PRAGMA query_only=ON').close()
        else:
            connection.execute('PRAGMA journal_mode=DELETE').close()
            connection.execute('PRAGMA synchronous=FULL').close()
        return connection
    except BaseException as error:
        _close(connection, error)
        raise


def _close(connection, error=None):
    if connection is None:
        return
    try:
        connection.close()
    except BaseException as cleanup_error:
        if error is None:
            raise
        error.add_note('Native prefix database cleanup also failed: ' + str(cleanup_error))


@contextmanager
def _cursor(connection, statement, parameters=()):
    cursor = connection.execute(statement, parameters)
    try:
        yield cursor
    except BaseException as error:
        try:
            cursor.close()
        except BaseException as cleanup_error:
            error.add_note('Native prefix cursor cleanup also failed: ' + str(cleanup_error))
        raise
    else:
        cursor.close()


def _sync_file(path):
    with path.open('r+b') as stream:
        os.fsync(stream.fileno())


def _sync_directory(directory):
    if _DIRECTORY_SYNC_SUPPORTED:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(descriptor)
        except BaseException as error:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                error.add_note('Native prefix directory descriptor cleanup also failed: ' + str(cleanup_error))
            raise
        else:
            os.close(descriptor)


def write_native_inference_prefix(directory, identity, *, metadata, initial_calibration,
                                  initial_court_keypoints, crop_execution, frames):
    """Publish a complete prefix once; retain all unsealed diagnostics on error.

    ``frames`` is consumed one record at a time. Four already closed audit JSON
    files must exist in the same directory. No source/normalized media is made
    resumable by this function; the worker must bind those independently.
    """
    identity_text, metadata, crop_execution = _identity(identity), _metadata(metadata), _crop(crop_execution)
    directory = _directory(directory)
    for name in (DATABASE_NAME, SEAL_NAME):
        try:
            (directory / name).lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError('Native inference prefix already has retained state: ' + name)
    initial_reg, initial_image, initial_anchor = _encode_registration(FrameRegistration(initial_calibration, None))
    points_meta, points = _array_parts(initial_court_keypoints)
    _array_metadata(points_meta, points, (14, 2))
    if points_meta is None or np.dtype(points_meta['dtype']).kind not in 'biuf':
        raise ValueError('Native initial court landmarks must be a real fourteen-point array')
    header = {'schema_version': 1, 'identity': _load_json(identity_text), 'metadata': metadata,
              'crop_execution': crop_execution}
    header_text = _canonical(header)
    if len(header_text.encode('ascii')) > MAX_HEADER_BYTES:
        raise ValueError('Native prefix header exceeds its bound')
    with tempfile.NamedTemporaryFile(dir=directory, prefix='.native_prefix.', suffix='.sqlite3', delete=False) as stream:
        staging = Path(stream.name)
    connection, iterator, iterator_closed = None, iter(frames), False
    try:
        connection = _connect(staging)
        connection.execute('CREATE TABLE header (slot INTEGER PRIMARY KEY, payload TEXT NOT NULL)').close()
        connection.execute('CREATE TABLE initial (slot INTEGER PRIMARY KEY, registration TEXT NOT NULL, image BLOB, '
                           'points_metadata TEXT NOT NULL, points BLOB NOT NULL)').close()
        connection.execute('CREATE TABLE frames (frame_index INTEGER PRIMARY KEY, role_1 INTEGER, role_2 INTEGER, '
                           'boxes TEXT NOT NULL, registration TEXT NOT NULL, image BLOB, anchor BLOB, '
                           'candidate TEXT NOT NULL, row_sha256 TEXT NOT NULL)').close()
        connection.execute('INSERT INTO header VALUES(1,?)', (header_text,)).close()
        connection.execute('INSERT INTO initial VALUES(1,?,?,?,?)',
                           (initial_reg, initial_image, _canonical(points_meta), points)).close()
        count = 0
        for record in iterator:
            if count >= metadata['frame_count']:
                raise ValueError('Native prefix producer has more frames than declared')
            connection.execute('INSERT INTO frames VALUES(?,?,?,?,?,?,?,?,?)', _frame_parts(record, count)).close()
            count += 1
            if count % 256 == 0:
                connection.commit()
        if count != metadata['frame_count']:
            raise ValueError('Native prefix producer ended before its declared frame count')
        close = getattr(iterator, 'close', None)
        if close is not None:
            iterator_closed = True
            close()
        connection.commit()
        _close(connection)
        connection = None
        _sync_file(staging)
        audits = _audit_records(directory, count)
        database = _file_record(staging)
        seal = {'schema_version': 1, 'stage': 'NATIVE_PLAYER_PREFIX_COMPLETED',
                'identity': _load_json(identity_text), 'metadata': metadata,
                'database': {'name': DATABASE_NAME, **database}, 'audits': audits}
        encoded = (_canonical(seal) + '\n').encode('ascii')
        if len(encoded) > MAX_SEAL_BYTES:
            raise ValueError('Native prefix seal exceeds its bound')
        # Both links are exclusive same-volume publications. A process loss
        # between them leaves no valid marker and cannot admit partial state.
        os.link(staging, directory / DATABASE_NAME)
        with tempfile.NamedTemporaryFile(dir=directory, prefix='.native_prefix.', suffix='.json', delete=False) as stream:
            marker_staging = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if _file_record(directory / DATABASE_NAME) != database or _audit_records(directory, count) != audits:
            raise ValueError('Native prefix artifacts changed before seal publication')
        os.link(marker_staging, directory / SEAL_NAME)
        _sync_directory(directory)
        # Staging links remain retained diagnostics. They share the immutable
        # bytes, rather than doubling the database's storage or deleting files.
        return _load_json(encoded)
    except BaseException as error:
        try:
            close = getattr(iterator, 'close', None)
            if close is not None and not iterator_closed:
                iterator_closed = True
                close()
        except BaseException as cleanup_error:
            error.add_note('Native prefix producer cleanup also failed: ' + str(cleanup_error))
        _close(connection, error)
        raise


class _FieldView(Sequence):
    def __init__(self, owner, field):
        self._owner, self._field = owner, field

    def __len__(self):
        return len(self._owner)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._owner[index]
        if self._field == 'indices':
            return record.source_indices
        if self._field in (1, 2):
            return record.player1 if self._field == 1 else record.player2
        if self._field == 'calibration':
            return record.registration.calibration
        if self._field == 'transform':
            return record.registration.anchor_to_frame_px
        return getattr(record, self._field)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class _Players(Mapping):
    def __init__(self, owner):
        self._roles = {role: _FieldView(owner, role) for role in (1, 2)}
        self.selected_detection_indices = _FieldView(owner, 'indices')

    def __len__(self):
        return 2

    def __iter__(self):
        return iter((1, 2))

    def __getitem__(self, role):
        if type(role) is not int:
            raise KeyError(role)
        return self._roles[role]

    def index(self, role, frame_index):
        if type(role) is not int or role not in (1, 2):
            raise KeyError(role)
        return self.selected_detection_indices[frame_index][role]


class NativeInferencePrefix(Sequence):
    """Read-only aligned views; each read returns detached typed observations."""
    def __init__(self, directory, identity, *, expected_metadata=None):
        self.directory = _directory(directory)
        self._identity = _identity(identity)
        self._expected_metadata = None if expected_metadata is None else _metadata(expected_metadata)
        self._connection = self._db_signature = None
        self._entered = False
        self.players = _Players(self)
        self.registrations = _FieldView(self, 'registration')
        self.registrations.calibrations = _FieldView(self, 'calibration')
        self.registrations.transforms = _FieldView(self, 'transform')
        self.candidate_frames = _FieldView(self, 'candidate_frame')

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Native prefix reader cannot be reused')
        self._entered = True
        try:
            seal_path = self.directory / SEAL_NAME
            details = _regular(seal_path)
            if details.st_size > MAX_SEAL_BYTES:
                raise ValueError('Native prefix seal exceeds its bound')
            with seal_path.open('rb') as stream:
                if _signature(os.fstat(stream.fileno())) != _signature(details):
                    raise ValueError('Native prefix seal changed while opening')
                encoded = stream.read(MAX_SEAL_BYTES + 1)
            if len(encoded) != details.st_size or _signature(_regular(seal_path)) != _signature(details):
                raise ValueError('Native prefix seal changed during reading')
            seal = _load_json(encoded)
            if (type(seal) is not dict or set(seal) != {'schema_version', 'stage', 'identity', 'metadata', 'database', 'audits'}
                    or type(seal['schema_version']) is not int or seal['schema_version'] != 1
                    or seal['stage'] != 'NATIVE_PLAYER_PREFIX_COMPLETED' or _identity(seal['identity']) != self._identity):
                raise ValueError('Native prefix seal identity/schema differs')
            self._metadata = _metadata(seal['metadata'])
            if self._expected_metadata is not None and _canonical(self._expected_metadata) != _canonical(self._metadata):
                raise ValueError('Native prefix video metadata differs')
            database = seal['database']
            if (type(database) is not dict or set(database) != {'name', 'bytes', 'sha256'}
                    or database['name'] != DATABASE_NAME or type(database['bytes']) is not int or database['bytes'] <= 0
                    or type(database['sha256']) is not str or _SHA.fullmatch(database['sha256']) is None):
                raise ValueError('Native prefix database binding is malformed')
            path = self.directory / DATABASE_NAME
            self._db_signature = _signature(_regular(path))
            if _file_record(path) != {key: database[key] for key in ('bytes', 'sha256')}:
                raise ValueError('Native prefix database hash differs')
            if _canonical(_audit_records(self.directory, self._metadata['frame_count'])) != _canonical(seal['audits']):
                raise ValueError('Native prefix audit bindings differ')
            if _signature(_regular(path)) != self._db_signature:
                raise ValueError('Native prefix database changed during audit verification')
            self._connection = _connect(path, readonly=True)
            self._validate_database()
            self._seal = seal
            return self
        except BaseException as error:
            connection, self._connection = self._connection, None
            _close(connection, error)
            raise

    def _one(self, statement, parameters=()):
        with _cursor(self._open(), statement, parameters) as cursor:
            return cursor.fetchone()

    def _open(self):
        if self._connection is None:
            raise RuntimeError('Native prefix reader is not open')
        if _signature(_regular(self.directory / DATABASE_NAME)) != self._db_signature:
            raise ValueError('Native prefix database changed while open')
        return self._connection

    def _validate_database(self):
        connection = self._open()
        # A prefix owns exactly three public schema objects. Count first and
        # bound retrieval even for a forged seal/database; extra views,
        # triggers and indexes must not be silently accepted as unused state.
        public_objects = "lower(substr(name,1,7)) != 'sqlite_'"
        if self._one('SELECT COUNT(*) FROM sqlite_master WHERE ' + public_objects) != (3,):
            raise ValueError('Native prefix SQLite schema differs')
        with _cursor(connection, 'SELECT type,name FROM sqlite_master WHERE ' + public_objects +
                     ' ORDER BY name LIMIT 4') as cursor:
            objects = cursor.fetchmany(4)
        if objects != [('table', 'frames'), ('table', 'header'), ('table', 'initial')]:
            raise ValueError('Native prefix SQLite schema differs')
        expected = {
            'frames': (('frame_index', 'INTEGER', 0, 1), ('role_1', 'INTEGER', 0, 0),
                       ('role_2', 'INTEGER', 0, 0), ('boxes', 'TEXT', 1, 0),
                       ('registration', 'TEXT', 1, 0), ('image', 'BLOB', 0, 0),
                       ('anchor', 'BLOB', 0, 0), ('candidate', 'TEXT', 1, 0),
                       ('row_sha256', 'TEXT', 1, 0)),
            'header': (('slot', 'INTEGER', 0, 1), ('payload', 'TEXT', 1, 0)),
            'initial': (('slot', 'INTEGER', 0, 1), ('registration', 'TEXT', 1, 0),
                        ('image', 'BLOB', 0, 0), ('points_metadata', 'TEXT', 1, 0),
                        ('points', 'BLOB', 1, 0)),
        }
        for table, columns in expected.items():
            with _cursor(connection, 'PRAGMA table_info(' + table + ')') as cursor:
                rows = cursor.fetchmany(len(columns) + 1)
                actual = tuple((row[1], row[2], row[3], row[5]) for row in rows)
            if actual != columns:
                raise ValueError('Native prefix SQLite columns differ')
        if self._one('PRAGMA quick_check(1)') != ('ok',):
            raise ValueError('Native prefix SQLite integrity failed')
        if self._one('SELECT COUNT(*),MIN(slot),MAX(slot) FROM header') != (1, 1, 1):
            raise ValueError('Native prefix header is missing or duplicated')
        size = self._one('SELECT length(CAST(payload AS BLOB)) FROM header WHERE slot=1')[0]
        if type(size) is not int or size > MAX_HEADER_BYTES:
            raise ValueError('Native prefix header exceeds its bound')
        header = _load_json(self._one('SELECT payload FROM header WHERE slot=1')[0])
        if (type(header) is not dict or set(header) != {'schema_version', 'identity', 'metadata', 'crop_execution'}
                or type(header['schema_version']) is not int or header['schema_version'] != 1
                or _identity(header['identity']) != self._identity
                or _canonical(_metadata(header['metadata'])) != _canonical(self._metadata)):
            raise ValueError('Native prefix header binding differs')
        self._crop_execution = _crop(header['crop_execution'])
        count = self._metadata['frame_count']
        if self._one('SELECT COUNT(*),MIN(frame_index),MAX(frame_index) FROM frames') != (count, 0, count - 1):
            raise ValueError('Native prefix frame count/contiguity differs')
        invalid = self._one('SELECT COUNT(*) FROM frames WHERE '
            'length(CAST(boxes AS BLOB))>? OR length(CAST(registration AS BLOB))>? OR '
            'length(CAST(candidate AS BLOB))>? OR length(image)>144 OR length(anchor)>144 OR '
            'length(CAST(row_sha256 AS BLOB))!=64',
            (MAX_BOX_BYTES, MAX_REGISTRATION_BYTES, MAX_CANDIDATE_BYTES))[0]
        if invalid:
            raise ValueError('Native prefix stored row exceeds its bound')
        if self._one('SELECT COUNT(*),MIN(slot),MAX(slot) FROM initial') != (1, 1, 1):
            raise ValueError('Native prefix initial record is missing or duplicated')
        sizes = self._one('SELECT length(CAST(registration AS BLOB)),length(image),'
                          'length(CAST(points_metadata AS BLOB)),length(points) FROM initial WHERE slot=1')
        if (type(sizes[0]) is not int or sizes[0] > MAX_REGISTRATION_BYTES
                or (sizes[1] is not None and sizes[1] > 144) or type(sizes[2]) is not int
                or sizes[2] > 1024 or type(sizes[3]) is not int or sizes[3] > 448):
            raise ValueError('Native prefix initial buffers exceed their fixed bounds')
        payload, image, points_meta, points = self._one('SELECT registration,image,points_metadata,points FROM initial WHERE slot=1')
        self._initial_registration = _decode_registration(payload, image, None)
        self._points_meta, self._points = _load_json(points_meta), points
        _array_metadata(self._points_meta, self._points, (14, 2))
        if self._points_meta is None or np.dtype(self._points_meta['dtype']).kind not in 'biuf':
            raise ValueError('Native initial landmarks must be real numeric coordinates')

    @property
    def metadata(self):
        self._open()
        return _load_json(_canonical(self._metadata))

    @property
    def crop_execution(self):
        self._open()
        return _load_json(_canonical(self._crop_execution))

    @property
    def initial_calibration(self):
        self._open()
        payload, image, anchor = _encode_registration(self._initial_registration)
        return _decode_registration(payload, image, anchor).calibration

    @property
    def initial_court_keypoints(self):
        self._open()
        return _array_from_parts(self._points_meta, self._points)

    @property
    def seal(self):
        self._open()
        return _load_json(_canonical(self._seal))

    def __len__(self):
        self._open()
        return self._metadata['frame_count']

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if type(index) is not int:
            raise TypeError('Native prefix frame index must be a plain integer')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sizes = self._one('SELECT length(CAST(boxes AS BLOB)),length(CAST(registration AS BLOB)),length(image),'
                          'length(anchor),length(CAST(candidate AS BLOB)) FROM frames WHERE frame_index=?', (index,))
        if (sizes is None or any(type(sizes[pos]) is not int or sizes[pos] > bound for pos, bound in
                ((0, MAX_BOX_BYTES), (1, MAX_REGISTRATION_BYTES), (4, MAX_CANDIDATE_BYTES)))
                or any(size is not None and (type(size) is not int or size > 144) for size in sizes[2:4])):
            raise ValueError('Native prefix frame is missing or exceeds its fixed bounds')
        row = self._one('SELECT ' + ','.join(_FRAME_COLUMNS) + ' FROM frames WHERE frame_index=?', (index,))
        if type(row[-1]) is not str or _SHA.fullmatch(row[-1]) is None or _digest(row[:-1]) != row[-1]:
            raise ValueError('Native prefix frame hash differs')
        boxes = _players_decode(row[3], maximum=2)
        indices = {1: row[1], 2: row[2]}
        if len(boxes) != sum(value is not None for value in indices.values()):
            raise ValueError('Native prefix role boxes and indices disagree')
        offset = 0
        players = {}
        for role in (1, 2):
            players[role] = None if indices[role] is None else boxes[offset]
            offset += int(indices[role] is not None)
        registration = _decode_registration(row[4], row[5], row[6])
        node = _load_json(row[7])
        _validate_typed_tree(node)
        candidate = _identity_decode(node)
        result = NativePrefixFrame(players[1], players[2], indices, registration, candidate)
        # Reapply producer schema/type rules, including index/box agreement.
        _frame_parts(result, index)
        return result

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __exit__(self, error_type, error, _traceback):
        connection, self._connection = self._connection, None
        _close(connection, error)
        return False


def open_native_inference_prefix(directory, identity, *, expected_metadata=None):
    return NativeInferencePrefix(directory, identity, expected_metadata=expected_metadata)
