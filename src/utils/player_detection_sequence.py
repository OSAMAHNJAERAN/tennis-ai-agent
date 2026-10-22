"""Owned indexed player observations without match-long Python box lists.

The private typed JSON codec preserves numeric bits, NumPy scalar types and
anatomical annotations. It never uses pickle or executes an artifact payload.
Reads are independent snapshots: track IDs or equal boxes do not establish
source-object identity, a court role, or confirmed match player A/B.
"""
from collections.abc import Sequence
import json
import operator
from pathlib import Path
import sqlite3
import struct
import tempfile

import numpy as np

from src.utils.bbox_utils import BBox

BOX_FIELDS = ('x1', 'y1', 'x2', 'y2', 'confidence', 'class_id', 'track_id')
ANNOTATIONS = ('keypoints_px', 'ground_point_px', 'ground_point_source',
               'ground_point_quality', 'ground_point_confidence')


def _numeric(value):
    return type(value) in (bool, int, float) or (isinstance(value, np.generic)
                                               and value.dtype.kind in 'biuf')


def _encode_value(value):
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in 'biuf':
            raise TypeError('Player observations require plain real numeric NumPy values.')
        return ['numpy_scalar' if isinstance(value, np.generic) else 'numpy_array',
                array.dtype.str, list(array.shape), array.tobytes(order='C').hex()]
    if value is None:
        return ['null']
    if type(value) is float:
        return ['float64', struct.pack('!d', value).hex()]
    if type(value) in (bool, int, str):
        return [{bool: 'bool', int: 'int', str: 'str'}[type(value)], value]
    if isinstance(value, (list, tuple)):
        return ['list' if isinstance(value, list) else 'tuple', [_encode_value(item) for item in value]]
    raise TypeError('Unsupported player observation value.')


def _decode_value(node):
    if not isinstance(node, list) or not node or type(node[0]) is not str:
        raise ValueError('Malformed player observation scalar.')
    kind = node[0]
    if kind == 'null' and len(node) == 1:
        return None
    if kind in ('bool', 'int', 'str') and len(node) == 2:
        if type(node[1]) is not {'bool': bool, 'int': int, 'str': str}[kind]:
            raise ValueError('Malformed player observation scalar type.')
        return node[1]
    if kind == 'float64' and len(node) == 2:
        payload = bytes.fromhex(node[1])
        if len(payload) != 8:
            raise ValueError('Malformed player floating-point buffer.')
        return struct.unpack('!d', payload)[0]
    if kind in ('tuple', 'list') and len(node) == 2 and isinstance(node[1], list):
        if len(node[1]) > 51:
            raise ValueError('Player anatomical record exceeds its fixed landmark shape.')
        items = [_decode_value(item) for item in node[1]]
        return tuple(items) if kind == 'tuple' else items
    if kind in ('numpy_scalar', 'numpy_array') and len(node) == 4:
        dtype = np.dtype(node[1])
        shape = node[2]
        if (dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biuf'
                or not isinstance(shape, list) or len(shape) > 2
                or any(type(size) is not int or size < 0 for size in shape)
                or (kind == 'numpy_scalar' and shape)):
            raise ValueError('Unsafe player observation array metadata.')
        count = 1
        for dimension in shape:
            count *= dimension
        if count > 51:
            raise ValueError('Player observation array exceeds 17 anatomical landmarks.')
        payload = bytes.fromhex(node[3])
        if len(payload) != count * dtype.itemsize:
            raise ValueError('Player observation array shape and buffer disagree.')
        result = np.frombuffer(payload, dtype=dtype).reshape(shape).copy(order='C')
        return result[()] if kind == 'numpy_scalar' else result
    raise ValueError('Unknown private player observation encoding.')


def _validate_annotation(name, value):
    if name in ('ground_point_source', 'ground_point_quality'):
        if type(value) is not str:
            raise TypeError('Ground-point provenance must be explicit text.')
        return
    if name == 'ground_point_confidence':
        if value is not None and not _numeric(value):
            raise TypeError('Ground-point confidence must be a numeric scalar or null.')
        return
    if name == 'ground_point_px' and value is None:
        return
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise TypeError('Player anatomical coordinates require typed numeric points.')
    array = np.asarray(value)
    expected = (2,) if name == 'ground_point_px' else (17, 3)
    if (array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in 'biuf'
            or (array.shape != expected and not (name == 'keypoints_px' and array.shape == (0,)))):
        raise ValueError('Player annotations require a 2D ground point or 17 three-value keypoints.')


def _encode_box(box):
    # Importing the dataclass does not construct or load its neural model.
    from src.detection.yolo26_adapters import PoseBBox
    if type(box) not in (BBox, PoseBBox):
        raise TypeError('Player storage accepts only BBox and PoseBBox observations.')
    if set(vars(box)) - set(BOX_FIELDS) - set(ANNOTATIONS):
        raise TypeError('Unknown player attributes cannot be silently dropped.')
    for name in BOX_FIELDS[:5]:
        if not _numeric(getattr(box, name)):
            raise TypeError('Box coordinates and confidence require numeric scalars.')
    for name in ('class_id', 'track_id'):
        value = getattr(box, name)
        if not (name == 'track_id' and value is None) and not (type(value) is int or isinstance(value, np.integer)):
            raise TypeError('Detection class and track IDs require integers or a null track ID.')
    annotations = {}
    for name in ANNOTATIONS:
        if hasattr(box, name):
            value = getattr(box, name)
            _validate_annotation(name, value)
            annotations[name] = _encode_value(value)
    return {'kind': 'PoseBBox' if type(box) is PoseBBox else 'BBox',
            'box': {name: _encode_value(getattr(box, name)) for name in BOX_FIELDS},
            'annotations': annotations}


def _encode_frame(detections, maximum=4096):
    records = []
    iterator = iter(detections)
    try:
        for box in iterator:
            if len(records) >= maximum:
                raise ValueError('Player detection frame exceeds the configured observation bound.')
            records.append(_encode_box(box))
    except BaseException as error:
        try:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        except BaseException as cleanup_error:
            error.add_note(f'Player frame producer cleanup also failed: {cleanup_error}')
        raise
    else:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
    return json.dumps({'schemaVersion': 1, 'detections': records}, separators=(',', ':'), allow_nan=False)


def _decode_frame(payload, maximum=4096):
    from src.detection.yolo26_adapters import PoseBBox
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate typed player observation field.')
            value[key] = item
        return value
    value = json.loads(payload, object_pairs_hook=unique_object)
    if (not isinstance(value, dict) or set(value) != {'schemaVersion', 'detections'}
            or type(value['schemaVersion']) is not int or value['schemaVersion'] != 1
            or not isinstance(value['detections'], list) or len(value['detections']) > maximum):
        raise ValueError('Malformed player observation frame schema.')
    boxes = []
    for record in value['detections']:
        if (not isinstance(record, dict) or set(record) != {'kind', 'box', 'annotations'}
                or record['kind'] not in ('BBox', 'PoseBBox')
                or not isinstance(record['box'], dict) or set(record['box']) != set(BOX_FIELDS)
                or not isinstance(record['annotations'], dict) or set(record['annotations']) - set(ANNOTATIONS)
                or (record['kind'] == 'PoseBBox' and set(record['annotations']) != set(ANNOTATIONS))):
            raise ValueError('Malformed typed player detection record.')
        fields = {name: _decode_value(node) for name, node in record['box'].items()}
        box = (PoseBBox if record['kind'] == 'PoseBBox' else BBox)(**fields)
        for name, node in record['annotations'].items():
            setattr(box, name, _decode_value(node))
        # Reapply the same schema checks to decoded data rather than trusting a
        # scratch file or loading an arbitrary class named in its contents.
        _encode_box(box)
        boxes.append(box)
    return boxes


def _positive_integer(value, label):
    if isinstance(value, bool):
        raise ValueError(label + ' must be a positive integer.')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(label + ' must be a positive integer.') from error
    if value < 1:
        raise ValueError(label + ' must be a positive integer.')
    return value


class PlayerDetectionSequence(Sequence):
    """Indexed owned scratch records, retaining a closed diagnostic on failure.

    Normal context completion removes only this database and exact SQLite
    sidecars. Writes commit at a bounded cadence. Slices materialize only their
    requested records; callers should iterate for match-long operations.
    """
    def __init__(self, directory, *, transaction_size=256, max_detections_per_frame=4096):
        self.directory = Path(directory).resolve()
        self.transaction_size = _positive_integer(transaction_size, 'Transaction size')
        self.maximum = _positive_integer(max_detections_per_frame, 'Detection frame bound')
        self._path = self._connection = None
        self._length = self._pending = 0

    @property
    def path(self):
        return self._path

    def _open(self):
        if self._connection is None:
            raise RuntimeError('Player detection storage is not open.')
        return self._connection

    def _execute(self, sql, parameters=()):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.fetchone() if cursor.description is not None else cursor.rowcount
        finally:
            cursor.close()

    def __enter__(self):
        if self._path is not None:
            raise RuntimeError('Player detection storage cannot be reused.')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='player_detections_', suffix='.sqlite3', delete=False) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned player storage escaped its directory.')
            self._connection = sqlite3.connect(self._path)
            self._execute('PRAGMA journal_mode=DELETE')
            self._execute('PRAGMA cache_size=-2048')
            self._execute('PRAGMA temp_store=FILE')
            self._execute('PRAGMA mmap_size=0')
            self._execute('CREATE TABLE player_frames (frame_index INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            self._connection.commit()
            return self
        except BaseException as error:
            self._close_preserving(error)
            raise

    def __len__(self):
        return self._length

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return index

    def __getitem__(self, index):
        self._open()
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._length))]
        index = self._index(index)
        row = self._execute('SELECT payload FROM player_frames WHERE frame_index=?', (index,))
        if row is None:
            raise RuntimeError('Player detection frame is missing from owned storage.')
        return _decode_frame(row[0], self.maximum)

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def _mutated(self):
        self._pending += 1
        if self._pending >= self.transaction_size:
            self._open().commit()
            self._pending = 0

    def append(self, detections):
        payload = _encode_frame(detections, self.maximum)
        self._execute('INSERT INTO player_frames(frame_index,payload) VALUES(?,?)', (self._length, payload))
        self._length += 1
        self._mutated()

    def __setitem__(self, index, detections):
        if isinstance(index, slice):
            raise TypeError('Player frame replacement requires an integer index.')
        index = self._index(index)
        payload = _encode_frame(detections, self.maximum)
        if self._execute('UPDATE player_frames SET payload=? WHERE frame_index=?', (payload, index)) != 1:
            raise RuntimeError('Player detection frame is missing from owned storage.')
        self._mutated()

    def extend(self, frames):
        iterator = iter(frames)
        try:
            for detections in iterator:
                self.append(detections)
        except BaseException as error:
            try:
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
            except BaseException as cleanup_error:
                error.add_note(f'Player producer cleanup also failed: {cleanup_error}')
            raise
        else:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()

    def _close_preserving(self, error):
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                try:
                    connection.commit()
                finally:
                    connection.close()
            except BaseException as cleanup_error:
                error.add_note(f'Player database cleanup also failed: {cleanup_error}')

    def __exit__(self, error_type, error, _traceback):
        if error_type is not None:
            self._close_preserving(error)
            return False
        try:
            self._open().commit()
            self._connection.close()
            self._connection = None
            for suffix in ('-journal', '-wal', '-shm', ''):
                owned = Path(str(self._path) + suffix)
                if owned.resolve().parent != self.directory:
                    raise ValueError('Owned player cleanup escaped its directory.')
                owned.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            self._close_preserving(cleanup_error)
            raise
        return False
