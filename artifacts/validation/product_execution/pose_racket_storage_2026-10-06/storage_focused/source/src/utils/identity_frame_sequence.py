"""Owned temporary frame observations with ordered integer player identities."""
from collections.abc import Sequence
import json
import operator
from pathlib import Path
import sqlite3
import struct
import tempfile

import numpy as np


def _array_node(value, scalar):
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in 'biufc':
        raise TypeError('Identity observations support only plain numeric NumPy values')
    return ['scalar' if scalar else 'array', array.dtype.str, list(array.shape), array.tobytes(order='C').hex()]


def _encode(value):
    # NumPy float64 can also satisfy isinstance(value, float); preserve its type
    # until the existing raw exporter decides how to sanitize that observation.
    if isinstance(value, np.ndarray):
        return _array_node(value, False)
    if isinstance(value, np.generic):
        return _array_node(value, True)
    if value is None:
        return ['null']
    if isinstance(value, bool):
        return ['bool', value]
    if isinstance(value, int):
        return ['int', value]
    if isinstance(value, float):
        return ['float', struct.pack('!d', value).hex()]
    if isinstance(value, str):
        return ['str', value]
    if isinstance(value, dict):
        return ['dict', [[_encode(key), _encode(item)] for key, item in value.items()]]
    if isinstance(value, (list, tuple)):
        return ['list' if isinstance(value, list) else 'tuple', [_encode(item) for item in value]]
    raise TypeError(f'Unsupported identity observation value: {type(value).__name__}')


def _decode(node):
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError('Malformed private identity observation')
    kind = node[0]
    if kind == 'null' and len(node) == 1:
        return None
    if kind in ('bool', 'int', 'str') and len(node) == 2:
        expected = {'bool': bool, 'int': int, 'str': str}[kind]
        if type(node[1]) is not expected:
            raise ValueError('Malformed private identity scalar')
        return node[1]
    if kind == 'float' and len(node) == 2:
        payload = bytes.fromhex(node[1])
        if len(payload) != 8:
            raise ValueError('Malformed private floating point value')
        return struct.unpack('!d', payload)[0]
    if kind in ('list', 'tuple') and len(node) == 2 and isinstance(node[1], list):
        values = [_decode(item) for item in node[1]]
        return values if kind == 'list' else tuple(values)
    if kind == 'dict' and len(node) == 2 and isinstance(node[1], list):
        values = {}
        for pair in node[1]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError('Malformed private identity dictionary')
            key, value = _decode(pair[0]), _decode(pair[1])
            if key in values:
                raise ValueError('Duplicate private identity dictionary key')
            values[key] = value
        return values
    if kind in ('array', 'scalar') and len(node) == 4:
        dtype = np.dtype(node[1])
        shape = node[2]
        if (dtype.hasobject or dtype.fields is not None or dtype.kind not in 'biufc'
                or not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                or (kind == 'scalar' and shape)):
            raise ValueError('Unsafe private identity array metadata')
        payload = bytes.fromhex(node[3])
        size = dtype.itemsize
        for dimension in shape:
            size *= dimension
        if size != len(payload):
            raise ValueError('Private identity array shape and buffer differ')
        result = np.frombuffer(payload, dtype=dtype).reshape(tuple(shape)).copy(order='C')
        return result[()] if kind == 'scalar' else result
    raise ValueError('Unknown private identity observation encoding')


def _frame(record):
    if not isinstance(record, dict) or any(isinstance(key, (bool, np.bool_))
            or not isinstance(key, (int, np.integer)) for key in record):
        raise TypeError('Frame observations require an integer identity dictionary')
    return record


class IdentityFrameSequence(Sequence):
    """Append frame snapshots without retaining the whole video's observations.

    Each read returns fresh ordered dictionaries, including integer identities,
    tuple/list distinctions, IEEE floats and plain numeric NumPy values. One
    private SQLite database uses a 2 MiB page cache, no mmap and bounded commit
    batches. Context exit removes only its exact database and sidecars on both
    success and failure; errors retain their original cause and cleanup notes.
    """

    def __init__(self, directory, *, transaction_size=256):
        if isinstance(transaction_size, bool):
            raise ValueError('Transaction size must be a positive integer')
        try:
            transaction_size = operator.index(transaction_size)
        except TypeError as error:
            raise ValueError('Transaction size must be a positive integer') from error
        if transaction_size <= 0:
            raise ValueError('Transaction size must be a positive integer')
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self._path = self._connection = None
        self._length = self._pending = 0

    @property
    def path(self):
        return self._path

    def __len__(self):
        return self._length

    def _open(self):
        if self._connection is None:
            raise RuntimeError('Identity frame storage is not open')
        return self._connection

    def _execute(self, sql, parameters=(), *, fetch=False):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.fetchone() if fetch else None
        finally:
            cursor.close()

    def __enter__(self):
        if self._path is not None:
            raise RuntimeError('Identity frame storage cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='identity_frames_', suffix='.sqlite3', delete=False) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned identity storage escaped its directory')
            self._connection = sqlite3.connect(self._path)
            self._execute('PRAGMA journal_mode=DELETE', fetch=True)
            self._execute('PRAGMA cache_size=-2048')
            self._execute('PRAGMA temp_store=FILE')
            self._execute('PRAGMA mmap_size=0', fetch=True)
            self._execute('CREATE TABLE identity_frames (frame_index INTEGER PRIMARY KEY, observation TEXT NOT NULL)')
            self._connection.commit()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def append(self, record):
        self._open()
        payload = json.dumps(_encode(_frame(record)), separators=(',', ':'), allow_nan=False)
        self._execute('INSERT INTO identity_frames(frame_index,observation) VALUES(?,?)', (self._length, payload))
        self._length += 1
        self._pending += 1
        if self._pending >= self.transaction_size:
            self._connection.commit()
            self._pending = 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._length))]
        index = operator.index(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        row = self._execute('SELECT observation FROM identity_frames WHERE frame_index=?', (index,), fetch=True)
        if row is None:
            raise RuntimeError(f'Owned identity frame {index} is missing')
        return _frame(_decode(json.loads(row[0])))

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def _dispose(self, primary=None):
        failures = []
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                if primary is None:
                    connection.commit()
            except BaseException as error:
                failures.append(('commit', error))
            finally:
                try:
                    connection.close()
                except BaseException as error:
                    failures.append(('close', error))
        if self._path is not None:
            for suffix in ('-journal', '-wal', '-shm', ''):
                path = Path(str(self._path) + suffix)
                try:
                    if path.is_symlink() or path.resolve().parent != self.directory:
                        raise ValueError('Owned identity cleanup escaped its directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(('temporary-file cleanup', error))
        if primary is None and failures:
            _, primary = failures.pop(0)
            for stage, error in failures:
                primary.add_note(f'Identity storage {stage} also failed: {error}')
            raise primary
        if primary is not None:
            for stage, error in failures:
                primary.add_note(f'Identity storage {stage} also failed: {error}')

    def __exit__(self, error_type, error, traceback):
        self._dispose(error)
        return False


class _SanitizedFrameArray(list):
    """The indented stdlib JSON encoder reads this list one frame at a time."""

    def __init__(self, frames, sanitize):
        self._frames, self._sanitize = frames, sanitize

    def __len__(self):
        return len(self._frames)

    def __iter__(self):
        for frame in self._frames:
            yield self._sanitize(frame)


def write_identity_frame_json(destination, metadata, array_key, frames, sanitize):
    """Keep established indent=2/native-newline/no-final-newline raw JSON bytes.

    Metadata is sanitized with the unchanged caller's converter. Each frame is
    converted only when the encoder requests it, preserving that converter's
    existing tuple and NumPy array behavior without materializing all frames.
    """
    if not isinstance(array_key, str) or array_key in metadata:
        raise ValueError('Array key must be absent from metadata')
    output = {**sanitize(metadata), array_key: _SanitizedFrameArray(frames, sanitize)}
    with open(destination, 'w', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
