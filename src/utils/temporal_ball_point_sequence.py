"""Owned, bounded scratch storage for typed temporal ball point snapshots.

This stores offline results, not a temporal filter checkpoint. Source PTS and
camera/identity review remain separate evidence; a stored point timestamp alone
does not establish native timing, metric qualification or resumable inference.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
import json
import operator
from pathlib import Path
import sqlite3
import struct
import tempfile
from typing import overload

import numpy as np

from src.tracking.temporal_ball_tracker import BallState, TemporalBallPoint


SQLITE_CACHE_KIB = 2048
MAX_TRANSACTION_SIZE = 256
_NUMBER_FIELDS = ('timestamp_seconds', 'x_px', 'y_px', 'court_x_m', 'court_y_m',
                  'confidence', 'speed_kmh')
_POINT_FIELDS = frozenset(('frame_index', 'state', 'source', 'velocity_px_per_sec', *_NUMBER_FIELDS))


def _encode_number(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        dtype = value.dtype
        if dtype.fields is not None or dtype.hasobject or dtype.kind not in 'iuf' or dtype.itemsize > 8:
            raise TypeError('Ball point numeric fields require real numeric scalars')
        return ['numpy', dtype.str, value.tobytes().hex()]
    if type(value) is int:
        return ['int', value]
    if type(value) is float:
        # Preserve signed zero, subnormals and IEEE NaN payloads without emitting
        # nonfinite JSON tokens or allowing SQLite to normalize floats to NULL.
        return ['float64', struct.pack('!d', value).hex()]
    raise TypeError('Ball point numeric fields require real numeric scalars')


def _decode_number(node):
    if node is None:
        return None
    if not isinstance(node, list) or not node:
        raise ValueError('Malformed ball point numeric field')
    if len(node) == 2 and node[0] == 'int' and type(node[1]) is int:
        return node[1]
    if len(node) == 2 and node[0] == 'float64' and isinstance(node[1], str):
        payload = bytes.fromhex(node[1])
        if len(payload) == 8:
            return struct.unpack('!d', payload)[0]
    if len(node) == 3 and node[0] == 'numpy' and isinstance(node[1], str) and isinstance(node[2], str):
        dtype = np.dtype(node[1])
        if dtype.fields is None and not dtype.hasobject and dtype.kind in 'iuf' and dtype.itemsize <= 8:
            payload = bytes.fromhex(node[2])
            if len(payload) == dtype.itemsize:
                return np.frombuffer(payload, dtype=dtype).copy()[0]
    raise ValueError('Malformed ball point numeric field')


def _encode_point(point: TemporalBallPoint) -> str:
    if not isinstance(point, TemporalBallPoint):
        raise TypeError('Temporal ball storage accepts TemporalBallPoint records')
    if isinstance(point.frame_index, (bool, np.bool_)):
        raise TypeError('Ball point frame index requires an integer')
    try:
        frame_index = operator.index(point.frame_index)
    except TypeError as error:
        raise TypeError('Ball point frame index requires an integer') from error
    if point.state is not None and not isinstance(point.state, BallState):
        raise TypeError('Ball point state requires BallState or None')
    if not isinstance(point.source, str):
        raise TypeError('Ball point source requires a string')
    if point.timestamp_seconds is None:
        raise TypeError('Ball point timestamp is required')
    velocity = point.velocity_px_per_sec
    if velocity is not None and (not isinstance(velocity, tuple) or len(velocity) != 2
                                 or any(value is None for value in velocity)):
        raise TypeError('Ball point velocity requires a two-number tuple or None')
    record = {name: _encode_number(getattr(point, name)) for name in _NUMBER_FIELDS}
    record.update(frame_index=frame_index, state=None if point.state is None else point.state.value,
                  source=point.source,
                  velocity_px_per_sec=None if velocity is None else [_encode_number(value) for value in velocity])
    return json.dumps({'schema_version': 1, 'point': record}, separators=(',', ':'), allow_nan=False)


def _decode_point(payload: str) -> TemporalBallPoint:
    document = json.loads(payload)
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'point'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
        raise ValueError('Unsupported temporal ball point schema')
    record = document['point']
    if (not isinstance(record, dict) or set(record) != _POINT_FIELDS
            or type(record['frame_index']) is not int or not isinstance(record['source'], str)):
        raise ValueError('Malformed temporal ball point record')
    values = {name: _decode_number(record[name]) for name in _NUMBER_FIELDS}
    if values['timestamp_seconds'] is None:
        raise ValueError('Missing temporal ball point timestamp')
    velocity = record['velocity_px_per_sec']
    if velocity is not None:
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise ValueError('Malformed temporal ball point velocity')
        velocity = tuple(_decode_number(value) for value in velocity)
        if any(value is None for value in velocity):
            raise ValueError('Missing temporal ball point velocity component')
    return TemporalBallPoint(frame_index=record['frame_index'],
                             state=None if record['state'] is None else BallState(record['state']),
                             source=record['source'], velocity_px_per_sec=velocity, **values)


class _PointView(Sequence[TemporalBallPoint]):
    """Range-backed slice; retains neither a cursor nor decoded point cache."""

    def __init__(self, owner: TemporalBallPointSequence, indices: range):
        self._owner, self._indices = owner, indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index):
        selected = self._indices[index]
        return _PointView(self._owner, selected) if isinstance(selected, range) else self._owner[selected]

    def __setitem__(self, index: int, point: TemporalBallPoint) -> None:
        if isinstance(index, slice):
            raise TypeError('Write back one ball point by index')
        self._owner[self._indices[index]] = point


class TemporalBallPointSequence(Sequence[TemporalBallPoint]):
    """Typed indexed disk snapshots with a 2 MiB cache and at most 256 writes.

    Reads and iteration return independent point snapshots. Attribute mutation
    never writes through: use ``point = points[i]; point.x_px = x; points[i] =
    point``. Slices are lazy writable views; slice assignment is unsupported.
    ``extend`` consumes and closes a producer one point at a time. The sequence
    retains no point cache or whole-match list.

    The caller supplies an existing output directory. Successful context exit
    deletes only this exclusively created database and its exact SQLite
    sidecars. Failure closes the connection, attempts to commit accepted writes
    and retains scratch for diagnosis. A retained file is not a supported
    pipeline resume checkpoint. Context instances cannot be reopened or reused.
    """

    def __init__(self, directory: str | Path, *, transaction_size: int = MAX_TRANSACTION_SIZE):
        try:
            value = operator.index(transaction_size)
        except TypeError as error:
            raise ValueError('Transaction size must be an integer from 1 through 256') from error
        if isinstance(transaction_size, (bool, np.bool_)) or not 1 <= value <= MAX_TRANSACTION_SIZE:
            raise ValueError('Transaction size must be an integer from 1 through 256')
        self.directory = Path(directory).resolve()
        self.transaction_size = value
        self._path: Path | None = None
        self._connection: sqlite3.Connection | None = None
        self._length = 0
        self._pending_mutations = 0

    @property
    def path(self) -> Path | None:
        """Owned scratch path, retained for diagnostic discovery after closure."""
        return self._path

    def __enter__(self) -> TemporalBallPointSequence:
        if self._path is not None:
            raise RuntimeError('Temporal ball storage cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='temporal_ball_points_',
                                             suffix='.sqlite3', delete=False) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned temporal ball storage escaped its directory')
            self._connection = sqlite3.connect(self._path)
            for pragma in ('journal_mode=DELETE', f'cache_size=-{SQLITE_CACHE_KIB}',
                           'temp_store=FILE', 'mmap_size=0'):
                self._connection.execute(f'PRAGMA {pragma}').close()
            self._connection.execute('CREATE TABLE temporal_points '
                                     '(position INTEGER PRIMARY KEY, payload TEXT NOT NULL)').close()
            self._connection.commit()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def _require_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError('Temporal ball storage is not open')
        return self._connection

    def __len__(self) -> int:
        return self._length

    def _index(self, index: int) -> int:
        index = operator.index(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return index

    @overload
    def __getitem__(self, index: int) -> TemporalBallPoint: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[TemporalBallPoint]: ...

    def __getitem__(self, index):
        connection = self._require_open()
        if isinstance(index, slice):
            return _PointView(self, range(*index.indices(self._length)))
        position = self._index(index)
        cursor = connection.execute('SELECT payload FROM temporal_points WHERE position=?', (position,))
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError(f'Temporal ball point {position} is missing from owned storage')
        return _decode_point(row[0])

    def __iter__(self) -> Iterator[TemporalBallPoint]:
        for index in range(self._length):
            yield self[index]

    def flush(self) -> None:
        """Explicitly persist accepted snapshots; does not checkpoint a tracker."""
        self._require_open().commit()
        self._pending_mutations = 0

    def _record_mutation(self) -> None:
        self._pending_mutations += 1
        if self._pending_mutations >= self.transaction_size:
            self.flush()

    def append(self, point: TemporalBallPoint) -> None:
        connection = self._require_open()
        payload = _encode_point(point)
        connection.execute('INSERT INTO temporal_points(position,payload) VALUES(?,?)',
                           (self._length, payload)).close()
        self._length += 1
        self._record_mutation()

    def extend(self, points: Iterable[TemporalBallPoint]) -> None:
        iterator = iter(points)
        try:
            for point in iterator:
                self.append(point)
        except BaseException as error:
            try:
                close = getattr(iterator, 'close', None)
                if close is not None:
                    close()
            except BaseException as close_error:
                error.add_note(f'Temporal ball producer cleanup also failed: {close_error}')
            raise
        else:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()

    def __setitem__(self, index: int, point: TemporalBallPoint) -> None:
        connection = self._require_open()
        if isinstance(index, slice):
            raise TypeError('Write back one ball point by index')
        position = self._index(index)
        payload = _encode_point(point)
        cursor = connection.execute('UPDATE temporal_points SET payload=? WHERE position=?', (payload, position))
        try:
            if cursor.rowcount != 1:
                raise RuntimeError(f'Temporal ball point {position} is missing from owned storage')
        finally:
            cursor.close()
        self._record_mutation()

    def _dispose(self, primary: BaseException | None = None) -> None:
        connection, self._connection = self._connection, None
        failures = []
        if connection is not None:
            try:
                connection.commit()
            except BaseException as error:
                failures.append(('commit', error))
            try:
                connection.close()
            except BaseException as error:
                failures.append(('close', error))
        if primary is None and not failures and self._path is not None:
            try:
                for suffix in ('-journal', '-wal', '-shm', ''):
                    owned = Path(str(self._path) + suffix)
                    if owned.resolve().parent != self.directory:
                        raise ValueError('Owned temporal ball cleanup escaped its directory')
                    owned.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(('unlink', error))
        if primary is not None:
            for stage, error in failures:
                primary.add_note(f'Temporal ball storage {stage} also failed: {error}')
        elif failures:
            _, first = failures[0]
            for stage, error in failures[1:]:
                first.add_note(f'Temporal ball storage {stage} also failed: {error}')
            raise first

    def __exit__(self, error_type, error, traceback) -> bool:
        self._dispose(error)
        return False
