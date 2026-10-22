"""Caller-owned bounded forecast history and append-only point identities.

This is temporary per-revision software storage, not a durable inference
checkpoint. It imports no numerical or inference runtime. Reads detach one
plain record; the owner must outlive every history/index view and iterator.
"""
from collections.abc import MutableMapping, Sequence
import json
import math
import operator
from pathlib import Path
import sqlite3
import tempfile


MAX_HISTORY_BYTES = 64 * 1024
MAX_POINT_ID_CHARACTERS = 200


def _require(condition, message):
    if not condition:
        raise ValueError('Forecast storage: ' + message)


def _directory(directory):
    path = Path(directory).absolute()
    _require('..' not in path.parts, 'directory parent traversal is forbidden')
    for current in (path, *path.parents):
        _require(not current.is_symlink() and not getattr(current, 'is_junction', lambda: False)(),
                 'directory links and junctions are forbidden')
    _require(path.is_dir(), 'caller-owned directory must already exist')
    return path.resolve()


def _history_payload(value):
    _require(type(value) is dict, 'history requires a plain JSON object')
    # Traverse one item at a time, with a lower size bound, so an oversized
    # supplied object is not copied into another unbounded Python tree first.
    active, minimum_bytes = set(), 0

    def size(amount):
        nonlocal minimum_bytes
        minimum_bytes += amount
        _require(minimum_bytes <= MAX_HISTORY_BYTES, 'history object exceeds 64 KiB')

    def visit(item):
        kind = type(item)
        if kind is str:
            size(len(item) + 2)
        elif item is None or kind is bool:
            size(4 if item is None or item is True else 5)
        elif kind is int:
            size(len(str(item)))
        elif kind is float:
            _require(math.isfinite(item), 'history numbers must be finite')
            size(len(repr(item)))
        elif kind in (dict, list):
            identity = id(item)
            _require(identity not in active, 'history cannot contain a circular object')
            active.add(identity)
            size(2)
            try:
                if kind is dict:
                    for index, (key, child) in enumerate(item.items()):
                        _require(type(key) is str, 'history object keys must be plain strings')
                        if index:
                            size(1)
                        visit(key)
                        size(1)
                        visit(child)
                else:
                    for index, child in enumerate(item):
                        if index:
                            size(1)
                        visit(child)
            finally:
                active.remove(identity)
        else:
            raise ValueError('Forecast storage: history accepts only plain JSON primitives')

    try:
        visit(value)
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        _require(len(payload.encode('utf-8')) <= MAX_HISTORY_BYTES, 'history object exceeds 64 KiB')
        return payload
    except (RecursionError, UnicodeError) as error:
        raise ValueError('Forecast storage: history requires encodable finite plain JSON') from error


def _point_id(value):
    _require(type(value) is str and bool(value.strip()) and len(value) <= MAX_POINT_ID_CHARACTERS,
             'point ID must be a nonblank plain string of at most 200 characters')
    return value


def _point_value(value, player_ids):
    _require(type(value) is tuple and len(value) == 3,
             'point value must be an exact (server, winner, timestamp) tuple')
    server, winner, timestamp = value
    _require(type(server) is int and type(winner) is int
             and server in player_ids and winner in player_ids,
             'point server and winner must be exact participant integers')
    finite = False
    if type(timestamp) in (int, float):
        try:
            finite = math.isfinite(timestamp) and timestamp >= 0
        except OverflowError:
            pass
    _require(finite, 'point timestamp must be a finite nonnegative integer or float')
    return value


class _History(Sequence):
    def __init__(self, owner, indices=None):
        self.owner, self.indices = owner, indices

    def __len__(self):
        self.owner._open()
        return self.owner._count('history') if self.indices is None else len(self.indices)

    def __getitem__(self, index):
        self.owner._open()
        if isinstance(index, slice):
            indices = range(len(self)) if self.indices is None else self.indices
            return _History(self.owner, indices[index])
        if isinstance(index, bool):
            raise TypeError('Forecast history index must be an integer')
        index = operator.index(index)
        indices = range(len(self)) if self.indices is None else self.indices
        position = indices[index]
        row = self.owner._one('SELECT payload FROM history WHERE position=?', (position + 1,))
        if row is None:
            raise RuntimeError('Owned forecast history record is missing')
        value = json.loads(row[0])
        _history_payload(value)
        return value

    def __iter__(self):
        if self.indices is not None:
            for index in range(len(self)):
                yield self[index]
            return
        rows = self.owner._rows('SELECT payload FROM history ORDER BY position')
        try:
            for (payload,) in rows:
                value = json.loads(payload)
                _history_payload(value)
                yield value
        finally:
            rows.close()

    def append(self, value):
        if self.indices is not None:
            raise TypeError('Append only to the owning forecast history')
        self.owner._write()
        payload = _history_payload(value)
        self.owner._execute('INSERT INTO history(payload) VALUES(?)', (payload,))
        self.owner._changed()


class _PointIndex(MutableMapping):
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return self.owner._count('point_index')

    def __getitem__(self, key):
        self.owner._open()
        key = _point_id(key)
        row = self.owner._one('SELECT payload FROM point_index WHERE point_id=?', (key,))
        if row is None:
            raise KeyError(key)
        return _point_value(tuple(json.loads(row[0])), self.owner.player_ids)

    def __contains__(self, key):
        self.owner._open()
        key = _point_id(key)
        return self.owner._one('SELECT 1 FROM point_index WHERE point_id=?', (key,)) is not None

    def __setitem__(self, key, value):
        self.owner._write()
        key, value = _point_id(key), _point_value(value, self.owner.player_ids)
        existing = self.owner._one('SELECT payload FROM point_index WHERE point_id=?', (key,))
        if existing is not None:
            _require(tuple(json.loads(existing[0])) == value,
                     'a point ID cannot be reused for a different outcome')
            return
        self.owner._execute('INSERT INTO point_index(point_id,payload) VALUES(?,?)',
                            (key, json.dumps(value, allow_nan=False, separators=(',', ':'))))
        self.owner._changed()

    def __iter__(self):
        rows = self.owner._rows('SELECT point_id FROM point_index ORDER BY rowid')
        try:
            for (point_id,) in rows:
                yield point_id
        finally:
            rows.close()

    def __delitem__(self, key):
        self.owner._open()
        raise TypeError('Forecast point identities are append-only')

    def clear(self):
        self.owner._open()
        raise TypeError('Forecast point identities are append-only')

    def pop(self, key, *default):
        self.owner._open()
        raise TypeError('Forecast point identities are append-only')

    def popitem(self):
        self.owner._open()
        raise TypeError('Forecast point identities are append-only')


class ForecastSequence:
    """One SQLite owner for causal history and strict point deduplication.

    ``point_index`` is a MutableMapping suitable for the predictor's explicit
    point-index seam. It permits new inserts and identical retries, never
    deletion or conflicting replacement. ``seal`` freezes both collections.
    No Python history or point-ID cache is retained.
    """
    def __init__(self, directory, *, player_ids=(1, 2), transaction_size=256):
        _require(type(player_ids) in (tuple, list) and len(player_ids) == 2
                 and all(type(player) is int and player > 0 for player in player_ids)
                 and player_ids[0] != player_ids[1], 'two distinct positive participant integers are required')
        _require(type(transaction_size) is int and 1 <= transaction_size <= 256,
                 'transaction size must be an integer from 1 through 256')
        self.directory = Path(directory).absolute()
        self.player_ids, self.transaction_size = tuple(player_ids), transaction_size
        self.path = self.connection = None
        self.pending, self.sealed = 0, False
        self._entered, self._failure = False, None
        self._readers = set()
        self.history, self.point_index = _History(self), _PointIndex(self)

    def _open(self):
        if self.connection is None:
            raise RuntimeError('Forecast storage is not open')
        if self._failure is not None:
            raise RuntimeError('Forecast storage is invalid after transaction failure: ' + self._failure)
        return self.connection

    def _write(self):
        self._open()
        _require(not self.sealed, 'a sealed forecast revision cannot change')

    def _one(self, sql, parameters=()):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    def _execute(self, sql, parameters=()):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.lastrowid
        finally:
            cursor.close()

    def _count(self, table):
        return self._one('SELECT COUNT(*) FROM ' + table)[0]

    def _rows(self, sql):
        cursor = self._open().execute(sql)
        self._readers.add(cursor)
        try:
            while True:
                self._open()
                row = cursor.fetchone()
                if row is None:
                    return
                yield row
        finally:
            if cursor in self._readers:
                self._readers.remove(cursor)
                cursor.close()

    def __enter__(self):
        _require(not self._entered, 'forecast storage cannot be reused')
        self._entered = True
        try:
            self.directory = _directory(self.directory)
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='forecast_sequence_',
                                             suffix='.sqlite3', delete=False) as temporary:
                self.path = Path(temporary.name).resolve()
            _require(self.path.parent == self.directory, 'owned forecast path escaped its directory')
            self.connection = sqlite3.connect(self.path)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._one('PRAGMA ' + pragma)
            self._execute('CREATE TABLE history(position INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            self._execute('CREATE TABLE point_index(point_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            self.flush()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def _changed(self):
        self.pending += 1
        if self.pending >= self.transaction_size:
            self.flush()

    def flush(self):
        connection = self._open()
        try:
            connection.commit()
        except BaseException as error:
            self._failure = type(error).__name__ + ': ' + str(error)
            try:
                connection.rollback()
            except BaseException as cleanup:
                error.add_note('Forecast transaction rollback also failed: ' + str(cleanup))
            raise
        finally:
            self.pending = 0

    def seal(self):
        self._open()
        self.flush()
        self.sealed = True
        return self

    def _dispose(self, primary=None):
        failures = []
        for cursor in tuple(self._readers):
            try:
                cursor.close()
            except BaseException as error:
                failures.append(error)
            finally:
                self._readers.discard(cursor)
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                failures.append(error)
        if self.path is not None:
            for suffix in ('-journal', '-wal', '-shm', ''):
                path = Path(str(self.path) + suffix)
                try:
                    _require(not path.is_symlink() and path.resolve().parent == self.directory,
                             'owned forecast cleanup escaped its directory')
                    for parent in (self.directory, *self.directory.parents):
                        _require(not parent.is_symlink()
                                 and not getattr(parent, 'is_junction', lambda: False)(),
                                 'owned forecast cleanup parent became a link')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(error)
        if primary is not None:
            for error in failures:
                primary.add_note('Forecast owner cleanup also failed: ' + str(error))
        elif failures:
            primary = failures.pop(0)
            for error in failures:
                primary.add_note('Forecast owner cleanup also failed: ' + str(error))
            raise primary

    def __exit__(self, kind, error, traceback):
        self._dispose(error)
        return False
