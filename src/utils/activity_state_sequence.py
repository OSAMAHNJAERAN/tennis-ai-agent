"""Caller-owned, immutable activity snapshots on bounded temporary SQLite.

This stores classifier output only. It never establishes reviewed match
activity, event authority, source eligibility or measurement qualification.
"""
from collections.abc import Sequence
import json
import math
import operator
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from src.utils.identity_frame_sequence import _decode, _encode


def _state_type():
    # Delayed to keep the classifier free to accept this optional output owner.
    from src.events.activity_state import VisualActivityState
    return VisualActivityState


def _finite(value):
    if isinstance(value, (np.ndarray, np.generic)):
        if not np.isfinite(value).all():
            raise ValueError('Activity evidence must contain finite numeric values')
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Activity evidence must contain finite numeric values')
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite(key)
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _payload(evidence):
    if not isinstance(evidence, dict):
        raise TypeError('Activity evidence must be a dictionary')
    encoded = _encode(evidence)
    _finite(evidence)
    return json.dumps({'schema_version': 1, 'evidence': encoded}, separators=(',', ':'), allow_nan=False)


def _evidence(payload):
    document = json.loads(payload)
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'evidence'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1):
        raise ValueError('Malformed private activity evidence')
    value = _decode(document['evidence'])
    _payload(value)
    return value


class _ActivityView(Sequence):
    __slots__ = ('_owner', '_field', '_indices')

    def __init__(self, owner, field, indices=None):
        self._owner, self._field, self._indices = owner, field, indices

    def __len__(self):
        self._owner._readable()
        return self._owner._length if self._indices is None else len(self._indices)

    def __getitem__(self, index):
        self._owner._readable()
        indices = range(self._owner._length) if self._indices is None else self._indices
        if isinstance(index, slice):
            return type(self)(self._owner, self._field, indices[index])
        return self._owner._read(indices[operator.index(index)], self._field)

    def __iter__(self):
        self._owner._readable()
        indices = range(self._owner._length) if self._indices is None else self._indices
        for index in indices:
            yield self._owner._read(index, self._field)


class ActivityStateSequence:
    """Append once, seal once, then read detached states/evidence lazily.

    The owner must outlive all views. No row or decoded evidence is cached.
    A sealed frontier maps UNKNOWN to PRE_PLAY without rewriting the prefix.
    Any storage failure makes the owner unusable, including uncertain commits.
    """
    def __init__(self, directory, *, transaction_size=256):
        if isinstance(transaction_size, (bool, np.bool_)):
            raise ValueError('Activity transaction size must be an integer from 1 through 256')
        try:
            transaction_size = operator.index(transaction_size)
        except TypeError as error:
            raise ValueError('Activity transaction size must be an integer from 1 through 256') from error
        if not 1 <= transaction_size <= 256:
            raise ValueError('Activity transaction size must be an integer from 1 through 256')
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self._path = self._connection = self._failure = None
        self._entered = self._sealed = False
        self._length = self._pending = self._preplay_before = 0
        self.states, self.evidence = _ActivityView(self, 'state'), _ActivityView(self, 'evidence')

    @property
    def path(self):
        return self._path

    def _open(self):
        if self._failure is not None:
            raise RuntimeError('Activity storage cannot continue after a storage failure') from self._failure
        if self._connection is None:
            raise RuntimeError('Activity storage is not open')
        return self._connection

    def _readable(self):
        self._open()
        if not self._sealed:
            raise RuntimeError('Seal activity storage before reading its views')

    def __len__(self):
        self._open()
        return self._length

    def _execute(self, sql, args=(), *, one=False):
        cursor, primary = None, None
        try:
            cursor = self._open().execute(sql, args)
            return cursor.fetchone() if one else None
        except BaseException as error:
            self._failure = primary = error
            raise
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except BaseException as cleanup:
                    self._failure = primary or cleanup
                    if primary is None:
                        raise
                    primary.add_note('Activity cursor cleanup also failed: '+str(cleanup))

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Activity storage cannot be reused')
        self._entered = True
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='activity_states_',
                                             suffix='.sqlite3', delete=False) as stream:
                self._path = Path(stream.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned activity storage escaped its directory')
            self._connection = sqlite3.connect(self._path)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._execute('PRAGMA '+pragma, one=True)
            self._execute('CREATE TABLE activity(position INTEGER PRIMARY KEY,state TEXT NOT NULL,evidence TEXT NOT NULL)')
            self._flush()
            return self
        except BaseException as error:
            self._failure = error
            self._dispose(error)
            raise

    def require_empty(self):
        self._open()
        if self._sealed or self._length:
            raise ValueError('Activity output must be fresh, empty and unsealed')

    def _flush(self):
        try:
            self._open().commit()
        except BaseException as error:
            self._failure = error
            raise
        self._pending = 0

    def append(self, state, evidence):
        self._open()
        if self._sealed:
            raise ValueError('Sealed activity output cannot change')
        if not isinstance(state, _state_type()):
            raise TypeError('Activity state must be VisualActivityState')
        payload = _payload(evidence)  # Complete safe finite validation before SQL.
        try:
            self._execute('INSERT INTO activity VALUES(?,?,?)', (self._length, state.value, payload))
            self._length += 1
            self._pending += 1
            if self._pending >= self.transaction_size:
                self._flush()
        except BaseException as error:
            self._failure = error
            raise

    def seal(self, preplay_before=0):
        self._open()
        if self._sealed:
            raise ValueError('Activity output can be sealed only once')
        if isinstance(preplay_before, (bool, np.bool_)):
            raise ValueError('Activity pre-play frontier must be an integer within the stored timeline')
        try:
            frontier = operator.index(preplay_before)
        except TypeError as error:
            raise ValueError('Activity pre-play frontier must be an integer within the stored timeline') from error
        if not 0 <= frontier <= self._length:
            raise ValueError('Activity pre-play frontier must be an integer within the stored timeline')
        self._flush()
        self._preplay_before, self._sealed = frontier, True
        return self

    def _read(self, index, field):
        self._readable()
        try:
            row = self._execute('SELECT state,evidence FROM activity WHERE position=?', (index,), one=True)
            if row is None:
                raise RuntimeError('Owned activity snapshot is missing')
            state = _state_type()(row[0])
            if field == 'state':
                return (_state_type().PRE_PLAY if index < self._preplay_before and state == _state_type().UNKNOWN else state)
            return _evidence(row[1])
        except BaseException as error:
            self._failure = error
            raise

    def _dispose(self, primary=None):
        failures = []
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                failures.append(('close', error))
        if self._path is not None:
            for suffix in ('-journal', '-wal', '-shm', ''):
                path = Path(str(self._path)+suffix)
                try:
                    if path.is_symlink() or path.resolve().parent != self.directory:
                        raise ValueError('Owned activity cleanup escaped its directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(('temporary-file cleanup', error))
        if primary is None and failures:
            _, primary = failures.pop(0)
            self._failure = primary
            for stage, error in failures:
                primary.add_note(f'Activity storage {stage} also failed: {error}')
            raise primary
        if primary is not None:
            self._failure = self._failure or primary
            for stage, error in failures:
                primary.add_note(f'Activity storage {stage} also failed: {error}')

    def __exit__(self, error_type, error, traceback):
        self._dispose(error)
        return False
