"""Caller-owned disk snapshots for offline shot linking and frame badges.

This is temporary analysis storage, not a durable inference checkpoint. The
owner must remain open through all consumers. Mutations require indexed writes.
"""
from collections.abc import Sequence
from dataclasses import fields
import json
import math
import operator
from pathlib import Path
import sqlite3
import tempfile

from src.shot_analysis.shot_types import (ShotEventEvidence, ShotType, ShotDirection,
    ShotClassificationSource, ShotOutcome, CourtZone3x3)
from src.utils.identity_frame_sequence import _encode, _decode
from src.utils.mapped_sequence_view import MappedSequenceView


_ENUM_FIELDS = {'shot_type': ShotType, 'direction': ShotDirection,
                'classification_source': ShotClassificationSource, 'outcome': ShotOutcome,
                'landing_zone': CourtZone3x3}


def _encode_shot(value):
    if not isinstance(value, ShotEventEvidence):
        raise TypeError('Shot storage requires typed ShotEventEvidence')
    record = {item.name: getattr(value, item.name) for item in fields(ShotEventEvidence)}
    for name, kind in _ENUM_FIELDS.items():
        item = record[name]
        if item is None and name == 'landing_zone':
            continue
        if not isinstance(item, kind):
            raise TypeError(f'Shot storage requires declared {name} enum semantics')
        record[name] = item.value
    return json.dumps({'schema_version': 1, 'record': _encode(record)}, separators=(',', ':'), allow_nan=False)


def _decode_shot(payload):
    envelope = json.loads(payload)
    if (not isinstance(envelope, dict) or set(envelope) != {'schema_version', 'record'}
            or type(envelope['schema_version']) is not int or envelope['schema_version'] != 1):
        raise ValueError('Malformed typed shot envelope')
    record = _decode(envelope['record'])
    if not isinstance(record, dict) or set(record) != {item.name for item in fields(ShotEventEvidence)}:
        raise ValueError('Malformed typed shot fields')
    for name, kind in _ENUM_FIELDS.items():
        if record[name] is not None or name != 'landing_zone':
            record[name] = kind(record[name])
    value = ShotEventEvidence(**record)
    _encode_shot(value)
    return value


class ShotEvidenceSequence(Sequence):
    """Typed snapshots with a 2 MiB SQLite cache and at most 256 pending writes."""
    def __init__(self, directory, *, transaction_size=256):
        if isinstance(transaction_size, bool):
            raise ValueError('Shot transaction size must be an integer from 1 through 256')
        transaction_size = operator.index(transaction_size)
        if not 1 <= transaction_size <= 256:
            raise ValueError('Shot transaction size must be an integer from 1 through 256')
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self._path = self._connection = None
        self._length = self._pending = 0
        self._sealed = False

    @property
    def path(self):
        return self._path

    def __len__(self):
        return self._length

    def _open(self):
        if self._connection is None:
            raise RuntimeError('Shot evidence storage is not open')
        return self._connection

    def _execute(self, sql, parameters=(), *, fetch=False):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.fetchone() if fetch else None
        finally:
            cursor.close()

    def __enter__(self):
        if self._path is not None:
            raise RuntimeError('Shot evidence storage cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='shot_evidence_',
                                             suffix='.sqlite3', delete=False) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned shot storage escaped its directory')
            self._connection = sqlite3.connect(self._path)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._execute(f'PRAGMA {pragma}', fetch=True)
            self._execute('CREATE TABLE shots(position INTEGER PRIMARY KEY, frame INTEGER NOT NULL, payload TEXT NOT NULL, '
                          'is_dead INTEGER NOT NULL, has_landing INTEGER NOT NULL, shot_type TEXT NOT NULL)')
            self._execute('CREATE INDEX shot_frame_order ON shots(frame,position)')
            self._execute('CREATE INDEX shot_live_order ON shots(is_dead,position)')
            self._execute('CREATE INDEX shot_landing_order ON shots(has_landing,position)')
            self.flush()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def flush(self):
        self._open().commit()
        self._pending = 0

    def seal(self):
        """Freeze snapshots before exposing analytics views; exit still owns cleanup."""
        self.flush()
        self._sealed = True
        return self

    def select(self, *, live_only=False, has_landing=None):
        self._open()
        if not self._sealed:
            raise ValueError('Analytics views require a sealed shot evidence owner')
        if type(live_only) is not bool or has_landing is not None and type(has_landing) is not bool:
            raise ValueError('Shot selection flags must be explicit booleans')
        clauses, parameters = [], []
        if live_only:
            clauses.append('is_dead=0')
        if has_landing is not None:
            clauses.append('has_landing=?')
            parameters.append(int(has_landing))
        return ShotSelection(self, ' AND '.join(clauses) or '1', tuple(parameters))

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _index: value)[index]
        row = self._execute('SELECT payload FROM shots WHERE position=?', (self._index(index),), fetch=True)
        if row is None:
            raise RuntimeError('Owned shot evidence record is missing')
        return _decode_shot(row[0])

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _write(self, index, value, *, update=False):
        self._open()
        if self._sealed:
            raise ValueError('A sealed shot evidence owner cannot be changed')
        payload = _encode_shot(value)
        if isinstance(value.frame_index, bool):
            raise ValueError('Shot frame must be a nonnegative integer')
        frame = operator.index(value.frame_index)
        if frame < 0:
            raise ValueError('Shot frame must be a nonnegative integer')
        timestamp = float(value.timestamp_s)
        if isinstance(value.timestamp_s, bool) or not math.isfinite(timestamp) or timestamp != value.timestamp_s:
            raise ValueError('Shot timestamp must remain finite exactly representable binary64')
        selectors = (int(bool(value.is_dead_ball)), int(value.landing_court_position_m is not None), value.shot_type.value)
        if update:
            self._execute('UPDATE shots SET frame=?,payload=?,is_dead=?,has_landing=?,shot_type=? WHERE position=?',
                          (frame, payload, *selectors, index))
        else:
            self._execute('INSERT INTO shots VALUES (?,?,?,?,?,?)', (index, frame, payload, *selectors))
            self._length += 1
        self._pending += 1
        if self._pending >= self.transaction_size:
            self.flush()

    def append(self, value):
        self._write(len(self), value)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError('Write back one shot evidence snapshot by index')
        self._write(self._index(index), value, update=True)

    def badges_by_frame(self, lower, upper):
        """Legacy dict semantics: last record per frame, first insertion order."""
        cursor = self._open().execute(
            'SELECT latest.payload FROM shots AS latest JOIN '
            '(SELECT frame,MIN(position) AS first,MAX(position) AS last FROM shots '
            'WHERE frame>=? AND frame<=? GROUP BY frame) AS selected '
            'ON latest.position=selected.last ORDER BY selected.first',
            (operator.index(lower), operator.index(upper)))
        try:
            for (payload,) in cursor:
                yield _decode_shot(payload)
        finally:
            if self._connection is not None:
                cursor.close()

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
                path = Path(str(self._path) + suffix)
                try:
                    if path.is_symlink() or path.resolve().parent != self.directory:
                        raise ValueError('Owned shot cleanup escaped its directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(('cleanup', error))
        if primary is None and failures:
            _, primary = failures.pop(0)
            for stage, error in failures:
                primary.add_note(f'Shot storage {stage} also failed: {error}')
            raise primary
        if primary is not None:
            for stage, error in failures:
                primary.add_note(f'Shot storage {stage} also failed: {error}')

    def __exit__(self, error_type, error, traceback):
        self._dispose(error)
        return False


class ShotSelection(Sequence):
    """Stable ordered views over the sealed caller-owned source, without row caches."""
    def __init__(self, owner, where, parameters):
        self.owner, self.where, self.parameters = owner, where, parameters

    def __len__(self):
        return self.owner._execute('SELECT COUNT(*) FROM shots WHERE ' + self.where,
                                   self.parameters, fetch=True)[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _index: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.owner._execute('SELECT payload FROM shots WHERE ' + self.where +
            ' ORDER BY position LIMIT 1 OFFSET ?', (*self.parameters, index), fetch=True)
        if row is None:
            raise RuntimeError('Sealed shot selection record is missing')
        return _decode_shot(row[0])

    def __iter__(self):
        cursor = self.owner._open().execute('SELECT payload FROM shots WHERE ' + self.where +
                                           ' ORDER BY position', self.parameters)
        try:
            for (payload,) in cursor:
                yield _decode_shot(payload)
        finally:
            if self.owner._connection is not None:
                cursor.close()

    def field(self, name):
        if name not in {item.name for item in fields(ShotEventEvidence)}:
            raise ValueError('Select a declared shot evidence field')
        return ShotFieldView(self, name)


class ShotFieldView(Sequence):
    def __init__(self, source, name):
        self.source, self.name = source, name

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _index: value)[index]
        return getattr(self.source[index], self.name)

    def __iter__(self):
        iterator = iter(self.source)
        try:
            for value in iterator:
                yield getattr(value, self.name)
        finally:
            iterator.close()


def require_shot_result_owner(shots, owner):
    """Opt-in analytics must identify their exact open stable source explicitly."""
    if not isinstance(owner, ShotEvidenceSequence) or shots is not owner:
        raise ValueError('Native analytics require the exact caller-owned shot source alias')
    owner._open()
    if not owner._sealed:
        raise ValueError('Native analytics require a sealed shot evidence owner')
    return owner
