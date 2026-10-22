"""Caller-owned typed event results, ranking and traces on bounded SQLite.

This is temporary offline analysis storage, not a resumable inference state.
Reads return independent snapshots; mutations require explicit indexed writes.
The owner must outlive every consumer and closes/removes its exact scratch file
and SQLite sidecars on successful, failed and cancelled context exits.
"""
from __future__ import annotations
from collections.abc import Sequence
from dataclasses import fields
import json
import math
import operator
from pathlib import Path
import sqlite3
import tempfile


def _record_types():
    # Delay this import to avoid a cycle with the detector's opt-in output seam.
    from src.events.event_detector import EventCandidate, EventType, TennisEvent
    return {'candidates': EventCandidate, 'events': TennisEvent}, EventType


def _encode_record(kind, value):
    from src.utils.identity_frame_sequence import _encode
    classes, event_type = _record_types()
    if kind == 'traces':
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError('Event trace storage requires a string-keyed dictionary')
        record = value
    else:
        if kind not in classes or not isinstance(value, classes[kind]):
            raise TypeError('Event storage requires the declared typed candidate/event record')
        record = {item.name: getattr(value, item.name) for item in fields(classes[kind])}
        if kind == 'events':
            if not isinstance(value.event_type, event_type):
                raise TypeError('Event storage requires EventType semantics')
            record['event_type'] = value.event_type.value
        if not isinstance(value.evidence, dict):
            raise TypeError('Event evidence must remain a dictionary')
    return json.dumps({'schema_version': 1, 'kind': kind, 'record': _encode(record)},
                      separators=(',', ':'), allow_nan=False)


def _decode_record(kind, payload):
    from src.utils.identity_frame_sequence import _decode
    document = json.loads(payload)
    if (not isinstance(document, dict) or set(document) != {'schema_version', 'kind', 'record'}
            or type(document['schema_version']) is not int or document['schema_version'] != 1
            or document['kind'] != kind):
        raise ValueError('Malformed typed event result envelope')
    record = _decode(document['record'])
    if not isinstance(record, dict):
        raise ValueError('Malformed typed event result record')
    classes, event_type = _record_types()
    if kind == 'traces':
        value = record
    else:
        if kind not in classes or set(record) != {item.name for item in fields(classes[kind])}:
            raise ValueError('Malformed typed event candidate/event fields')
        if kind == 'events':
            record['event_type'] = event_type(record['event_type'])
        value = classes[kind](**record)
    _encode_record(kind, value)
    return value


class _Rows(Sequence):
    def __init__(self, owner, kind, indices=None):
        self.owner, self.kind, self.indices = owner, kind, indices

    def __len__(self):
        return self.owner._lengths[self.kind] if self.indices is None else len(self.indices)

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return index if self.indices is None else self.indices[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(len(self)) if self.indices is None else self.indices
            return _Rows(self.owner, self.kind, indices[index])
        position = self._index(index)
        row = self.owner._execute('SELECT payload FROM records WHERE kind=? AND position=?',
                                  (self.kind, position), fetch=True)
        if row is None:
            raise RuntimeError('Owned event result record is missing')
        return _decode_record(self.kind, row[0])

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError('Write back one event result snapshot by index')
        self.owner._write(self.kind, self._index(index), value, update=True)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def append(self, value):
        if self.indices is not None:
            raise TypeError('Append only to the owning event result sequence')
        self.owner._write(self.kind, len(self), value)


class EventAnalysisSequence:
    """Three indexed sequences plus disk-ranked provisional/final events.

    The 2MiB SQLite cache, disabled mmap and at most256 writes per transaction
    bound owned storage state. There is no Python result cache. Typed payloads
    reuse the existing safe private scalar/container codec, including exact
    IEEE bits, tuple/list distinctions and NumPy scalar/array metadata.
    Active ranking keys must be finite exactly representable binary64 values;
    arbitrary IEEE values in unranked evidence remain unchanged.
    """
    def __init__(self, directory, *, transaction_size=256):
        if isinstance(transaction_size, bool):
            raise ValueError('Event result transaction size must be an integer from1through256')
        transaction_size = operator.index(transaction_size)
        if not 1 <= transaction_size <= 256:
            raise ValueError('Event result transaction size must be an integer from1through256')
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self._path = self._connection = None
        self._pending = 0
        self._events_generation = 0
        self._finalized = False
        self._lengths = {'candidates': 0, 'events': 0, 'traces': 0}
        self.candidates, self.events, self.verification_traces = (
            _Rows(self, kind) for kind in ('candidates', 'events', 'traces'))

    @property
    def path(self):
        return self._path

    @property
    def events_generation(self):
        """In-process change token for completed render-source snapshots."""
        self._open()
        return self._events_generation

    def _open(self):
        if self._connection is None:
            raise RuntimeError('Event analysis storage is not open')
        return self._connection

    def _execute(self, sql, parameters=(), *, fetch=False):
        cursor = self._open().execute(sql, parameters)
        try:
            return cursor.fetchone() if fetch else None
        finally:
            cursor.close()

    def __enter__(self):
        if self._path is not None:
            raise RuntimeError('Event analysis storage cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='event_analysis_',
                                             suffix='.sqlite3', delete=False) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError('Owned event analysis storage escaped its directory')
            self._connection = sqlite3.connect(self._path)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._execute(f'PRAGMA {pragma}', fetch=True)
            self._execute('CREATE TABLE records (kind TEXT NOT NULL, position INTEGER NOT NULL, '
                          'payload TEXT NOT NULL, frame INTEGER, timestamp REAL, PRIMARY KEY(kind,position))')
            self._execute('CREATE INDEX event_frame_order ON records(kind,frame,position)')
            self._execute('CREATE TABLE provisional (candidate_id INTEGER PRIMARY KEY, confidence REAL NOT NULL, '
                          'timestamp REAL NOT NULL, payload TEXT NOT NULL)')
            self._execute('CREATE INDEX provisional_rank ON provisional(confidence DESC,timestamp,candidate_id)')
            self._execute('CREATE TABLE accepted (candidate_id INTEGER PRIMARY KEY, timestamp REAL NOT NULL, rank_order INTEGER NOT NULL)')
            self._execute('CREATE INDEX accepted_event_time ON accepted(timestamp,rank_order)')
            self.flush()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def flush(self):
        self._open().commit()
        self._pending = 0

    def _mutation(self):
        self._pending += 1
        if self._pending >= self.transaction_size:
            self.flush()

    @staticmethod
    def _rank_number(value, name):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f'Event result ranking requires finite binary64 {name}') from error
        if isinstance(value, bool) or not math.isfinite(number) or number != value:
            raise ValueError(f'Event result ranking requires finite binary64 {name}')
        return number

    def _write(self, kind, position, value, *, update=False):
        payload = _encode_record(kind, value)
        frame = timestamp = None
        if kind != 'traces':
            if isinstance(value.frame_index, bool):
                raise ValueError('Event result frame must be a nonnegative integer')
            frame = operator.index(value.frame_index)
            if frame < 0:
                raise ValueError('Event result frame must be a nonnegative integer')
            timestamp = self._rank_number(value.timestamp_s, 'timestamp')
        if update:
            self._execute('UPDATE records SET payload=?,frame=?,timestamp=? WHERE kind=? AND position=?',
                          (payload, frame, timestamp, kind, position))
        else:
            self._execute('INSERT INTO records VALUES (?,?,?,?,?)', (kind, position, payload, frame, timestamp))
            self._lengths[kind] += 1
        if kind == 'events':
            self._events_generation += 1
        self._mutation()

    def add_provisional(self, candidate_id, event):
        confidence = self._rank_number(event.confidence, 'confidence')
        timestamp = self._rank_number(event.timestamp_s, 'timestamp')
        self._execute('INSERT INTO provisional VALUES (?,?,?,?)',
                      (candidate_id, confidence, timestamp, _encode_record('events', event)))
        self._mutation()

    def finalize(self, settings):
        """Preserve stable confidence ranking and the exact absolute-time rule."""
        if self._finalized or len(self.events):
            raise ValueError('Event analysis output must be finalized only once')
        interval = self._rank_number(settings.final_event_min_interval_seconds, 'suppression interval')
        self.flush()
        cursor = self._open().execute('SELECT candidate_id,timestamp FROM provisional '
                                     'ORDER BY confidence DESC,timestamp,candidate_id')
        try:
            for rank_order, (candidate_id, timestamp) in enumerate(cursor):
                suppressed = None
                if interval >= 0:
                    widened = math.nextafter(interval, math.inf)
                    lower = math.nextafter(timestamp - widened, -math.inf)
                    upper = math.nextafter(timestamp + widened, math.inf)
                    suppressed = self._execute('SELECT 1 FROM accepted WHERE timestamp>=? AND timestamp<=? '
                        'AND abs(timestamp-?)<=? LIMIT 1', (lower, upper, timestamp, interval), fetch=True)
                if suppressed is None:
                    self._execute('INSERT INTO accepted VALUES (?,?,?)', (candidate_id, timestamp, rank_order))
                    self._mutation()
                else:
                    trace = self.verification_traces[candidate_id - 1]
                    trace.update(verification_pass=False, rejection_stage='TEMPORAL_SUPPRESSION',
                                 rejection_reason='DUPLICATE_VERIFIED_EVENT')
                    self.verification_traces[candidate_id - 1] = trace
        finally:
            cursor.close()
        self.flush()
        cursor = self._open().execute('SELECT accepted.candidate_id,provisional.payload FROM accepted '
            'JOIN provisional USING(candidate_id) ORDER BY accepted.timestamp,accepted.rank_order')
        try:
            for event_id, (candidate_id, payload) in enumerate(cursor, 1):
                event = _decode_record('events', payload)
                event.event_id = event_id
                trace = self.verification_traces[candidate_id - 1]
                trace['stage_pass']['temporal_suppression'] = True
                trace['stage_pass']['final_authoritative_event'] = True
                trace.update(final_event_emitted=True, final_event_id=event_id, final_event_type=event.event_type.value)
                self.events.append(event)
                self.verification_traces[candidate_id - 1] = trace
        finally:
            cursor.close()
        self.flush()
        self._finalized = True

    def events_by_frame(self, lower=None, upper=None, *, event_type=None):
        """Yield one snapshot at a time with the legacy stable frame ordering."""
        sql, parameters = "SELECT payload FROM records WHERE kind='events'", []
        for limit, relation in ((lower, '>='), (upper, '<=')):
            if limit is not None:
                sql += f' AND frame{relation}?'
                parameters.append(operator.index(limit))
        cursor = self._open().execute(sql + ' ORDER BY frame,position', parameters)
        try:
            for (payload,) in cursor:
                event = _decode_record('events', payload)
                if event_type is None or event.event_type == event_type:
                    yield event
        finally:
            # A pipeline failure may close the caller's ExitStack before a
            # suspended read producer is finalized. The connection has already
            # closed all its cursors in that case; do not mask that failure.
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
                        raise ValueError('Owned event analysis cleanup escaped its directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(('cleanup', error))
        if primary is None and failures:
            _, primary = failures.pop(0)
            for stage, error in failures:
                primary.add_note(f'Event analysis storage {stage} also failed: {error}')
            raise primary
        if primary is not None:
            for stage, error in failures:
                primary.add_note(f'Event analysis storage {stage} also failed: {error}')

    def __exit__(self, error_type, error, traceback):
        self._dispose(error)
        return False
