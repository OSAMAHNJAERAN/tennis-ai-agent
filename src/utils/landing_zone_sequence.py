"""Owned ground-contact display records, calibration registry and time queries.

This stores already admitted presentation evidence; it never qualifies contacts
or calibration. Its owner must outlive rendering and streamed JSON publication.
"""
from collections.abc import Mapping, Sequence
import json
import math
import operator
from pathlib import Path
import sqlite3
import tempfile

from src.utils.analytics_json_artifact import _destination
from src.utils.mapped_sequence_view import MappedSequenceView


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


class LandingZoneSequence:
    def __init__(self, directory, analysis_id, revision=0):
        if not isinstance(analysis_id, str) or not analysis_id or type(revision) is not int or revision < 0:
            raise ValueError('Landing records require a bound analysis and nonnegative revision')
        self.directory = Path(directory).resolve()
        self.analysis_id, self.revision = analysis_id, revision
        self.path = self.connection = None
        self.pending = self.count = 0
        self.sealed = False
        self.failure = None
        self.readers = set()
        self.records = LandingRecords(self)
        self.calibrations = LandingCalibrations(self)

    def _open(self):
        if self.failure is not None:
            raise RuntimeError('Landing owner cannot continue after uncertain storage failure') from self.failure
        if self.connection is None:
            raise RuntimeError('Landing owner is not open')
        return self.connection

    def _execute(self, sql, args=(), *, one=False):
        cursor = self._open().execute(sql, args)
        try:
            return cursor.fetchone() if one else None
        finally:
            cursor.close()

    def __enter__(self):
        if self.path is not None:
            raise RuntimeError('Landing owner cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='landing_zones_', suffix='.sqlite3', delete=False) as stream:
                self.path = Path(stream.name).resolve()
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._execute('PRAGMA ' + pragma, one=True)
            self._execute('CREATE TABLE records(position INTEGER PRIMARY KEY,start REAL NOT NULL,partition TEXT NOT NULL,payload TEXT NOT NULL)')
            self._execute('CREATE INDEX landing_time ON records(partition,start,position)')
            self._execute('CREATE TABLE calibrations(id TEXT PRIMARY KEY,position INTEGER NOT NULL,payload TEXT NOT NULL)')
            self._execute('CREATE INDEX calibration_order ON calibrations(position)')
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def flush(self):
        connection = self._open()
        try:
            connection.commit()
        except BaseException as error:
            self.failure = error
            raise
        self.pending = 0

    def append(self, record, calibration):
        connection = self._open()
        if self.sealed:
            raise ValueError('Sealed landing evidence cannot change')
        if (not isinstance(record, dict) or not isinstance(calibration, dict)
                or any(value.get('analysis_id') != self.analysis_id
                       or type(value.get('analysis_revision')) is not int
                       or value['analysis_revision'] != self.revision for value in (record, calibration))):
            raise ValueError('Landing record and calibration must belong to this analysis revision')
        ident = record.get('calibration_id')
        timestamp, duration = record.get('timestamp_s'), record.get('duration_s')
        if (not isinstance(ident, str) or not ident
                or record.get('partition') not in ('service', 'tactical_3x3', 'tactical_4x3')
                or type(calibration.get('frame_index')) is not int or calibration['frame_index'] < 0
                or type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0
                or type(duration) not in (int, float) or not math.isfinite(duration) or not .6 <= duration <= 1.):
            raise ValueError('Landing display requires a finite time, bounded duration, partition and calibration frame')
        # Complete finite encoding precedes mutation. Store one record at a
        # time; calibration replacement preserves its first insertion order.
        payload, calibration_payload = _json(record), _json(calibration)
        if self.pending + 2 > 256:
            self.flush()
        released = False
        try:
            if not connection.in_transaction:
                self._execute('BEGIN')
            self._execute('SAVEPOINT landing_record')
            self._execute('INSERT INTO calibrations VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                          (ident, self.count, calibration_payload))
            self._execute('INSERT INTO records VALUES(?,?,?,?)',
                          (self.count, float(timestamp), record['partition'], payload))
            self._execute('RELEASE landing_record')
            released = True
            self.count += 1
            self.pending += 2
            if self.pending >= 256:
                self.flush()
        except BaseException as error:
            if self.failure is None:
                if released:
                    self.failure = error
                else:
                    try:
                        self._execute('ROLLBACK TO landing_record')
                        self._execute('RELEASE landing_record')
                    except BaseException as cleanup:
                        error.add_note(f'Landing record rollback also failed: {cleanup}')
                        self.failure = error
            raise

    def seal(self):
        self.flush()
        self.sealed = True
        return self

    def active_records(self, partition, timestamp):
        self._open()
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError('Landing display time must be finite')
        # Broaden the indexed candidates by one representable value on each
        # side, then repeat the original Python predicate exactly. This avoids
        # replacing addition-based expiry with rounded subtraction semantics.
        lower = math.nextafter(float(timestamp) - 1., -math.inf)
        upper = math.nextafter(float(timestamp), math.inf)
        return ActiveLandings(self, partition, timestamp, lower, upper)

    def _rows(self, sql, args=()):
        cursor = self._open().execute(sql, args)
        self.readers.add(cursor)
        try:
            while True:
                self._open()
                row = cursor.fetchone()
                if row is None:
                    break
                yield row
        finally:
            if cursor in self.readers:
                cursor.close()
                self.readers.discard(cursor)

    def write_json(self, destination):
        self._open()
        if not self.sealed:
            raise ValueError('Seal landing evidence before publication')
        records, calibrations = iter(self.records), iter(self.calibrations.items())
        primary = None
        try:
            with _destination(destination) as stream:
                stream.write('{"schema_version":"1.0","analysis_id":')
                stream.write(_json(self.analysis_id))
                stream.write(',"analysis_revision":' + str(self.revision) + ',"calibrations":{')
                first = True
                for key, value in calibrations:
                    if not first:
                        stream.write(',')
                    first = False
                    stream.write(_json(key) + ':' + _json(value))
                stream.write('},"records":[')
                first = True
                for record in records:
                    if not first:
                        stream.write(',')
                    first = False
                    stream.write(_json(record))
                stream.write(']}\n')
        except BaseException as error:
            primary = error
            raise
        finally:
            for reader in (records, calibrations):
                try:
                    reader.close()
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    primary.add_note(f'Landing export reader cleanup also failed: {cleanup}')

    def _dispose(self, primary=None):
        failures = []
        for cursor in tuple(self.readers):
            try:
                cursor.close()
            except BaseException as error:
                failures.append(error)
            finally:
                self.readers.discard(cursor)
        if self.connection is not None:
            try:
                self.connection.close()
            except BaseException as error:
                failures.append(error)
            self.connection = None
        if self.path is not None:
            for suffix in ('-journal', '-wal', '-shm', ''):
                target = Path(str(self.path) + suffix)
                try:
                    if target.is_symlink() or target.resolve().parent != self.directory:
                        raise ValueError('Landing cleanup escaped its owned directory')
                    target.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(error)
        if primary is not None:
            for error in failures:
                primary.add_note(f'Landing owner cleanup also failed: {error}')
        elif failures:
            raise failures[0]

    def __exit__(self, kind, error, traceback):
        self._dispose(error)
        return False


class LandingRecords(Sequence):
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        self.owner._open()
        return self.owner.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return json.loads(self.owner._execute('SELECT payload FROM records WHERE position=?', (index,), one=True)[0])

    def __iter__(self):
        reader = self.owner._rows('SELECT payload FROM records ORDER BY position')
        try:
            for row in reader:
                yield json.loads(row[0])
        finally:
            reader.close()


class LandingCalibrations(Mapping):
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return self.owner._execute('SELECT COUNT(*) FROM calibrations', one=True)[0]

    def __getitem__(self, key):
        hash(key)
        row = self.owner._execute('SELECT id,payload FROM calibrations WHERE id=?', (key,), one=True)
        if row is None or key != row[0]:
            raise KeyError(key)
        return json.loads(row[1])

    def __iter__(self):
        reader = self.owner._rows('SELECT id FROM calibrations ORDER BY position')
        try:
            for row in reader:
                yield row[0]
        finally:
            reader.close()

    def items(self):
        reader = self.owner._rows('SELECT id,payload FROM calibrations ORDER BY position')
        try:
            for row in reader:
                yield row[0], json.loads(row[1])
        finally:
            reader.close()


class ActiveLandings:
    def __init__(self, owner, partition, timestamp, lower, upper):
        self.owner, self.partition, self.timestamp = owner, partition, timestamp
        self.lower, self.upper = lower, upper

    def __iter__(self):
        reader = self.owner._rows('SELECT payload FROM records WHERE partition=? AND start>=? AND start<=? ORDER BY position',
                                  (self.partition, self.lower, self.upper))
        try:
            for row in reader:
                record = json.loads(row[0])
                if record['timestamp_s'] <= self.timestamp < record['timestamp_s'] + record['duration_s']:
                    yield record
        finally:
            reader.close()
