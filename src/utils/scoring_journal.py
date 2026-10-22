"""Caller-owned SQLite rule journal with revision-pinned, uncached views.

Ordinary transitions commit within 256 SQL writes. A correction is deliberately
one complete replay transaction: its SQLite cache remains 2 MiB, while its disk
transaction and retained historical revisions can grow with the input. Explicit
``history.to_list()`` is a compatibility materialization; exports never use it.
The owner is temporary storage, not a resumable checkpoint or a timeline revision
coordinator. Every view and suspended iterator requires its owner to remain open.
"""
from collections.abc import Sequence, Set
from contextlib import contextmanager
import json
import operator
import os
from pathlib import Path
import sqlite3
import tempfile

from src.scoring.event_history import ScoringEventLogEntry
from src.utils.mapped_sequence_view import MappedSequenceView


def _json(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':'))


class ScoringJournal:
    """Private, single-caller journal; use inside the analysis export ExitStack."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.connection = None
        self.pending = 0
        self.readers = set()
        self._stage = None
        self._failure_reason = None

    def _open(self):
        if self.connection is None:
            raise RuntimeError('Scoring journal owner is not open')
        if self._failure_reason is not None:
            raise RuntimeError('Scoring journal is invalid after a transaction failure: '
                               + self._failure_reason)
        return self.connection

    def _invalidate(self, primary):
        """Fail the entire analysis when an accepted batch cannot be retained.

        The engine can undo one in-flight rule transition, but cannot undo the
        previously accepted transitions in an ordinary uncommitted batch. A
        failed batch commit therefore makes every subsequent view unusable.
        """
        self._failure_reason = type(primary).__name__ + ': ' + str(primary)
        try:
            self.connection.rollback()
        except BaseException as cleanup:
            primary.add_note(f'Scoring journal rollback also failed: {cleanup}')
        self.pending = 0

    def _execute(self, sql, args=(), *, one=False):
        cursor = self._open().execute(sql, args)
        try:
            return cursor.fetchone() if one else None
        finally:
            cursor.close()

    def __enter__(self):
        if self.path is not None:
            raise RuntimeError('Scoring journal owner cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='scoring_journal_',
                                             suffix='.sqlite3', delete=False) as stream:
                self.path = Path(stream.name).resolve()
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self._execute('PRAGMA '+pragma, one=True)
            self._execute('CREATE TABLE revisions(number INTEGER PRIMARY KEY, parent INTEGER, event_id INTEGER, reason TEXT, published INTEGER NOT NULL)')
            self._execute('CREATE TABLE current_revision(number INTEGER NOT NULL)')
            self._execute('CREATE TABLE transitions(revision INTEGER, position INTEGER, event_id INTEGER, history TEXT NOT NULL, input TEXT NOT NULL, PRIMARY KEY(revision,position), UNIQUE(revision,event_id))')
            self._execute("CREATE INDEX scoring_line_call_frame ON transitions(revision,json_extract(input,'$.line_call.bounce_frame'),position) WHERE json_type(input,'$.line_call')='object'")
            # BLOB affinity preserves SQLite scalar identity (e.g. integer 1 and
            # string "1"), rather than coercing every stable point ID to text.
            self._execute('CREATE TABLE completed_points(revision INTEGER, point_id BLOB, PRIMARY KEY(revision,point_id))')
            self._execute('INSERT INTO revisions VALUES(0,NULL,NULL,NULL,1)')
            self._execute('INSERT INTO current_revision VALUES(0)')
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def revision(self, number=0):
        number = operator.index(number)
        row = self._execute('SELECT published FROM revisions WHERE number=?', (number,), one=True)
        if row is None or not row[0]:
            raise ValueError('Unknown published scoring revision')
        return RevisionView(self, number)

    def flush(self):
        if self._stage is not None:
            raise ValueError('A correction replay cannot be flushed before publication')
        connection = self._open()
        try:
            connection.commit()
        except BaseException as error:
            self._invalidate(error)
            raise
        self.pending = 0

    def _active_number(self):
        return self._execute('SELECT number FROM current_revision', one=True)[0]

    def _dispose(self, primary=None):
        failures = []
        if self._stage is not None:
            self._stage._discarded = True
            self._stage = None
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
                path = Path(str(self.path)+suffix)
                try:
                    if path.is_symlink() or path.resolve().parent != self.directory:
                        raise ValueError('Scoring cleanup escaped its owned directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(error)
        if primary is not None:
            for error in failures:
                primary.add_note(f'Scoring journal cleanup also failed: {error}')
        elif failures:
            raise failures[0]

    def __exit__(self, kind, error, traceback):
        self._dispose(error)
        return False


class RevisionView:
    def __init__(self, owner, number):
        self.owner, self.number = owner, number
        self._discarded = self._publish_requested = False
        self.history = JournalHistory(self)
        self.inputs = JournalRows(self, 'input')
        self.completed_point_ids = JournalSet(self, completed=True)
        self.corrections = CorrectionRows(self)

    def _read(self):
        self.owner._open()
        if self._discarded:
            raise ValueError('A discarded scoring revision cannot be used')

    def _write(self):
        self._read()
        if self.owner._stage is self:
            if self._publish_requested:
                raise ValueError('Correction publication was already requested')
        elif self.owner._stage is not None or self.number != self.owner._active_number():
            raise ValueError('A historical scoring revision cannot change')

    def line_calls(self, decode_input):
        """Typed evidence is projected from accepted inputs, never cached again."""
        self._read()
        return JournalLineCalls(self, decode_input)

    def append_transition(self, history_record, input_record, completed_point_id=None):
        self._write()
        # Encode the complete pair before any disk mutation; the engine handles
        # typed evidence and its fixed-size state/resolver rollback separately.
        position = len(self.inputs)
        entry = ScoringEventLogEntry(sequence_number=position+1, **history_record)
        if not isinstance(input_record, dict) or input_record.get('event_id') != entry.event_id:
            raise ValueError('History and rule input require the same event ID')
        history_payload, input_payload = _json(entry.__dict__), _json(input_record)
        if completed_point_id is not None:
            hash(completed_point_id)
        owner = self.owner
        writes = 2 if completed_point_id is not None else 1
        ordinary = owner._stage is None
        if ordinary and owner.pending + writes > 256:
            owner.flush()
        connection = owner._open()
        released = False
        try:
            if not connection.in_transaction:
                owner._execute('BEGIN')
            owner._execute('SAVEPOINT scoring_event')
            owner._execute('INSERT INTO transitions VALUES(?,?,?,?,?)',
                           (self.number, position, entry.event_id, history_payload, input_payload))
            if completed_point_id is not None:
                owner._execute('INSERT OR IGNORE INTO completed_points VALUES(?,?)',
                               (self.number, completed_point_id))
            owner._execute('RELEASE scoring_event')
            released = True
            if ordinary:
                owner.pending += writes
                if owner.pending >= 256:
                    owner.flush()
        except BaseException as error:
            if owner._failure_reason is not None:
                # flush() already invalidated an uncertain batch commit.
                pass
            elif released:
                owner._invalidate(error)
            else:
                try:
                    owner._execute('ROLLBACK TO scoring_event')
                    owner._execute('RELEASE scoring_event')
                except BaseException as cleanup:
                    error.add_note(f'Scoring event savepoint rollback also failed: {cleanup}')
                    owner._invalidate(error)
            raise
        return entry

    @contextmanager
    def staged_correction(self, event_id, reason):
        self._write()
        if self.owner._stage is not None:
            raise ValueError('A scoring correction is already in progress')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Correction reason is required')
        if not self.history.is_processed(event_id):
            raise ValueError('Unknown source event ID')
        owner = self.owner
        owner.flush()
        staged = RevisionView(owner, self.number+1)
        owner._stage = staged
        connection = owner._open()
        committing = False
        try:
            owner._execute('BEGIN IMMEDIATE')
            owner._execute('INSERT INTO revisions VALUES(?,?,?,?,0)',
                           (staged.number, self.number, event_id, reason))
            yield staged
            if not staged._publish_requested:
                raise ValueError('Correction replay was not published')
            owner._execute('UPDATE revisions SET published=1 WHERE number=?', (staged.number,))
            owner._execute('UPDATE current_revision SET number=?', (staged.number,))
            committing = True
            connection.commit()
        except BaseException as error:
            if committing:
                # COMMIT may have reached durable storage before its error was
                # observed. Never expose an old engine against a new revision.
                owner._invalidate(error)
            else:
                try:
                    connection.rollback()
                except BaseException as cleanup:
                    error.add_note(f'Scoring correction rollback also failed: {cleanup}')
                    owner._invalidate(error)
            staged._discarded = True
            raise
        finally:
            owner._stage = None
            owner.pending = 0

    def publish(self):
        """Request atomic publication on clean correction-context exit."""
        self._write()
        owner = self.owner
        if owner._stage is not self:
            raise ValueError('Only an active correction can be published')
        parent = owner._execute('SELECT parent FROM revisions WHERE number=?', (self.number,), one=True)[0]
        # A score correction changes evidence, not source-event identity/order.
        mismatch = owner._execute('SELECT 1 FROM transitions a LEFT JOIN transitions b ON b.revision=? AND a.position=b.position WHERE a.revision=? AND (b.event_id IS NULL OR a.event_id!=b.event_id) LIMIT 1',
                                  (self.number, parent), one=True)
        old_count = owner._execute('SELECT COUNT(*) FROM transitions WHERE revision=?', (parent,), one=True)[0]
        if mismatch or len(self.inputs) != old_count:
            raise ValueError('Correction replay must preserve every source event in order')
        self._publish_requested = True

    def save_history_json(self, filepath):
        """Atomically stream the legacy v2 JSON shape, including prior logs."""
        self._read()
        target = Path(filepath)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                    prefix=target.name+'.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write('{"schema_version":"2.0","revision":'+str(self.number)+',"score_history":')
                _write_array(stream, JournalRows(self, 'history', dictionaries=True))
                stream.write(',"corrections":[')
                first = True
                with _iterating(self.corrections) as corrections:
                    for correction in corrections:
                        if not first:
                            stream.write(',')
                        first = False
                        previous = correction.pop('previous_history')
                        stream.write(_json(correction)[:-1]+',"previous_history":')
                        _write_array(stream, previous)
                        stream.write('}')
                stream.write(']}')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            temporary = None
        except BaseException as error:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as cleanup:
                    error.add_note(f'Scoring export temporary cleanup also failed: {cleanup}')
            raise


@contextmanager
def _iterating(records):
    """Close a suspended SQL reader without masking its consumer's failure."""
    iterator = iter(records)
    try:
        yield iterator
    except BaseException as error:
        close = getattr(iterator, 'close', None)
        if close is not None:
            try:
                close()
            except BaseException as cleanup:
                error.add_note(f'Scoring reader cleanup also failed: {cleanup}')
        raise
    else:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()


def _write_array(stream, records):
    stream.write('[')
    with _iterating(records) as iterator:
        first = True
        for record in iterator:
            if not first:
                stream.write(',')
            first = False
            stream.write(_json(record))
    stream.write(']')


class JournalLineCalls(Sequence):
    """Accepted-order evidence and indexed, last-per-frame display selections.

    The supplied decoder is the scoring engine's existing finite input codec.
    Every read remains pinned to this revision and requires its open owner.
    """
    _where = "revision=? AND json_type(input,'$.line_call')='object'"
    _frame = "json_extract(input,'$.line_call.bounce_frame')"

    def __init__(self, revision, decode_input):
        self.revision, self.decode_input = revision, decode_input

    def __len__(self):
        self.revision._read()
        return self.revision.owner._execute('SELECT COUNT(*) FROM transitions WHERE '+self._where,
                                           (self.revision.number,), one=True)[0]

    def _decode(self, payload):
        return self.decode_input(json.loads(payload))['line_call']

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.revision.owner._execute('SELECT input FROM transitions WHERE '+self._where+
            ' ORDER BY position LIMIT 1 OFFSET ?', (self.revision.number, index), one=True)
        return self._decode(row[0])

    def _records(self, sql, args, *, framed=False):
        self.revision._read()
        owner = self.revision.owner
        cursor = owner._open().execute(sql, args)
        owner.readers.add(cursor)
        try:
            while True:
                self.revision._read()
                row = cursor.fetchone()
                if row is None:
                    break
                yield (row[0], self._decode(row[1])) if framed else self._decode(row[0])
        finally:
            if cursor in owner.readers:
                cursor.close()
                owner.readers.discard(cursor)

    def __iter__(self):
        return self._records('SELECT input FROM transitions WHERE '+self._where+' ORDER BY position',
                             (self.revision.number,))

    def by_frame(self, lower, upper):
        """Inclusive window; last value per frame, first insertion order."""
        lower, upper = operator.index(lower), operator.index(upper)
        return self._records('SELECT selected.frame,latest.input FROM transitions AS latest JOIN '
            '(SELECT '+self._frame+' AS frame,MIN(position) AS first,MAX(position) AS last '
            'FROM transitions WHERE '+self._where+' AND '+self._frame+'>=? AND '+self._frame+'<=? '
            'GROUP BY '+self._frame+') AS selected '
            'ON latest.revision=? AND latest.position=selected.last ORDER BY selected.first',
            (self.revision.number, lower, upper, self.revision.number), framed=True)


class JournalRows(Sequence):
    def __init__(self, revision, column, *, dictionaries=False):
        if column not in ('history', 'input'):
            raise ValueError('Unknown scoring row selector')
        self.revision, self.column, self.dictionaries = revision, column, dictionaries

    def __len__(self):
        self.revision._read()
        return self.revision.owner._execute('SELECT COUNT(*) FROM transitions WHERE revision=?',
                                            (self.revision.number,), one=True)[0]

    def _decode(self, payload):
        record = json.loads(payload)
        return (ScoringEventLogEntry(**record)
                if self.column == 'history' and not self.dictionaries else record)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.revision.owner._execute('SELECT '+self.column+' FROM transitions WHERE revision=? AND position=?',
                                           (self.revision.number, index), one=True)
        return self._decode(row[0])

    def __iter__(self):
        self.revision._read()
        owner = self.revision.owner
        cursor = owner._open().execute('SELECT '+self.column+' FROM transitions WHERE revision=? ORDER BY position',
                                       (self.revision.number,))
        owner.readers.add(cursor)
        try:
            while True:
                self.revision._read()
                row = cursor.fetchone()
                if row is None:
                    break
                yield self._decode(row[0])
        finally:
            if cursor in owner.readers:
                cursor.close()
                owner.readers.discard(cursor)


class JournalSet(Set):
    def __init__(self, revision, *, completed=False):
        self.revision, self.completed = revision, completed

    @property
    def _table(self):
        return 'completed_points' if self.completed else 'transitions'

    @property
    def _column(self):
        return 'point_id' if self.completed else 'event_id'

    def __len__(self):
        self.revision._read()
        return self.revision.owner._execute('SELECT COUNT(*) FROM '+self._table+' WHERE revision=?',
                                            (self.revision.number,), one=True)[0]

    def __contains__(self, value):
        hash(value)
        self.revision._read()
        return self.revision.owner._execute('SELECT 1 FROM '+self._table+' WHERE revision=? AND '+self._column+'=?',
                                            (self.revision.number, value), one=True) is not None

    def __iter__(self):
        self.revision._read()
        owner = self.revision.owner
        cursor = owner._open().execute('SELECT '+self._column+' FROM '+self._table+' WHERE revision=?',
                                       (self.revision.number,))
        owner.readers.add(cursor)
        try:
            while True:
                self.revision._read()
                row = cursor.fetchone()
                if row is None:
                    break
                yield row[0]
        finally:
            if cursor in owner.readers:
                cursor.close()
                owner.readers.discard(cursor)


class JournalHistory:
    def __init__(self, revision):
        self.entries = JournalRows(revision, 'history')
        self.processed_event_ids = JournalSet(revision)

    def is_processed(self, event_id):
        return event_id in self.processed_event_ids

    def to_list(self):
        """Explicit legacy materialization; bounded callers use streamed export."""
        return [entry.to_dict() for entry in self.entries]


class CorrectionRows(Sequence):
    def __init__(self, revision):
        self.revision = revision

    def __len__(self):
        self.revision._read()
        return self.revision.owner._execute('SELECT COUNT(*) FROM revisions WHERE number>0 AND number<=?',
                                            (self.revision.number,), one=True)[0]

    def _decode(self, row):
        number, parent, event_id, reason = row
        previous = RevisionView(self.revision.owner, parent)
        return dict(revision=number, event_id=event_id, reason=reason,
                    previous_history=JournalRows(previous, 'history', dictionaries=True),
                    source='human_review')

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.revision.owner._execute('SELECT number,parent,event_id,reason FROM revisions WHERE number>0 AND number<=? ORDER BY number LIMIT 1 OFFSET ?',
                                           (self.revision.number, index), one=True)
        return self._decode(row)

    def __iter__(self):
        self.revision._read()
        owner = self.revision.owner
        cursor = owner._open().execute('SELECT number,parent,event_id,reason FROM revisions WHERE number>0 AND number<=? ORDER BY number',
                                       (self.revision.number,))
        owner.readers.add(cursor)
        try:
            while True:
                self.revision._read()
                row = cursor.fetchone()
                if row is None:
                    break
                yield self._decode(row)
        finally:
            if cursor in owner.readers:
                cursor.close()
                owner.readers.discard(cursor)
