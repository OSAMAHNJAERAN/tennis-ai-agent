"""Caller-owned, revision-bound point/attempt/rally snapshots and memberships."""
from collections.abc import Sequence, Set
import json
import operator
from pathlib import Path
import sqlite3
import tempfile

from src.utils.mapped_sequence_view import MappedSequenceView


class PointAnalysisSequence:
    def __init__(self, directory, analysis_id, revision=0, *, source_fingerprint=''):
        if not isinstance(analysis_id, str) or not analysis_id or type(revision) is not int or revision < 0:
            raise ValueError('Point analysis requires an identity and nonnegative revision')
        self.directory = Path(directory).resolve()
        self.analysis_id, self.revision = analysis_id, revision
        self.source_fingerprint = source_fingerprint
        self.path = self.connection = None
        self.pending, self.sealed = 0, False
        self.readers = set()

    def _open(self):
        if self.connection is None:
            raise RuntimeError('Point analysis owner is not open')
        return self.connection

    def execute(self, sql, args=(), *, one=False):
        cursor = self._open().execute(sql, args)
        try:
            return cursor.fetchone() if one else None
        finally:
            cursor.close()

    def __enter__(self):
        if self.path is not None:
            raise RuntimeError('Point analysis owner cannot be reused')
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix='point_analysis_', suffix='.sqlite3', delete=False) as file:
                self.path = Path(file.name).resolve()
            self.connection = sqlite3.connect(self.path)
            for pragma in ('journal_mode=DELETE', 'cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
                self.execute('PRAGMA '+pragma, one=True)
            self.execute('CREATE TABLE records(kind TEXT, position INTEGER, uid TEXT, parent TEXT, status TEXT, event_id INTEGER, frame INTEGER, payload TEXT, PRIMARY KEY(kind,uid), UNIQUE(kind,position))')
            self.execute('CREATE INDEX point_parent_order ON records(kind,parent,position)')
            self.execute('CREATE UNIQUE INDEX point_event ON records(kind,event_id) WHERE event_id IS NOT NULL')
            self.execute('CREATE INDEX point_status ON records(kind,status,position)')
            self.execute('CREATE INDEX point_frame ON records(kind,frame,position)')
            self.execute("CREATE INDEX point_dead_order ON records(position,event_id) WHERE kind='transitions' AND json_extract(payload,'$.outcome_type')='DEAD_BALL_IGNORED'")
            self.execute('CREATE TABLE members(position INTEGER PRIMARY KEY,shot_id INTEGER,event_id INTEGER,point_uid TEXT,attempt_uid TEXT,rally_uid TEXT,scope TEXT,participant INTEGER,has_landing INTEGER,is_serve INTEGER)')
            for key in ('point_uid', 'attempt_uid', 'rally_uid', 'scope'):
                self.execute(f'CREATE INDEX member_{key} ON members({key},position)')
            self.flush()
            return self
        except BaseException as error:
            self._dispose(error)
            raise

    def flush(self):
        self._open().commit()
        self.pending = 0

    def _write(self):
        if self.sealed:
            raise ValueError('A sealed point revision cannot change')
        self._open()

    def _changed(self):
        self.pending += 1
        if self.pending >= 256:
            self.flush()

    def put(self, kind, uid, record, *, parent=None, status=None, event_id=None, frame=None):
        self._write()
        payload = json.dumps(record, allow_nan=False, separators=(',', ':'))
        existing = self.execute('SELECT position FROM records WHERE kind=? AND uid=?', (kind, uid), one=True)
        position = existing[0] if existing else self.execute('SELECT COALESCE(MAX(position),-1)+1 FROM records WHERE kind=?', (kind,), one=True)[0]
        self.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(kind,uid) DO UPDATE SET parent=excluded.parent,status=excluded.status,event_id=excluded.event_id,frame=excluded.frame,payload=excluded.payload',
                     (kind, position, uid, parent, status, event_id, frame, payload))
        self._changed()

    def get(self, kind, uid):
        row = self.execute('SELECT payload FROM records WHERE kind=? AND uid=?', (kind, uid), one=True)
        return json.loads(row[0]) if row else None

    def event(self, event_id):
        row = self.execute('SELECT payload FROM records WHERE kind=? AND event_id=?', ('transitions', event_id), one=True)
        return json.loads(row[0]) if row else None

    def dead_event_ids(self):
        """Read-only set of accepted dead outcomes, without a second ID cache."""
        self._open()
        return DeadEventIds(self)

    def add_member(self, position, shot, transition, scope, participant):
        self._write()
        self.execute('INSERT INTO members VALUES(?,?,?,?,?,?,?,?,?,?)', (position, shot.shot_id, shot.match_event_id,
            transition.get('point_uid'), transition.get('attempt_uid'), transition.get('rally_uid'), scope, participant,
            int(shot.landing_court_position_m is not None), int(transition.get('physical_event_type') == 'SERVE_CONTACT')))
        self._changed()

    def rows(self, kind, *, parent=None, status=None):
        clauses, args = ['kind=?'], [kind]
        if parent is not None:
            clauses.append('parent=?'); args.append(parent)
        if status is not None:
            clauses.append('status=?'); args.append(status)
        return PointRows(self, ' AND '.join(clauses), tuple(args), kind)

    def member_ids(self, key, uid):
        if key not in ('point_uid', 'attempt_uid', 'rally_uid'):
            raise ValueError('Unknown point membership selector')
        return SQLValues(self, f'SELECT shot_id FROM members WHERE {key}=? ORDER BY position', (uid,))

    def eligible_positions(self, *, landings=False):
        return SQLValues(self, "SELECT position FROM members WHERE scope='LEGAL_RALLY'"+
                         (' AND has_landing=1' if landings else '')+' ORDER BY position', ())

    def participant(self, position):
        row = self.execute('SELECT participant FROM members WHERE position=?', (position,), one=True)
        return row[0] if row else None

    def frame_state(self, frame, initial):
        row = self.execute("SELECT payload FROM records WHERE kind='transitions' AND frame<=? ORDER BY frame DESC,position DESC LIMIT 1", (operator.index(frame),), one=True)
        return json.loads(row[0])['post_state'] if row else dict(initial)

    def member_rows(self):
        return MembershipRows(self, 'SELECT position,shot_id,event_id,point_uid,attempt_uid,rally_uid,scope,participant FROM members ORDER BY position', ())

    def seal(self):
        self.flush()
        self.sealed = True
        return self

    def _dispose(self, primary=None):
        failures = []
        for cursor in tuple(self.readers):
            try:
                cursor.close()
            except BaseException as error:
                failures.append(error)
            finally:
                self.readers.remove(cursor)
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
                        raise ValueError('Point cleanup escaped its owned directory')
                    path.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(error)
        if primary is not None:
            for error in failures:
                primary.add_note(f'Point owner cleanup also failed: {error}')
        elif failures:
            raise failures[0]

    def __exit__(self, kind, error, traceback):
        self._dispose(error)
        return False


class DeadEventIds(Set):
    _where = "kind='transitions' AND json_extract(payload,'$.outcome_type')='DEAD_BALL_IGNORED'"

    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return self.owner.execute('SELECT COUNT(*) FROM records WHERE '+self._where, one=True)[0]

    def __contains__(self, value):
        hash(value)
        row = self.owner.execute('SELECT event_id FROM records WHERE kind=? AND event_id=? '
            "AND json_extract(payload,'$.outcome_type')='DEAD_BALL_IGNORED'",
            ('transitions', value), one=True)
        # SQLite INTEGER affinity can match text "3" to ID 3. Preserve the
        # old Python set's equality semantics after the indexed lookup.
        return row is not None and value == row[0]

    def __iter__(self):
        return iter(SQLValues(self.owner, 'SELECT event_id FROM records WHERE '+self._where+
                              ' ORDER BY position', ()))


class SQLValues(Sequence):
    def __init__(self, owner, sql, args):
        self.owner, self.sql, self.args = owner, sql, args

    def __len__(self):
        return self.owner.execute('SELECT COUNT(*) FROM ('+self.sql+')', self.args, one=True)[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.owner.execute(self.sql+' LIMIT 1 OFFSET ?', (*self.args, index), one=True)[0]

    def __iter__(self):
        cursor = self.owner._open().execute(self.sql, self.args)
        self.owner.readers.add(cursor)
        try:
            for row in cursor:
                yield row[0]
        finally:
            if cursor in self.owner.readers:
                cursor.close(); self.owner.readers.remove(cursor)


class PointRows(SQLValues):
    def __init__(self, owner, where, args, kind):
        super().__init__(owner, 'SELECT payload FROM records WHERE '+where+' ORDER BY position', args)
        self.kind = kind

    def _decode(self, payload):
        value = json.loads(payload)
        if self.kind in ('points', 'rallies'):
            value['shot_ids'] = self.owner.member_ids('point_uid' if self.kind == 'points' else 'rally_uid', value['uid'])
            value['shot_membership_scope'] = ('ALL_POINT_CONTACT_OBSERVATIONS; SEE_SHOT_PLAY_SCOPE'
                if self.kind == 'points' else 'LEGAL_RALLY_CONTACTS')
        return value

    def __getitem__(self, index):
        if isinstance(index, slice):
            return MappedSequenceView(self, lambda value, _: value)[index]
        return self._decode(super().__getitem__(index))

    def __iter__(self):
        iterator = super().__iter__()
        try:
            for payload in iterator:
                yield self._decode(payload)
        finally:
            iterator.close()


class MembershipRows(SQLValues):
    @staticmethod
    def _decode(row):
        return dict(zip(('shot_position', 'shot_id', 'event_id', 'point_uid', 'attempt_uid',
                         'rally_uid', 'play_scope', 'participant_id'), row))

    def __getitem__(self,index):
        if isinstance(index,slice):
            return MappedSequenceView(self,lambda value,_:value)[index]
        index=operator.index(index)
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        row=self.owner.execute(self.sql+' LIMIT 1 OFFSET ?',(*self.args,index),one=True)
        return self._decode(row)

    def __iter__(self):
        cursor = self.owner._open().execute(self.sql, self.args)
        self.owner.readers.add(cursor)
        try:
            for row in cursor:
                yield self._decode(row)
        finally:
            if cursor in self.owner.readers:
                cursor.close()
                self.owner.readers.remove(cursor)
