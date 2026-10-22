"""Owner admission and retained logical-byte accounting, without model imports.

Call admission methods inside the caller's BEGIN IMMEDIATE transaction, then
insert the session/job/correction on that same connection. Legacy uploads and
retries instead keep a durable reservation until their job insertion commits.
Reservations retain ownership after release, including for crash leftovers.

This is admission accounting and a worker-stage guard, not a filesystem quota:
files can grow between stage checks. Nothing here deletes retained user data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sqlite3
import stat

from fastapi import HTTPException
from filelock import FileLock, Timeout


_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_UNAVAILABLE = 'Storage usage could not be checked. Please try again.'


class ResourceQuotas:
    def __init__(self, jobs, settings):
        self.jobs = jobs
        self.root = Path(jobs.root)
        self.settings = settings
        with jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('''CREATE TABLE IF NOT EXISTS resource_reservations (
                analysis_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                kind TEXT NOT NULL CHECK(kind IN ('legacy','retry')),
                active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                recovery_version INTEGER NOT NULL DEFAULT 0 CHECK(recovery_version IN (0,1)))''')
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(resource_reservations)')}
            if 'recovery_version' not in columns:
                # Old legacy HTTP processes did not hold a lifetime lock. A
                # new server cannot infer their death from a missing queue row.
                conn.execute('ALTER TABLE resource_reservations ADD COLUMN '
                             'recovery_version INTEGER NOT NULL DEFAULT 0 CHECK(recovery_version IN (0,1))')
            conn.execute('CREATE INDEX IF NOT EXISTS resource_reservations_owner '
                         'ON resource_reservations(owner_id,active)')

    @staticmethod
    def _table(conn, name):
        return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (name,)).fetchone() is not None

    @staticmethod
    def _integer(value):
        if type(value) is not int or value < 0:
            raise ValueError('Expected a nonnegative integer.')
        return value

    def _limits(self):
        return {
            'pending_jobs': self._integer(getattr(self.settings, 'max_owner_pending_jobs', 32)),
            'storage_bytes': self._integer(getattr(self.settings, 'max_owner_storage_bytes', 20 * 1024 ** 3)),
        }

    @staticmethod
    def _write_transaction(conn):
        if not conn.in_transaction:
            raise RuntimeError('Quota admission requires an existing write transaction.')
        # sqlite3 exposes no BEGIN-mode getter. This zero-row write obtains the
        # write lock even if a caller supplied BEGIN rather than BEGIN IMMEDIATE;
        # a stale read transaction fails closed instead of admitting a race.
        try:
            conn.execute('UPDATE resource_reservations SET active=active WHERE 0')
        except sqlite3.Error:
            raise HTTPException(503, _UNAVAILABLE) from None

    @staticmethod
    def _component(value):
        if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
            raise ValueError('Invalid storage identifier.')
        # Windows device names remain special even without a suffix.
        if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
                             *(f'LPT{i}' for i in range(1, 10))}:
            raise ValueError('Invalid storage identifier.')
        return value

    @staticmethod
    def _stat(path):
        try:
            value = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(value.st_mode) or getattr(value, 'st_file_attributes', 0) & _REPARSE_POINT:
            raise ValueError('Redirected storage path.')
        if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
            raise ValueError('Unsupported storage entry.')
        return value

    def _checked_path(self, *components):
        path = self.root.joinpath(*components)
        # lstat every ancestor: checking the leaf alone would miss a junction
        # or a symlink at uploads/, analyses/, or the configured storage root.
        for ancestor in reversed((path, *path.parents)):
            value = self._stat(ancestor)
            if value is not None and ancestor != path and not stat.S_ISDIR(value.st_mode):
                raise ValueError('Invalid storage parent.')
        return path

    def _bytes(self, path, *, input_names=frozenset(), depth=0):
        """Stream entries, counting hard links per retained path, never per inode."""
        if depth > 128:
            raise ValueError('Storage nesting exceeds supported depth.')
        value = self._stat(path)
        if value is None:
            return 0, 0
        if stat.S_ISREG(value.st_mode):
            amount = self._integer(value.st_size)
            return amount, amount if path.name in input_names else 0
        total = source = 0
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    amount, payload = self._bytes(Path(entry.path), input_names=(input_names if depth == 0 else frozenset()),
                                                  depth=depth + 1)
                    total += amount
                    source += payload
        except FileNotFoundError:
            # A worker can remove completed scratch between directory reads.
            return 0, 0
        return total, source

    def snapshot(self, conn, owner_id):
        """Return fresh counters on the supplied connection; no new connections.

        SQL UNION and cursor iteration bound Python memory by directory depth,
        rather than the owner's historical job count. Missing optional tables
        allow snapshots before the resumable-upload router has been installed.
        """
        try:
            limits = self._limits()
            uploads = self._table(conn, 'upload_sessions')
            corrections = self._table(conn, 'analysis_corrections')
            analysis_sources = ['SELECT id AS analysis_id FROM analyses WHERE owner_id=?',
                                'SELECT analysis_id FROM resource_reservations WHERE owner_id=?']
            pending_sources = ["SELECT id AS analysis_id FROM analyses WHERE owner_id=? "
                               "AND status IN ('UPLOADED','PROCESSING')",
                               'SELECT r.analysis_id FROM resource_reservations r WHERE r.owner_id=? AND r.active=1 '
                               'AND NOT EXISTS(SELECT 1 FROM analyses a WHERE a.id=r.analysis_id)']
            expected_sources = ['SELECT size_bytes FROM resource_reservations WHERE owner_id=? '
                                'AND analysis_id=? AND active=1']
            if uploads:
                analysis_sources.append('SELECT analysis_id FROM upload_sessions WHERE owner_id=?')
                pending_sources.append("SELECT s.analysis_id FROM upload_sessions s WHERE s.owner_id=? "
                                       "AND s.status='UPLOADING' AND NOT EXISTS"
                                       '(SELECT 1 FROM analyses a WHERE a.id=s.analysis_id)')
                expected_sources.append("SELECT size_bytes FROM upload_sessions WHERE owner_id=? "
                                        "AND analysis_id=? AND status='UPLOADING'")
            pending_analyses = conn.execute('SELECT COUNT(*) FROM (' + ' UNION '.join(pending_sources) + ')',
                                            (owner_id,) * len(pending_sources)).fetchone()[0]
            pending_corrections = 0
            if corrections:
                pending_corrections = conn.execute("SELECT COUNT(*) FROM analysis_corrections WHERE owner_id=? "
                                                   "AND status IN ('QUEUED','PROCESSING')", (owner_id,)).fetchone()[0]
            stored = reserved = 0
            for row in conn.execute(' UNION '.join(analysis_sources), (owner_id,) * len(analysis_sources)):
                ident = self._component(row[0])
                amount, source_bytes = self._bytes(self._checked_path('uploads', ident),
                    input_names=frozenset({'upload.part', 'original.mp4', 'original.mov', 'original.avi'}))
                stored += amount
                stored += self._bytes(self._checked_path('analyses', ident))[0]
                stored += self._bytes(self._checked_path('analyses', ident + '.annotated_web_credit.json'))[0]
                if uploads:
                    for session in conn.execute('SELECT id FROM upload_sessions WHERE owner_id=? AND analysis_id=?',
                                                (owner_id, ident)):
                        session_id = self._component(session[0])
                        amount, payload = self._bytes(self._checked_path('resumable', session_id),
                                                     input_names=frozenset({'upload.part'}))
                        stored += amount
                        source_bytes += payload
                # Queue insertion consumes the whole reservation, including
                # crash windows before session/reservation acknowledgement.
                if conn.execute('SELECT 1 FROM analyses WHERE id=?', (ident,)).fetchone() is None:
                    expected = conn.execute('SELECT MAX(size_bytes) FROM (' + ' UNION ALL '.join(expected_sources) + ')',
                                            (owner_id, ident) * len(expected_sources)).fetchone()[0]
                    if expected is not None:
                        reserved += max(0, self._integer(expected) - source_bytes)
            if corrections:
                for row in conn.execute('SELECT id FROM analysis_corrections WHERE owner_id=?', (owner_id,)):
                    stored += self._bytes(self._checked_path('corrections', self._component(row[0])))[0]
            return {'pending_jobs': pending_analyses + pending_corrections,
                    'pending_analyses': pending_analyses, 'pending_corrections': pending_corrections,
                    'stored_bytes': stored, 'reserved_input_bytes': reserved,
                    'committed_bytes': stored + reserved, 'limits': limits}
        except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError):
            raise HTTPException(503, _UNAVAILABLE) from None

    def check(self, conn, owner_id, *, additional_jobs=0, additional_bytes=0):
        self._write_transaction(conn)
        additional_jobs = self._integer(additional_jobs)
        additional_bytes = self._integer(additional_bytes)
        usage = self.snapshot(conn, owner_id)
        if usage['pending_jobs'] + additional_jobs > usage['limits']['pending_jobs']:
            raise HTTPException(429, 'Your pending-work limit has been reached. Wait for active work to finish.')
        if usage['committed_bytes'] + additional_bytes > usage['limits']['storage_bytes']:
            raise HTTPException(507, 'Your storage limit has been reached.')
        return usage

    def check_storage(self, conn, owner_id):
        """Worker stages enforce storage even after a pending-work limit changes."""
        usage = self.snapshot(conn, owner_id)
        if usage['committed_bytes'] > usage['limits']['storage_bytes']:
            raise HTTPException(507, 'Your storage limit has been reached.')
        return usage

    def reserve(self, conn, analysis_id, owner_id, size_bytes, kind='legacy', *, recovery_version=0):
        self._write_transaction(conn)
        ident = self._component(analysis_id)
        size_bytes = self._integer(size_bytes)
        if kind not in {'legacy', 'retry'} or not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Invalid resource reservation.')
        if type(recovery_version) is not int or recovery_version not in (0, 1):
            raise ValueError('Invalid reservation recovery protocol.')
        old = conn.execute('SELECT owner_id,size_bytes,kind,active FROM resource_reservations WHERE analysis_id=?',
                           (ident,)).fetchone()
        if old is not None:
            if tuple(old) == (owner_id, size_bytes, kind, 1):
                return self.snapshot(conn, owner_id)
            raise HTTPException(409, 'This upload reservation cannot be reused.')
        if conn.execute('SELECT 1 FROM analyses WHERE id=?', (ident,)).fetchone() is not None:
            raise HTTPException(409, 'This upload reservation cannot be reused.')
        now = datetime.now(timezone.utc).isoformat()
        # Tentative insertion lets the normal snapshot account for both an
        # incoming payload and any crash files at this identity. A failed
        # admission rolls back only this reservation, even if caught by caller.
        conn.execute('SAVEPOINT resource_quota_reserve')
        try:
            conn.execute('INSERT INTO resource_reservations '
                         '(analysis_id,owner_id,size_bytes,kind,active,created_at,updated_at,recovery_version) '
                         'VALUES (?,?,?,?,1,?,?,?)',
                         (ident, owner_id, size_bytes, kind, now, now, recovery_version))
            usage = self.check(conn, owner_id)
        except Exception:
            conn.execute('ROLLBACK TO resource_quota_reserve')
            conn.execute('RELEASE resource_quota_reserve')
            raise
        conn.execute('RELEASE resource_quota_reserve')
        return usage

    def release(self, conn, analysis_id):
        self._write_transaction(conn)
        conn.execute('UPDATE resource_reservations SET active=0,updated_at=? WHERE analysis_id=? AND active=1',
                     (datetime.now(timezone.utc).isoformat(), self._component(analysis_id)))

    def reservation_lock(self, analysis_id):
        """Hold before reserving a version-1 legacy request until settlement.

        Never unlink lockfiles: a second inode would split the OS-backed lock.
        This lock is independent of database connections and the worker lock.
        """
        ident = self._component(analysis_id)
        directory = self._checked_path('reservation-locks')
        directory.mkdir(exist_ok=True)
        return FileLock(str(self._checked_path('reservation-locks', ident + '.lock')), timeout=15)

    def _recovery_identity(self, conn, row):
        """Return a stable lock identity only for a known lifetime protocol."""
        if row['kind'] == 'legacy':
            return ('legacy', row['analysis_id']) if row['recovery_version'] == 1 else None
        if not self._table(conn, 'analysis_retry_intents'):
            return None
        intent = conn.execute('SELECT * FROM analysis_retry_intents WHERE analysis_id=?',
                              (row['analysis_id'],)).fetchone()
        if (intent is None or intent['owner_id'] != row['owner_id']
                or intent['size_bytes'] != row['size_bytes']):
            return None
        from src.api.retry_intents import KEY
        if not isinstance(intent['intent_key'], str) or KEY.fullmatch(intent['intent_key']) is None:
            return None
        source = conn.execute('SELECT owner_id FROM analyses WHERE id=?',
                              (intent['source_analysis_id'],)).fetchone()
        if source is None or source['owner_id'] != row['owner_id']:
            return None
        # Retry requests already held this owner/key lock in previous versions.
        return ('retry', intent['owner_id'], intent['intent_key'], intent['source_analysis_id'])

    def reconcile_owner(self, owner_id):
        """Release provably abandoned reservations, retaining every owned byte.

        Nonblocking request locks precede each short writer transaction. Never
        infer liveness from age, heartbeat or a missing analysis. Recheck under
        the writer lock so admission/acceptance cannot race the decision. Each
        row commits independently; retries after a partial failure are safe.
        """
        if not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Invalid reservation owner.')
        counts = {'reconciled_reservations': 0, 'busy_reservations': 0,
                  'unverified_reservations': 0}
        previous = ''
        try:
            while True:
                with self.jobs.connect() as conn:
                    row = conn.execute('SELECT * FROM resource_reservations WHERE owner_id=? '
                                       'AND active=1 AND analysis_id>? ORDER BY analysis_id LIMIT 1',
                                       (owner_id, previous)).fetchone()
                    if row is None:
                        break
                    previous = self._component(row['analysis_id'])
                    identity = self._recovery_identity(conn, row)
                if identity is None:
                    counts['unverified_reservations'] += 1
                    continue
                if identity[0] == 'legacy':
                    lock = self.reservation_lock(previous)
                else:
                    from src.api.retry_intents import retry_lock_path
                    lock = FileLock(str(retry_lock_path(self.jobs, owner_id, identity[2])), timeout=0)
                try:
                    lock.acquire(timeout=0)
                except Timeout:
                    counts['busy_reservations'] += 1
                    continue
                try:
                    with self.jobs.connect() as conn:
                        conn.execute('BEGIN IMMEDIATE')
                        current = conn.execute('SELECT * FROM resource_reservations '
                                               'WHERE analysis_id=? AND owner_id=? AND active=1',
                                               (previous, owner_id)).fetchone()
                        if current is None:
                            continue
                        if (dict(current) != dict(row)
                                or self._recovery_identity(conn, current) != identity):
                            counts['unverified_reservations'] += 1
                            continue
                        # A reservation must never retire an independently
                        # resumable session, including a cross-owner collision.
                        if self._table(conn, 'upload_sessions') and conn.execute(
                                'SELECT 1 FROM upload_sessions WHERE analysis_id=?', (previous,)).fetchone():
                            counts['unverified_reservations'] += 1
                            continue
                        accepted = conn.execute('SELECT * FROM analyses WHERE id=?', (previous,)).fetchone()
                        if accepted is not None and (accepted['owner_id'] != owner_id or (
                                identity[0] == 'retry' and accepted['retry_of'] != identity[3])):
                            counts['unverified_reservations'] += 1
                            continue
                        # Accepted work remains charged by analyses. Inactive
                        # reservation rows still attribute all retained paths.
                        conn.execute('UPDATE resource_reservations SET active=0,updated_at=? '
                                     'WHERE analysis_id=? AND owner_id=? AND active=1',
                                     (datetime.now(timezone.utc).isoformat(), previous, owner_id))
                    counts['reconciled_reservations'] += 1
                finally:
                    lock.release()
            with self.jobs.connect() as conn:
                conn.execute('BEGIN')
                return {**counts, 'usage': self.snapshot(conn, owner_id)}
        except (OSError, sqlite3.Error, ValueError, TypeError, OverflowError):
            raise HTTPException(503, _UNAVAILABLE) from None
