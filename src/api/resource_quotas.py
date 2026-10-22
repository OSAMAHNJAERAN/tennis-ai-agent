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


_COMPONENT = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
_REPARSE_POINT = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
_UNAVAILABLE = 'Storage usage could not be checked. Please try again.'


class ResourceQuotas:
    def __init__(self, jobs, settings):
        self.jobs = jobs
        self.root = Path(jobs.root)
        self.settings = settings
        with jobs.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS resource_reservations (
                analysis_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                kind TEXT NOT NULL CHECK(kind IN ('legacy','retry')),
                active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
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

    def reserve(self, conn, analysis_id, owner_id, size_bytes, kind='legacy'):
        self._write_transaction(conn)
        ident = self._component(analysis_id)
        size_bytes = self._integer(size_bytes)
        if kind not in {'legacy', 'retry'} or not isinstance(owner_id, str) or not owner_id:
            raise ValueError('Invalid resource reservation.')
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
                         '(analysis_id,owner_id,size_bytes,kind,active,created_at,updated_at) VALUES (?,?,?,?,1,?,?)',
                         (ident, owner_id, size_bytes, kind, now, now))
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
