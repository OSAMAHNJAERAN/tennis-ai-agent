"""Durable owner retry identities, serialized independently of worker locks."""
from contextlib import contextmanager
import hashlib
import re

from fastapi import HTTPException
from filelock import FileLock, Timeout

from src.api.jobs import utc_now

KEY = re.compile(r'[A-Za-z0-9_-]{16,128}\Z')


class RetryIntents:
    def __init__(self, jobs, quotas):
        self.jobs, self.quotas = jobs, quotas
        with jobs.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS analysis_retry_intents (
                owner_id TEXT NOT NULL, intent_key TEXT NOT NULL,
                source_analysis_id TEXT NOT NULL, analysis_id TEXT NOT NULL UNIQUE,
                size_bytes INTEGER NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(owner_id,intent_key))''')

    @contextmanager
    def lock(self, owner_id, key):
        if not isinstance(key, str) or KEY.fullmatch(key) is None:
            raise HTTPException(422, 'Send a stable Idempotency-Key of 16–128 letters, digits, underscores or hyphens.')
        from src.api.analysis_generation import _safe_path
        directory = _safe_path(self.jobs.root/'retry-locks')
        directory.mkdir(exist_ok=True)
        name = hashlib.sha256((owner_id+'\0'+key).encode('utf-8')).hexdigest()+'.lock'
        try:
            with FileLock(str(_safe_path(directory/name)), timeout=15):
                yield
        except Timeout as error:
            raise HTTPException(409, 'This retry request is still being accepted. Repeat it with the same Idempotency-Key.') from error

    @staticmethod
    def _matching(row, source_id):
        if row is not None and row['source_analysis_id'] != source_id:
            raise HTTPException(409, 'This Idempotency-Key already belongs to another retry request.')
        return dict(row) if row is not None else None

    def selected(self, owner_id, key, source_id):
        with self.jobs.connect() as conn:
            row = conn.execute('SELECT * FROM analysis_retry_intents WHERE owner_id=? AND intent_key=?',
                               (owner_id, key)).fetchone()
        return self._matching(row, source_id)

    def accepted(self, selection, owner_id):
        if selection is None:
            return None
        job = self.jobs.get(selection['analysis_id'])
        if job is not None and (job['owner_id'] != owner_id or job['retry_of'] != selection['source_analysis_id']):
            raise HTTPException(503, 'The saved retry acknowledgement could not be verified.')
        return job

    def admit(self, owner_id, key, source_id, size):
        """Reserve identity and capacity atomically; reuse a pre-queue crash ID."""
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM analysis_retry_intents WHERE owner_id=? AND intent_key=?',
                               (owner_id, key)).fetchone()
            selection = self._matching(row, source_id)
            if selection is None:
                ident = self.jobs.new_id()
                self.quotas.reserve(conn, ident, owner_id, size, kind='retry')
                conn.execute('INSERT INTO analysis_retry_intents VALUES (?,?,?,?,?,?)',
                             (owner_id, key, source_id, ident, size, utc_now()))
                return ident
            if selection['size_bytes'] != size:
                raise HTTPException(409, 'The retained source changed since this retry request. Upload the video again.')
            ident = selection['analysis_id']
            reservation = conn.execute('SELECT * FROM resource_reservations WHERE analysis_id=?', (ident,)).fetchone()
            if (reservation is None or reservation['owner_id'] != owner_id
                    or reservation['size_bytes'] != size or reservation['kind'] != 'retry'):
                raise HTTPException(503, 'The saved retry reservation could not be verified.')
            if not reservation['active']:
                conn.execute('UPDATE resource_reservations SET active=1,updated_at=? WHERE analysis_id=?',
                             (utc_now(), ident))
                self.quotas.check(conn, owner_id)
            else:
                # An admitted incoming request keeps its pending slot, including
                # after a lower configured pending limit. Storage still applies.
                self.quotas.check_storage(conn, owner_id)
            return ident
