"""Durable single-host analysis queue. No model imports in the HTTP process."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import uuid

LOCAL_OWNER = 'local'


class JobCancelled(Exception):
    """Processing reached a safe boundary after an owner requested cancellation."""

REPO_ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Settings:
    root: Path
    config: Path
    max_upload_bytes: int = 1024 * 1024 * 1024
    max_duration_seconds: int = 7200
    require_auth: bool = False
    secure_cookies: bool = False
    max_owner_pending_jobs: int = 32
    max_owner_storage_bytes: int = 20 * 1024 ** 3

    def __post_init__(self):
        for name in ('max_upload_bytes', 'max_duration_seconds',
                     'max_owner_pending_jobs', 'max_owner_storage_bytes'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')

    @classmethod
    def from_env(cls):
        return cls(
            Path(os.getenv("TENNIS_STORAGE_ROOT", str(REPO_ROOT / "outputs/product"))).resolve(),
            Path(os.getenv("TENNIS_PIPELINE_CONFIG", str(REPO_ROOT / "configs/production/yolo26.yaml"))).resolve(),
            int(os.getenv("TENNIS_MAX_UPLOAD_BYTES", str(1024 ** 3))),
            int(os.getenv("TENNIS_MAX_DURATION_SECONDS", "7200")),
            os.getenv('TENNIS_REQUIRE_AUTH', '1') == '1',
            os.getenv('TENNIS_SECURE_COOKIES', '0') == '1',
            int(os.getenv('TENNIS_MAX_OWNER_PENDING_JOBS', '32')),
            int(os.getenv('TENNIS_MAX_OWNER_STORAGE_BYTES', str(20 * 1024 ** 3))),
        )


class JobStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "analyses.sqlite3"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, extension TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('UPLOADED','PROCESSING','COMPLETED','FAILED')),
                stage TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
                error TEXT, size_bytes INTEGER NOT NULL, config_path TEXT NOT NULL
            )""")
            from src.api.product import migrate_product
            migrate_product(conn)
            for table in ('analyses', 'matches', 'players'):
                columns = {row['name'] for row in conn.execute(f'PRAGMA table_info({table})')}
                if 'owner_id' not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'")
                conn.execute(f'CREATE INDEX IF NOT EXISTS {table}_owner ON {table}(owner_id)')
            columns = {row['name'] for row in conn.execute('PRAGMA table_info(analyses)')}
            for name, definition in (('cancel_requested', 'INTEGER NOT NULL DEFAULT 0'),
                                     ('attempt', 'INTEGER NOT NULL DEFAULT 0'),
                                     ('retry_of', 'TEXT'), ('heartbeat_at', 'TEXT')):
                if name not in columns:
                    conn.execute(f'ALTER TABLE analyses ADD COLUMN {name} {definition}')
            from src.api.corrections import migrate_corrections
            migrate_corrections(conn)
            conn.execute('PRAGMA user_version=4')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def new_id(self):
        return "match_" + uuid.uuid4().hex

    def upload_dir(self, analysis_id):
        return self.root / "uploads" / analysis_id

    def output_dir(self, analysis_id):
        return self.root / "analyses" / analysis_id

    def create(self, analysis_id, filename, extension, size, config, match_id=None, owner_id=LOCAL_OWNER, retry_of=None):
        with self.connect() as conn:
            conn.execute("""INSERT INTO analyses
                (id,filename,extension,status,stage,created_at,size_bytes,config_path,owner_id,retry_of)
                VALUES (?,?,?,'UPLOADED','Waiting for the analysis worker',?,?,?,?,?)""",
                (analysis_id, filename, extension, utc_now(), size, str(config), owner_id, retry_of))
            from src.api.product import attach_analysis
            attach_analysis(conn, analysis_id, match_id, owner_id)
        return self.get(analysis_id)

    def get(self, analysis_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
        return dict(row) if row else None

    def list(self, owner_id=LOCAL_OWNER):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM analyses WHERE owner_id=? ORDER BY created_at DESC LIMIT 200", (owner_id,))]

    def claim(self):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM analyses WHERE status='UPLOADED' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            conn.execute("UPDATE analyses SET status='PROCESSING', started_at=?,heartbeat_at=?,attempt=attempt+1, stage='Validating video' WHERE id=?",
                         (utc_now(), utc_now(), row['id']))
        return self.get(row['id'])

    def stage(self, analysis_id, stage, step=0):
        with self.connect() as conn:
            row = conn.execute('SELECT cancel_requested FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            if row and row['cancel_requested']:
                raise JobCancelled('Analysis cancelled by its owner.')
            conn.execute("UPDATE analyses SET stage=?,step=?,heartbeat_at=? WHERE id=? AND status='PROCESSING'", (stage, step, utc_now(), analysis_id))

    def finish(self, analysis_id, error=None):
        with self.connect() as conn:
            cancelled = conn.execute('SELECT cancel_requested FROM analyses WHERE id=?', (analysis_id,)).fetchone()
            if cancelled and cancelled[0]:
                error = 'Analysis cancelled by its owner.'
            conn.execute("UPDATE analyses SET status=?,stage=?,step=?,completed_at=?,error=? WHERE id=?",
                         ('FAILED' if error else 'COMPLETED', 'Analysis failed' if error else 'Analysis complete',
                          0 if error else 11, utc_now(), error, analysis_id))

    def recover_interrupted(self, validate_completed_checkpoint=None):
        # Called only while holding the exclusive worker lock. A live inference
        # process retains this lock, including across HTTP server restarts.
        if validate_completed_checkpoint is None:
            with self.connect() as conn:
                conn.execute("""UPDATE analyses SET status='FAILED', stage='Analysis interrupted', completed_at=?,
                    error='Processing was interrupted. Retry uses the retained original video.' WHERE status='PROCESSING'""", (utc_now(),))
            return
        previous_id = ''
        while True:
            # Validate hashes outside a database transaction, one job at a time.
            # HTTP ownership/cancellation operations remain available meanwhile.
            with self.connect() as conn:
                row = conn.execute("SELECT * FROM analyses WHERE status='PROCESSING' AND id>? ORDER BY id LIMIT 1",
                                   (previous_id,)).fetchone()
            if row is None:
                return
            job = dict(row)
            previous_id = job['id']
            valid = False
            if not job['cancel_requested'] and job['attempt'] < 3:
                try:
                    valid = validate_completed_checkpoint(job) is True
                except Exception:
                    valid = False
            with self.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                current = conn.execute('SELECT status,cancel_requested,attempt FROM analyses WHERE id=?',
                                       (job['id'],)).fetchone()
                if not current or current['status'] != 'PROCESSING':
                    continue
                if valid and not current['cancel_requested'] and current['attempt'] < 3:
                    conn.execute("UPDATE analyses SET status='UPLOADED',stage='Resuming saved analysis finalization',"
                                 "step=11,completed_at=NULL,error=NULL,heartbeat_at=NULL WHERE id=?", (job['id'],))
                else:
                    error = ('Analysis cancelled by its owner.' if current['cancel_requested'] else
                             'Processing was interrupted. Retry uses the retained original video.')
                    conn.execute("UPDATE analyses SET status='FAILED',stage='Analysis interrupted',"
                                 "step=0,completed_at=?,error=? WHERE id=?", (utc_now(), error, job['id']))

    def cancel(self, analysis_id, owner_id):
        from fastapi import HTTPException
        with self.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM analyses WHERE id=? AND owner_id=?', (analysis_id, owner_id)).fetchone()
            if not row:
                raise HTTPException(404, 'Analysis not found.')
            if row['status'] in ('COMPLETED', 'FAILED'):
                return dict(row)
            conn.execute('UPDATE analyses SET cancel_requested=1 WHERE id=?', (analysis_id,))
            if row['status'] == 'UPLOADED':
                conn.execute("UPDATE analyses SET status='FAILED',stage='Analysis cancelled',error='Analysis cancelled by its owner.',completed_at=? WHERE id=?", (utc_now(), analysis_id))
        return self.get(analysis_id)


def public_job(job):
    return {**{key: value for key, value in job.items() if key not in {'config_path', 'extension', 'owner_id'}},
            'cancel_requested': bool(job.get('cancel_requested')),
            'cancelled': bool(job.get('cancel_requested')) and job['status'] == 'FAILED'}
