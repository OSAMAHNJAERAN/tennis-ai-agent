"""Owner-scoped resumable byte uploads, independent batch jobs and provenance.

The database offset is authoritative. After a crash, surplus file bytes are
truncated before the next append. Finishing is idempotent across HTTP restarts.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal
import uuid

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, Request
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from src.api.auth import current_owner
from src.api.jobs import public_job, utc_now
from src.api.creation_requests import CreationRequests
from src.api.product import MatchMetadata, ProductStore

CHUNK_LIMIT = 8 * 1024 * 1024
EXTENSIONS = {'.mp4', '.mov', '.avi'}
LOGGER = logging.getLogger(__name__)


def checked_filename(value):
    filename = value.replace('\\', '/').rsplit('/', 1)[-1].strip()
    extension = Path(filename).suffix.lower()
    if extension not in EXTENSIONS or not filename or len(filename) > 200 or any(ord(c) < 32 for c in filename):
        raise HTTPException(415, 'Choose an MP4, MOV or AVI video with a valid filename.')
    return filename, extension


def container_valid(header, extension):
    return len(header) >= 12 and ((header[:4] == b'RIFF' and header[8:12] == b'AVI ') if extension == '.avi'
                                  else header[4:8] in {b'ftyp', b'moov', b'mdat', b'wide', b'free', b'skip'})


class UploadSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    filename: str = Field(min_length=1, max_length=500)
    size_bytes: int = Field(ge=12)
    match_id: str | None = Field(default=None, max_length=100)
    batch_id: str | None = Field(default=None, max_length=100)
    clip_index: int | None = Field(default=None, ge=0)
    start_offset_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class BatchSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['independent', 'same_match'] = 'independent'
    files: list[UploadSpec] = Field(min_length=1, max_length=16)
    metadata: MatchMetadata | None = None

    @model_validator(mode='after')
    def explicit_order(self):
        if self.mode == 'same_match':
            indices = [item.clip_index for item in self.files]
            if sorted(index for index in indices if index is not None) != list(range(len(self.files))):
                raise ValueError('Same-match clips require unique clip_index values from 0 to file count minus one.')
        if any(item.batch_id is not None for item in self.files):
            raise ValueError('Batch file entries cannot refer to a different batch.')
        if self.metadata is not None and any(item.match_id is not None for item in self.files):
            raise ValueError('Use batch metadata or existing upload drafts, rather than both.')
        return self


def upload_router(jobs, settings, ensure_worker, *, quotas=None):
    if quotas is None:
        from src.api.resource_quotas import ResourceQuotas
        quotas = ResourceQuotas(jobs, settings)
    creations = CreationRequests(jobs)
    with jobs.connect() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS upload_batches (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, mode TEXT NOT NULL, created_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS upload_sessions (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, filename TEXT NOT NULL, extension TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'UPLOADING', match_id TEXT, batch_id TEXT,
            analysis_id TEXT NOT NULL, clip_index INTEGER, start_offset_seconds REAL,
            checksum TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(upload_sessions)')}
        if 'error' not in columns:
            conn.execute('ALTER TABLE upload_sessions ADD COLUMN error TEXT')
        conn.execute('CREATE INDEX IF NOT EXISTS uploads_owner ON upload_sessions(owner_id)')
    router = APIRouter(prefix='/api')

    def check_storage(owner_id):
        with jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            quotas.check_storage(conn, owner_id)

    def raw(upload_id, owner_id):
        with jobs.connect() as conn:
            row = conn.execute('SELECT * FROM upload_sessions WHERE id=? AND owner_id=?', (upload_id, owner_id)).fetchone()
        if not row:
            raise HTTPException(404, 'Upload not found.')
        return dict(row)

    def presented(row):
        return {**{key: value for key, value in row.items() if key not in {'owner_id', 'extension', 'match_id'}},
                'analysis_id': row['analysis_id'] if row['status'] == 'COMPLETED' else None}

    def directory(row):
        return jobs.root / 'resumable' / row['id']

    def validate(spec, owner_id, conn):
        checked_filename(spec.filename)
        if spec.size_bytes > settings.max_upload_bytes:
            raise HTTPException(413, 'Video exceeds the upload size limit.')
        if spec.match_id and not conn.execute("SELECT id FROM matches WHERE id=? AND owner_id=? AND source='upload' AND analysis_id IS NULL", (spec.match_id, owner_id)).fetchone():
            raise HTTPException(404, 'Upload draft not found.')
        if spec.batch_id and not conn.execute('SELECT id FROM upload_batches WHERE id=? AND owner_id=?', (spec.batch_id, owner_id)).fetchone():
            raise HTTPException(404, 'Batch not found.')

    def insert(conn, spec, owner_id, batch_id=None):
        filename, extension = checked_filename(spec.filename)
        now, upload_id = utc_now(), 'upload_' + uuid.uuid4().hex
        analysis_id = jobs.new_id()
        conn.execute('''INSERT INTO upload_sessions
            (id,owner_id,filename,extension,size_bytes,match_id,batch_id,analysis_id,clip_index,start_offset_seconds,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
                     (upload_id, owner_id, filename, extension, spec.size_bytes, spec.match_id, batch_id or spec.batch_id,
                      analysis_id, spec.clip_index, spec.start_offset_seconds, now, now))
        return upload_id

    @router.post('/uploads', status_code=201)
    def create(spec: UploadSpec, request: Request, owner_id: str = Depends(current_owner)):
        key, fingerprint = creations.key(request), creations.fingerprint(spec)
        with jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            saved = creations.find(conn, owner_id, key, 'upload', fingerprint)
            if saved is None:
                validate(spec, owner_id, conn)
                quotas.check(conn, owner_id, additional_jobs=1, additional_bytes=spec.size_bytes)
                upload_id = insert(conn, spec, owner_id)
                creations.save(conn, owner_id, key, 'upload', fingerprint, {'id': upload_id})
            else:
                upload_id = saved['id']
        return presented(raw(upload_id, owner_id))

    @router.get('/uploads/{upload_id}')
    def get(upload_id: str, owner_id: str = Depends(current_owner)):
        return presented(raw(upload_id, owner_id))

    def append(upload_id, owner_id, offset, temporary, size):
        row = raw(upload_id, owner_id)
        with FileLock(str(directory(row) / 'append.lock'), timeout=15):
            row = raw(upload_id, owner_id)
            if row['status'] != 'UPLOADING' or offset != row['offset']:
                raise HTTPException(409, 'Upload offset changed. Read the saved offset before retrying.')
            if offset + size > row['size_bytes']:
                raise HTTPException(413, 'Chunk exceeds the declared file size.')
            partial = directory(row) / 'upload.part'
            with partial.open('r+b' if partial.exists() else 'w+b') as target:
                target.seek(0, os.SEEK_END)
                if target.tell() < offset:
                    raise HTTPException(409, 'Stored upload bytes are incomplete. Start a new upload.')
                target.truncate(offset)
                target.seek(offset)
                with temporary.open('rb') as source:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            with jobs.connect() as conn:
                conn.execute('UPDATE upload_sessions SET offset=?,updated_at=? WHERE id=?', (offset + size, utc_now(), upload_id))
        return presented(raw(upload_id, owner_id))

    @router.patch('/uploads/{upload_id}')
    async def patch(upload_id: str, request: Request, owner_id: str = Depends(current_owner)):
        row = raw(upload_id, owner_id)
        offset = request.headers.get('upload-offset', '')
        if not offset.isdigit():
            raise HTTPException(400, 'Send the saved Upload-Offset header.')
        if int(offset) != row['offset'] or row['status'] != 'UPLOADING':
            raise HTTPException(409, 'Upload offset changed. Read the saved offset before retrying.')
        declared = request.headers.get('content-length')
        if declared and (not declared.isdigit() or int(declared) > CHUNK_LIMIT):
            raise HTTPException(413, 'Use chunks of at most 8 MiB.')
        target = directory(row)
        target.mkdir(parents=True, exist_ok=True)
        temporary = target / ('chunk_' + uuid.uuid4().hex + '.part')
        # Bound uncommitted transfer scratch to one chunk per admitted upload,
        # including while its async request body is still arriving.
        receive_lock = FileLock(str(target / 'receive.lock'), timeout=0, thread_local=False)
        acquired = False
        try:
            await run_in_threadpool(receive_lock.acquire)
            acquired = True
            await run_in_threadpool(check_storage, owner_id)
            size = 0
            async with aiofiles.open(temporary, 'xb') as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > CHUNK_LIMIT or int(offset) + size > row['size_bytes']:
                        raise HTTPException(413, 'Chunk exceeds the upload limits.')
                    await handle.write(chunk)
            if not size:
                raise HTTPException(400, 'Send a non-empty upload chunk.')
            await run_in_threadpool(check_storage, owner_id)
            return await run_in_threadpool(append, upload_id, owner_id, int(offset), temporary, size)
        except Timeout as exc:
            raise HTTPException(409, 'Another request is updating this upload. Read its offset and retry.') from exc
        except OSError as exc:
            raise HTTPException(507, 'Upload could not be stored. Check available disk space.') from exc
        finally:
            try:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    # The durable offset remains authoritative. Failed cleanup
                    # stays owner-attributed without replacing its acknowledgement
                    # or the original request error with a cleanup exception.
                    LOGGER.exception('Upload chunk cleanup left retained bytes: %s', upload_id)
            finally:
                if acquired:
                    receive_lock.release()

    @router.post('/uploads/{upload_id}/complete', status_code=202)
    def complete(upload_id: str, owner_id: str = Depends(current_owner)):
        row = raw(upload_id, owner_id)
        target = directory(row)
        target.mkdir(parents=True, exist_ok=True)
        try:
            with FileLock(str(target / 'append.lock'), timeout=15):
                row = raw(upload_id, owner_id)
                existing = jobs.get(row['analysis_id'])
                if existing:
                    with jobs.connect() as conn:
                        conn.execute("UPDATE upload_sessions SET status='COMPLETED',error=NULL,updated_at=? WHERE id=?", (utc_now(), upload_id))
                    return public_job(existing)
                if row['status'] != 'UPLOADING' or row['offset'] != row['size_bytes']:
                    raise HTTPException(409, 'Finish uploading every byte before starting analysis.')
                check_storage(owner_id)
                upload_directory = jobs.upload_dir(row['analysis_id'])
                original = upload_directory / ('original' + row['extension'])
                source = original if original.exists() else target / 'upload.part'
                with source.open('rb') as handle:
                    if source.stat().st_size != row['size_bytes'] or not container_valid(handle.read(32), row['extension']):
                        raise HTTPException(415, 'The file is not a supported video container.')
                    handle.seek(0)
                    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
                upload_directory.mkdir(parents=True, exist_ok=True)
                if source != original:
                    source.replace(original)
                with jobs.connect() as conn:
                    conn.execute('UPDATE upload_sessions SET checksum=?,updated_at=? WHERE id=?', (digest, utc_now(), upload_id))
                job = jobs.create(row['analysis_id'], row['filename'], row['extension'], row['size_bytes'], settings.config,
                                  row['match_id'], owner_id)
                with jobs.connect() as conn:
                    conn.execute("UPDATE upload_sessions SET status='COMPLETED',error=NULL,checksum=?,updated_at=? WHERE id=?", (digest, utc_now(), upload_id))
        except HTTPException as exc:
            if exc.status_code == 415:
                with jobs.connect() as conn:
                    conn.execute("UPDATE upload_sessions SET status='FAILED',error=?,updated_at=? WHERE id=?", (exc.detail, utc_now(), upload_id))
            raise
        except Timeout as exc:
            raise HTTPException(409, 'Another request is updating this upload. Retry completion.') from exc
        except OSError as exc:
            raise HTTPException(507, 'Upload could not be finalized. Check available disk space.') from exc
        ensure_worker()
        return public_job(job)

    @router.delete('/uploads/{upload_id}')
    def cancel(upload_id: str, owner_id: str = Depends(current_owner)):
        row = raw(upload_id, owner_id)
        target = directory(row)
        target.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target / 'append.lock'), timeout=15):
            row = raw(upload_id, owner_id)
            if row['status'] == 'COMPLETED' or jobs.get(row['analysis_id']):
                raise HTTPException(409, 'This upload was accepted. Cancel its analysis job instead.')
            with jobs.connect() as conn:
                conn.execute("UPDATE upload_sessions SET status='CANCELLED',updated_at=? WHERE id=?", (utc_now(), upload_id))
            (target / 'upload.part').unlink(missing_ok=True)
            # A crash between rename and queue insertion can leave the owned
            # original; it is safe to remove only if no analysis was accepted.
            (jobs.upload_dir(row['analysis_id']) / ('original' + row['extension'])).unlink(missing_ok=True)
        return presented(raw(upload_id, owner_id))

    @router.post('/batches', status_code=201)
    def batch(spec: BatchSpec, request: Request, owner_id: str = Depends(current_owner)):
        key, fingerprint = creations.key(request), creations.fingerprint(spec)
        with jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            saved = creations.find(conn, owner_id, key, 'batch', fingerprint)
            if saved is None:
                for file in spec.files:
                    validate(file, owner_id, conn)
                metadata = ProductStore(jobs).validate_metadata(conn, spec.metadata, owner_id) if spec.metadata is not None else None
                quotas.check(conn, owner_id, additional_jobs=len(spec.files),
                             additional_bytes=sum(file.size_bytes for file in spec.files))
                batch_id = 'batch_' + uuid.uuid4().hex
                conn.execute('INSERT INTO upload_batches VALUES (?,?,?,?)', (batch_id, owner_id, spec.mode, utc_now()))
                ids = []
                for file in spec.files:
                    if metadata is not None:
                        match_id, now = 'session_' + uuid.uuid4().hex, utc_now()
                        conn.execute("INSERT INTO matches(id,source,metadata,created_at,updated_at,owner_id) VALUES (?,'upload',?,?,?,?)",
                                     (match_id, metadata, now, now, owner_id))
                        file = file.model_copy(update={'match_id': match_id})
                    ids.append(insert(conn, file, owner_id, batch_id))
                creations.save(conn, owner_id, key, 'batch', fingerprint,
                               {'id': batch_id, 'mode': spec.mode, 'upload_ids': ids})
            else:
                batch_id, ids = saved['id'], saved['upload_ids']
        return {'id': batch_id, 'mode': spec.mode, 'grouping_status': 'unverified', 'uploads': [presented(raw(id, owner_id)) for id in ids]}

    @router.get('/batches/{batch_id}')
    def batch_status(batch_id: str, owner_id: str = Depends(current_owner)):
        with jobs.connect() as conn:
            batch = conn.execute('SELECT * FROM upload_batches WHERE id=? AND owner_id=?', (batch_id, owner_id)).fetchone()
            rows = conn.execute('SELECT * FROM upload_sessions WHERE batch_id=? AND owner_id=? ORDER BY clip_index,created_at,id', (batch_id, owner_id)).fetchall()
        if not batch:
            raise HTTPException(404, 'Batch not found.')
        uploads = [{**presented(dict(row)), 'job': public_job(job) if (job := jobs.get(row['analysis_id'])) else None} for row in rows]
        checksums = [item['checksum'] for item in uploads if item['checksum']]
        duplicates = len(checksums) != len(set(checksums))
        return {'id': batch_id, 'mode': batch['mode'], 'created_at': batch['created_at'], 'uploads': uploads,
                'grouping_status': 'duplicates_detected' if duplicates else 'unverified',
                'unified_report': None, 'reason': 'Match continuity and clip overlap require independent verification before merging.'}

    return router
