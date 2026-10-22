"""Owner-scoped durable semantic corrections, independent of completed analyses.

HTTP admission reads sealed source evidence and never loads inference models.
Only the separately locked worker processes commands and publishes a complete
new generation; a queued or failed correction leaves the old analysis usable.
"""
from contextlib import contextmanager
import hashlib
import json
import re
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from filelock import FileLock, Timeout
from starlette.concurrency import run_in_threadpool

from src.api.auth import current_owner
from src.api.jobs import JobCancelled, LOCAL_OWNER, utc_now
from src.scoring.corrected_rule_rebuild import (
    _canonical, _digest, _initial_state, _plain, _replacement, _request,
    MAX_REQUEST_BYTES,
)


TOKEN = re.compile(r'[0-9a-f]{64}\Z')
TERMINAL = frozenset(('COMPLETED', 'FAILED', 'CONFLICT', 'CANCELLED'))
STATUSES = TERMINAL | {'QUEUED', 'PROCESSING'}
PUBLIC_FIELDS = ('id', 'analysis_id', 'status', 'stage', 'step', 'created_at',
    'started_at', 'completed_at', 'error', 'attempt', 'cancel_requested',
    'base_generation', 'base_analysis_revision', 'analysis_revision', 'generation')
AUTH_FIELDS = {'schema_version', 'analysis_id', 'owner_id', 'reviewer_id',
    'correction_id', 'request_sha256', 'authorized_at', 'authorization_method'}


def migrate_corrections(conn):
    """Called by the coordinated version-four JobStore initialization."""
    conn.execute('''CREATE TABLE IF NOT EXISTS analysis_corrections (
        id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, owner_id TEXT NOT NULL,
        command_id TEXT NOT NULL, command_json TEXT NOT NULL, request_sha256 TEXT NOT NULL,
        authorization_json TEXT NOT NULL,
        base_generation TEXT NOT NULL, base_analysis_revision INTEGER NOT NULL,
        analysis_revision INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
        base_rule_replay_sha256 TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('QUEUED','PROCESSING','COMPLETED','FAILED','CONFLICT','CANCELLED')),
        stage TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT, heartbeat_at TEXT,
        error TEXT, attempt INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
        generation TEXT, UNIQUE(analysis_id,owner_id,command_id))''')
    conn.execute('CREATE INDEX IF NOT EXISTS correction_queue ON analysis_corrections(status,created_at,id)')
    conn.execute('CREATE INDEX IF NOT EXISTS correction_owner ON analysis_corrections(owner_id,analysis_id,created_at)')


def public_correction(row):
    return {**{key: row[key] for key in PUBLIC_FIELDS},
            'cancel_requested': bool(row['cancel_requested'])}


def _internal(row):
    value = dict(row)
    command, authorization = json.loads(value['command_json']), json.loads(value['authorization_json'])
    command = _request(command)
    if (_digest(command) != value['request_sha256'] or type(authorization) is not dict
            or set(authorization) != AUTH_FIELDS
            or type(authorization['schema_version']) is not int or authorization['schema_version'] != 1
            or authorization['analysis_id'] != value['analysis_id']
            or authorization['owner_id'] != value['owner_id']
            or authorization['reviewer_id'] != value['owner_id'] or not value['owner_id']
            or authorization['correction_id'] != command['correction_id']
            or authorization['request_sha256'] != value['request_sha256']
            or not isinstance(authorization['authorized_at'], str) or not authorization['authorized_at']
            or authorization['authorization_method'] not in ('AUTHENTICATED_OWNER', 'EXPLICIT_LOCAL_OWNER')
            or (authorization['authorization_method'] == 'EXPLICIT_LOCAL_OWNER') != (value['owner_id'] == LOCAL_OWNER)
            or command['analysis_id'] != value['analysis_id'] or command['reviewer_id'] != value['owner_id']
            or command['correction_id'] != value['command_id']
            or command['base_generation'] != value['base_generation']
            or command['base_analysis_revision'] != value['base_analysis_revision']
            or value['analysis_revision'] != value['base_analysis_revision']+1
            or command['source_sha256'] != value['source_sha256']
            or command['base_rule_replay_sha256'] != value['base_rule_replay_sha256']):
        raise ValueError('Durable correction command or owner authorization binding differs')
    value['command'], value['authorization'] = command, authorization
    value['cancel_requested'] = bool(value['cancel_requested'])
    return value


class CorrectionStore:
    def __init__(self, jobs, *, quotas=None):
        self.jobs = jobs
        self.quotas = quotas
        with jobs.connect() as conn:
            migrate_corrections(conn)

    def _owned_analysis(self, analysis_id, owner_id, *, conn=None):
        if conn is None:
            with self.jobs.connect() as connection:
                return self._owned_analysis(analysis_id, owner_id, conn=connection)
        row = conn.execute('SELECT * FROM analyses WHERE id=? AND owner_id=?',
                           (analysis_id, owner_id)).fetchone()
        if row is None:
            raise HTTPException(404, 'Analysis not found.')
        if row['status'] != 'COMPLETED':
            raise HTTPException(409, 'Only a completed saved analysis can be corrected.')
        return dict(row)

    def get(self, correction_id, owner_id=None, *, internal=False):
        with self.jobs.connect() as conn:
            row = conn.execute('SELECT * FROM analysis_corrections WHERE id=?'+
                ('' if internal else ' AND owner_id=?'),
                (correction_id,) if internal else (correction_id, owner_id)).fetchone()
        if row is None:
            raise HTTPException(404, 'Correction not found.')
        return _internal(row) if internal else public_correction(row)

    def list(self, analysis_id, owner_id):
        self._owned_analysis(analysis_id, owner_id)
        with self.jobs.connect() as conn:
            return [public_correction(row) for row in conn.execute(
                'SELECT * FROM analysis_corrections WHERE analysis_id=? AND owner_id=? '
                'ORDER BY created_at DESC,id DESC LIMIT 200', (analysis_id, owner_id))]

    def _base(self, command):
        from src.api.analysis_generation import resolve_generation
        snapshot = resolve_generation(self.jobs.output_dir(command['analysis_id']))
        if (snapshot is None or snapshot.analysis_id != command['analysis_id']
                or snapshot.generation != command['base_generation']
                or snapshot.analysis_revision != command['base_analysis_revision']
                or snapshot.file_sha256('source.mp4') != command['source_sha256']
                or snapshot.file_sha256('native_rule_replay.json') != command['base_rule_replay_sha256']):
            raise HTTPException(409, 'The saved analysis changed. Refresh its review source before correcting it.')
        return snapshot

    @contextmanager
    def _source(self, snapshot):
        from src.api.analysis_generation import _safe_path
        from src.scoring.rule_replay_source import open_rule_replay_source
        if snapshot.resolve_file('native_rule_replay.json') is None or snapshot.resolve_file('source.mp4') is None:
            raise HTTPException(409, 'This saved generation has no retained native correction source.')
        root = _safe_path(self.jobs.output_dir(snapshot.analysis_id)/'.correction-admission')
        root.mkdir(exist_ok=True)
        owner_id = self.jobs.get(snapshot.analysis_id)['owner_id']
        locks = _safe_path(self.jobs.root/'correction-admission-locks')
        locks.mkdir(exist_ok=True)
        lock_name = hashlib.sha256(owner_id.encode('utf-8')).hexdigest()+'.lock'

        def check_storage():
            if self.quotas is not None:
                with self.jobs.connect() as conn:
                    self.quotas.check_storage(conn, owner_id)

        try:
            # Both review reads and command validation expand retained source
            # JSON. Limit that scratch to one parser per owner and attribute it
            # to its analysis even if the server exits before temp cleanup.
            with FileLock(str(_safe_path(locks/lock_name)), timeout=0):
                check_storage()
                with tempfile.TemporaryDirectory(prefix='admission-', dir=root) as directory:
                    with open_rule_replay_source(snapshot.resolve_file('native_rule_replay.json'), directory,
                        artifact_sha256=snapshot.file_sha256('native_rule_replay.json'),
                        source_sha256=snapshot.file_sha256('source.mp4'), analysis_id=snapshot.analysis_id,
                        analysis_revision=snapshot.analysis_revision) as source:
                        check_storage()
                        if len(source.source_frames) != snapshot.frames:
                            raise ValueError('Saved source frames differ from the committed generation')
                        yield source
        except Timeout as error:
            raise HTTPException(429, 'Another review request is reading your source evidence. Retry shortly.') from error

    def _validate_source_command(self, command, snapshot):
        from src.scoring.point_timeline import event_input_record
        with self._source(snapshot) as source:
            initial = command['initial_state_amendment']
            if initial is not None and initial['expected_initial_state_sha256'] != _digest(source.document['initial_state']):
                raise HTTPException(409, 'The reviewed initial-state fingerprint differs. Refresh the source evidence.')
            _initial_state(source, command['initial_state_amendment'], snapshot.analysis_id)
            amendments = {item['event_id']: item for item in command['input_amendments']}
            used = set()
            for row in source.document['inputs']:
                original = _plain(row['source_input'])
                if _digest(original) != row['input_fingerprint']:
                    raise ValueError('Saved native input fingerprint differs')
                amendment = amendments.get(original['event_id'])
                if amendment is None:
                    continue
                if original['event_id'] in used or amendment['expected_input_fingerprint'] != row['input_fingerprint']:
                    raise HTTPException(409, 'The reviewed event fingerprint changed. Refresh the source evidence.')
                replacement = _replacement(original, amendment)
                event, context = source.decode_source_event(replacement)
                if _canonical(event_input_record(event, context)) != _canonical(replacement):
                    raise ValueError('Replacement input differs after current source decoding')
                used.add(original['event_id'])
            if used != set(amendments):
                raise HTTPException(409, 'An amended event is absent from the reviewed generation.')

    def submit(self, analysis_id, owner_id, payload, *, authorization_method):
        self._owned_analysis(analysis_id, owner_id)
        if type(payload) is not dict or 'analysis_id' in payload or 'reviewer_id' in payload:
            raise HTTPException(422, 'Analysis and reviewer identities are supplied by the owner session.')
        if authorization_method not in ('AUTHENTICATED_OWNER', 'EXPLICIT_LOCAL_OWNER'):
            raise ValueError('Unsupported correction authorization method')
        if (authorization_method == 'EXPLICIT_LOCAL_OWNER') != (owner_id == LOCAL_OWNER):
            raise ValueError('Correction authorization method differs from its owner mode')
        try:
            command = _request({**payload, 'analysis_id': analysis_id, 'reviewer_id': owner_id})
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            raise HTTPException(422, 'Invalid bounded semantic correction command.') from error
        encoded, digest = _canonical(command), _digest(command)
        # Idempotency precedes the current-base check: a committed command can
        # be acknowledged after its base has legitimately advanced.
        with self.jobs.connect() as conn:
            old = conn.execute('SELECT * FROM analysis_corrections WHERE analysis_id=? AND owner_id=? AND command_id=?',
                (analysis_id, owner_id, command['correction_id'])).fetchone()
        if old is not None:
            if old['command_json'] != encoded:
                raise HTTPException(409, 'A correction command ID cannot be reused for a different update.')
            _internal(old)
            return public_correction(old)
        if self.quotas is not None:
            with self.jobs.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                self.quotas.check(conn, owner_id, additional_jobs=1, additional_bytes=0)
        snapshot = self._base(command)
        try:
            self._validate_source_command(command, snapshot)
        except HTTPException:
            raise
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            raise HTTPException(422, 'Correction fields contradict the retained source evidence.') from error
        authorization = {'schema_version': 1, 'analysis_id': analysis_id,
            'owner_id': owner_id, 'reviewer_id': owner_id, 'correction_id': command['correction_id'],
            'request_sha256': digest, 'authorized_at': utc_now(), 'authorization_method': authorization_method}
        ident, now = 'correction_'+uuid.uuid4().hex, utc_now()
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._owned_analysis(analysis_id, owner_id, conn=conn)
            old = conn.execute('SELECT * FROM analysis_corrections WHERE analysis_id=? AND owner_id=? AND command_id=?',
                (analysis_id, owner_id, command['correction_id'])).fetchone()
            if old is not None:
                if old['command_json'] != encoded:
                    raise HTTPException(409, 'A correction command ID cannot be reused for a different update.')
                _internal(old)
                return public_correction(old)
            self._base(command)
            if self.quotas is not None:
                self.quotas.check(conn, owner_id, additional_jobs=1, additional_bytes=0)
            conn.execute('''INSERT INTO analysis_corrections
                (id,analysis_id,owner_id,command_id,command_json,request_sha256,authorization_json,
                 base_generation,base_analysis_revision,analysis_revision,source_sha256,
                 base_rule_replay_sha256,status,stage,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (ident, analysis_id, owner_id, command['correction_id'], encoded, digest, _canonical(authorization),
                 command['base_generation'], command['base_analysis_revision'], command['base_analysis_revision']+1,
                 command['source_sha256'], command['base_rule_replay_sha256'], 'QUEUED',
                 'Waiting for correction worker', now))
        return self.get(ident, owner_id)

    def claim(self):
        while True:
            with self.jobs.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute("SELECT * FROM analysis_corrections WHERE status='QUEUED' ORDER BY created_at,id LIMIT 1").fetchone()
                if row is None:
                    return None
                try:
                    internal = _internal(row)
                except (ValueError, TypeError, KeyError):
                    self._finish_row(conn, row['id'], 'FAILED', 'Retained correction authorization is invalid.')
                    continue
                try:
                    self._owned_analysis(row['analysis_id'], row['owner_id'], conn=conn)
                    self._base(internal['command'])
                except (HTTPException, ValueError, OSError):
                    self._finish_row(conn, row['id'], 'CONFLICT', 'The reviewed generation is no longer current.')
                    continue
                if row['attempt'] >= 3:
                    self._finish_row(conn, row['id'], 'FAILED', 'Correction attempt limit reached.')
                    continue
                conn.execute("UPDATE analysis_corrections SET status='PROCESSING',started_at=?,heartbeat_at=?,"
                    "attempt=attempt+1,stage='Validating reviewed source',step=0 WHERE id=?",
                    (utc_now(), utc_now(), row['id']))
            return self.get(row['id'], internal=True)

    def _active(self, conn, correction_id):
        row = conn.execute('SELECT * FROM analysis_corrections WHERE id=?', (correction_id,)).fetchone()
        if row is None:
            raise ValueError('Correction job is absent')
        value = _internal(row)
        if row['cancel_requested'] or row['status'] == 'CANCELLED':
            raise JobCancelled('Correction cancelled by its owner.')
        if row['status'] != 'PROCESSING':
            raise ValueError('Correction is not processing')
        self._owned_analysis(row['analysis_id'], row['owner_id'], conn=conn)
        return value

    def stage(self, correction_id, stage, step=0):
        if type(stage) is not str or not stage.strip() or len(stage) > 300 or type(step) is not int or not 0 <= step <= 11:
            raise ValueError('Invalid correction progress')
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._active(conn, correction_id)
            conn.execute('UPDATE analysis_corrections SET stage=?,step=?,heartbeat_at=? WHERE id=?',
                         (stage, step, utc_now(), correction_id))

    def _finish_row(self, conn, correction_id, status, error=None, generation=None):
        conn.execute('UPDATE analysis_corrections SET status=?,stage=?,step=?,completed_at=?,heartbeat_at=?,error=?,generation=? WHERE id=?',
            (status, 'Correction complete' if status == 'COMPLETED' else 'Correction '+status.lower(),
             11 if status == 'COMPLETED' else 0, utc_now(), utc_now(), error, generation, correction_id))

    def finish(self, correction_id, *, generation=None, error=None, status=None):
        status = status or ('COMPLETED' if generation is not None and error is None else 'FAILED')
        if status not in TERMINAL or (status == 'COMPLETED') != (type(generation) is str and bool(TOKEN.fullmatch(generation))):
            raise ValueError('Invalid terminal correction status or generation')
        if error is not None and (type(error) is not str or len(error) > 4000):
            raise ValueError('Invalid correction error')
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM analysis_corrections WHERE id=?', (correction_id,)).fetchone()
            if row is None:
                raise ValueError('Correction job is absent')
            value = _internal(row)
            if row['status'] in TERMINAL:
                return value
            if status == 'COMPLETED':
                if self._committed_generation(value) != generation:
                    raise ValueError('Correction completion requires its committed owner-bound generation')
            elif row['cancel_requested']:
                status, error = 'CANCELLED', 'Correction cancelled by its owner.'
            self._finish_row(conn, correction_id, status, error, generation)
        return self.get(correction_id, internal=True)

    def cancel(self, correction_id, owner_id):
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT * FROM analysis_corrections WHERE id=? AND owner_id=?',
                               (correction_id, owner_id)).fetchone()
            if row is None:
                raise HTTPException(404, 'Correction not found.')
            if row['status'] in TERMINAL:
                return public_correction(row)
            conn.execute('UPDATE analysis_corrections SET cancel_requested=1 WHERE id=?', (correction_id,))
            if row['status'] == 'QUEUED':
                self._finish_row(conn, correction_id, 'CANCELLED', 'Correction cancelled by its owner.')
        return self.get(correction_id, owner_id)

    def commit_guard(self, correction_id):
        with self.jobs.connect() as conn:
            value = self._active(conn, correction_id)
            self._base(value['command'])
            return value

    def _committed_generation(self, row):
        from src.api.analysis_generation import resolve_generation
        from src.api.result_document import ResultDocument
        snapshot = resolve_generation(self.jobs.output_dir(row['analysis_id']))
        if snapshot is None or snapshot.generation == row['base_generation']:
            return None
        if snapshot.analysis_id != row['analysis_id'] or snapshot.analysis_revision != row['analysis_revision']:
            return None
        if snapshot.file_sha256('source.mp4') != row['source_sha256']:
            return None
        with ResultDocument(self.jobs.output_dir(row['analysis_id']), row['analysis_id'], generation=snapshot.generation) as document:
            evidence = document.value.get('analysisCorrection')
            if (not isinstance(evidence, dict) or evidence.get('analysisId') != row['analysis_id']
                    or type(evidence.get('analysisRevision')) is not int or evidence['analysisRevision'] != row['analysis_revision']
                    or evidence.get('baseGeneration') != row['base_generation']
                    or evidence.get('sourceSha256') != row['source_sha256']
                    or evidence.get('requestSha256') != row['request_sha256']
                    or not isinstance(evidence.get('review'), dict)
                    or evidence['review'].get('correctionId') != row['command']['correction_id']
                    or evidence.get('authorizationStatus') != ('EXPLICIT_LOCAL_OWNER' if
                        row['authorization']['authorization_method'] == 'EXPLICIT_LOCAL_OWNER' else 'OWNER_AUTHORIZED')
                    or _canonical(evidence.get('authorizationReceipt')) != _canonical(row['authorization'])):
                return None
        return snapshot.generation

    def committed_generation(self, correction):
        """Validate an acknowledgement against the actual sealed current result."""
        if isinstance(correction, str):
            correction = self.get(correction, internal=True)
        # Read the immutable retained command again; a mutable worker snapshot
        # cannot authorize a different owner, command or generation.
        retained = self.get(correction['id'], internal=True)
        if (retained['command'] != correction['command'] or
                retained['authorization'] != correction['authorization']):
            raise ValueError('Correction acknowledgement snapshot differs')
        return self._committed_generation(retained)

    @contextmanager
    def publication_guard(self, correction_id):
        """Serialize cancellation through pointer commit and acknowledgement.

        Run publication without any other store writes in this context. On a
        lost exception acknowledgement, a fully bound actual pointer wins; a
        BaseException is never suppressed, though its committed state is saved.
        """
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self._active(conn, correction_id)
            self._base(row['command'])
            failure = None
            try:
                yield row
            except BaseException as error:
                failure = error
            # The yielded worker snapshot is mutable presentation data. Read
            # the immutable retained authorization again on the held writer
            # transaction before acknowledging the real committed pointer.
            retained = self._active(conn, correction_id)
            generation = self._committed_generation(retained)
            if generation is not None:
                self._finish_row(conn, correction_id, 'COMPLETED', generation=generation)
                # Commit before rethrowing interrupts: jobs.connect would roll
                # back on the deliberate exception after pointer acknowledgement.
                conn.commit()
                if failure is not None and not isinstance(failure, Exception):
                    raise failure
            elif failure is not None:
                raise failure
            else:
                raise ValueError('Correction publication returned without a bound committed generation')

    def recover_interrupted(self, validate_committed=None):
        """Called only by the worker while it owns the process lock."""
        previous = ''
        while True:
            with self.jobs.connect() as conn:
                row = conn.execute("SELECT * FROM analysis_corrections WHERE status='PROCESSING' AND id>? ORDER BY id LIMIT 1", (previous,)).fetchone()
            if row is None:
                return
            previous = row['id']
            try:
                value = _internal(row)
            except (ValueError, TypeError, KeyError):
                with self.jobs.connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    self._finish_row(conn, row['id'], 'FAILED', 'Retained correction authorization is invalid.')
                continue
            try:
                generation = validate_committed(value) if validate_committed is not None else self._committed_generation(value)
            except (ValueError, OSError, HTTPException):
                generation = None
            with self.jobs.connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                current = conn.execute('SELECT * FROM analysis_corrections WHERE id=?', (row['id'],)).fetchone()
                if current is None or current['status'] != 'PROCESSING':
                    continue
                if generation is not None and generation == self._committed_generation(value):
                    self._finish_row(conn, row['id'], 'COMPLETED', generation=generation)
                elif current['cancel_requested']:
                    self._finish_row(conn, row['id'], 'CANCELLED', 'Correction cancelled by its owner.')
                elif current['attempt'] >= 3:
                    self._finish_row(conn, row['id'], 'FAILED', 'Correction attempt limit reached.')
                else:
                    try:
                        self._owned_analysis(row['analysis_id'], row['owner_id'], conn=conn)
                        self._base(value['command'])
                    except (HTTPException, ValueError, OSError):
                        self._finish_row(conn, row['id'], 'CONFLICT', 'The reviewed generation is no longer current.')
                    else:
                        conn.execute("UPDATE analysis_corrections SET status='QUEUED',stage='Resuming correction from sealed source',"
                            'step=0,completed_at=NULL,error=NULL,heartbeat_at=NULL WHERE id=?', (row['id'],))

    def review_source(self, analysis_id, owner_id, *, generation=None, offset=0, limit=20):
        from src.api.analysis_generation import resolve_generation
        self._owned_analysis(analysis_id, owner_id)
        if (type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50
                or generation is not None and (type(generation) is not str or TOKEN.fullmatch(generation) is None)):
            raise HTTPException(422, 'Invalid bounded review-source selector.')
        snapshot = resolve_generation(self.jobs.output_dir(analysis_id), generation)
        if snapshot is None or snapshot.analysis_id != analysis_id:
            raise HTTPException(409, 'A complete saved native generation is required for correction review.')
        with self._source(snapshot) as source:
            initial = _plain(source.document['initial_state'])
            rows, total = [], len(source.document['inputs'])
            length = len(_canonical(initial).encode('utf-8'))
            for index in range(offset, min(total, offset+limit)):
                row = source.document['inputs'][index]
                entry = {'source_input': _plain(row['source_input']), 'input_fingerprint': row['input_fingerprint']}
                if _digest(entry['source_input']) != entry['input_fingerprint']:
                    raise ValueError('Saved native input fingerprint differs')
                source.decode_source_event(entry['source_input'])
                length += len(_canonical(entry).encode('utf-8'))
                if length > MAX_REQUEST_BYTES:
                    if not rows:
                        raise HTTPException(413, 'This source input exceeds the bounded review page size.')
                    break
                rows.append(entry)
            current = resolve_generation(self.jobs.output_dir(analysis_id))
            result = {'schema_version': 1, 'analysis_id': analysis_id,
                'source_sha256': snapshot.file_sha256('source.mp4'), 'base_generation': snapshot.generation,
                'base_analysis_revision': snapshot.analysis_revision,
                'base_rule_replay_sha256': snapshot.file_sha256('native_rule_replay.json'),
                'initial_state': initial, 'initial_state_sha256': _digest(initial),
                'initial_score': source.initial_state.to_dict(),
                'inputs': rows, 'offset': offset, 'limit': limit, 'total_inputs': total,
                'next_offset': offset+len(rows) if offset+len(rows) < total else None,
                'is_current': current is not None and current.generation == snapshot.generation,
                'editable_fields': ['initial_score', 'event_semantics', 'reviewed_participant_binding'],
                'measurement_edits_supported': False, 'independent_accuracy_validated': False}
        return result


def correction_router(jobs, settings, ensure_worker, *, quotas=None):
    router = APIRouter(prefix='/api/analyses')
    if quotas is None:
        from src.api.resource_quotas import ResourceQuotas
        quotas = ResourceQuotas(jobs, settings)
    store = CorrectionStore(jobs, quotas=quotas)

    @router.post('/{analysis_id}/corrections', status_code=202)
    async def submit(analysis_id: str, request: Request, owner_id: str = Depends(current_owner)):
        await run_in_threadpool(store._owned_analysis, analysis_id, owner_id)
        size, chunks = 0, []
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise HTTPException(413, 'Correction command exceeds one MiB.')
            chunks.append(chunk)
        try:
            def constant(_):
                raise ValueError('Finite correction JSON is required')
            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError('Duplicate correction JSON key')
                    result[key] = value
                return result
            payload = json.loads(b''.join(chunks), parse_constant=constant, object_pairs_hook=unique)
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            raise HTTPException(422, 'Correction command must be finite JSON.') from error
        method = 'AUTHENTICATED_OWNER' if settings.require_auth else 'EXPLICIT_LOCAL_OWNER'
        if settings.require_auth and (not getattr(request.state, 'user', None)
                                     or request.state.user['id'] != owner_id):
            raise HTTPException(401, 'Sign in to authorize the correction.')
        value = await run_in_threadpool(store.submit, analysis_id, owner_id, payload, authorization_method=method)
        ensure_worker()
        return value

    @router.get('/{analysis_id}/corrections')
    def listing(analysis_id: str, owner_id: str = Depends(current_owner)):
        return store.list(analysis_id, owner_id)

    @router.get('/{analysis_id}/corrections/{correction_id}')
    def status(analysis_id: str, correction_id: str, owner_id: str = Depends(current_owner)):
        store._owned_analysis(analysis_id, owner_id)
        value = store.get(correction_id, owner_id)
        if value['analysis_id'] != analysis_id:
            raise HTTPException(404, 'Correction not found.')
        return value

    @router.post('/{analysis_id}/corrections/{correction_id}/cancel')
    def cancel(analysis_id: str, correction_id: str, owner_id: str = Depends(current_owner)):
        store._owned_analysis(analysis_id, owner_id)
        value = store.get(correction_id, owner_id)
        if value['analysis_id'] != analysis_id:
            raise HTTPException(404, 'Correction not found.')
        return store.cancel(correction_id, owner_id)

    @router.get('/{analysis_id}/review-source')
    def review(analysis_id: str, generation: str | None = None,
               offset: int = Query(default=0, ge=0), limit: int = Query(default=20, ge=1, le=50),
               owner_id: str = Depends(current_owner)):
        return store.review_source(analysis_id, owner_id, generation=generation, offset=offset, limit=limit)

    return router
