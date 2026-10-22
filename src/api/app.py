"""Upload, job status and result API for the existing tennis pipeline."""

from contextlib import asynccontextmanager
from functools import partial
import asyncio
import json
import logging
import os
import sqlite3
from pathlib import Path
import subprocess
import sys
import shutil
import re

from filelock import FileLock, Timeout
from fastapi import FastAPI, HTTPException, Request, Depends
from starlette.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from src.api.coach import app as coach_app
from src.api.jobs import JobStore, REPO_ROOT, Settings, public_job
from src.api.product import product_router
from src.api.auth import AuthStore, current_owner
from src.api.uploads import upload_router
from src.api.corrections import correction_router
from src.api.resource_quotas import ResourceQuotas
from src.api.retry_intents import RetryIntents
from src.api.result_document import ResultDocument, ResultStreamingResponse, result_summary

EXTENSIONS = {'.mp4', '.mov', '.avi'}
MEDIA = {'source': 'source.mp4', 'annotated': 'annotated_web.mp4'}
LOGGER = logging.getLogger(__name__)


def create_app(settings=None, start_worker=True, demos_manifest=None, frontend_root=None):
    settings = settings or Settings.from_env()
    store = JobStore(settings.root)
    quotas = ResourceQuotas(store, settings)
    retries = RetryIntents(store, quotas)
    worker = None

    def ensure_worker():
        nonlocal worker
        if not start_worker or (worker is not None and worker.poll() is None):
            return
        # A worker can outlive the web server. Check the OS-backed lock before
        # launching a replacement, rather than spawning a losing process on
        # every supervision tick after a normal server restart.
        try:
            with FileLock(str(settings.root / 'worker.lock'), timeout=0):
                pass
        except Timeout:
            return
        env = {**os.environ, 'TENNIS_STORAGE_ROOT': str(settings.root),
               'TENNIS_MAX_DURATION_SECONDS': str(settings.max_duration_seconds),
               'TENNIS_MAX_OWNER_PENDING_JOBS': str(settings.max_owner_pending_jobs),
               'TENNIS_MAX_OWNER_STORAGE_BYTES': str(settings.max_owner_storage_bytes)}
        with (settings.root / 'worker.log').open('ab') as log:
            worker = subprocess.Popen([sys.executable, '-u', '-m', 'src.api.worker'], cwd=REPO_ROOT,
                                      env=env, stdout=log, stderr=log,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    @asynccontextmanager
    async def lifespan(_app):
        ensure_worker()
        async def supervise():
            while True:
                await asyncio.sleep(3)
                ensure_worker()
        watchdog = asyncio.create_task(supervise())
        try:
            yield
        finally:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
        # The separately locked worker keeps processing accepted jobs when the
        # web server restarts. Never infer that a job stopped from an HTTP timeout.

    app = FastAPI(title='Astra Tennis Analysis', lifespan=lifespan, docs_url='/api/docs', redoc_url=None)
    app.state.store = store
    app.state.quotas = quotas
    def curated_entries():
        from src.api.demos import curated_demos
        return curated_demos(store, demos_manifest)['demos']

    def catalog_generation(demo):
        selected = re.search(r'\?generation=([0-9a-f]{64})$', demo['previewUrl'])
        return selected[1] if selected is not None else None

    def curated_ids(generation=None):
        return {demo['analysisId'] for demo in curated_entries()
                if generation is None or catalog_generation(demo) == generation}

    def permitted_generation(job, owner_id, generation):
        if job['owner_id'] == owner_id:
            return generation
        demo = next((value for value in curated_entries() if value['analysisId'] == job['id']), None)
        if demo is None or generation is not None and generation != catalog_generation(demo):
            raise HTTPException(404, 'Analysis not found.')
        # Public bare URLs use the explicitly curated revision, even when a
        # newer private generation exists for the owner.
        return catalog_generation(demo)

    def public_read(request):
        if request.method not in {'GET', 'HEAD'}:
            return False
        if request.url.path == '/api/demos':
            return True
        match = re.fullmatch(r'/api/analyses/(match_[0-9a-f]{32})(?:/result|/media/(?:source|annotated)|/download/(?:original|annotated|report|readable-report|csv|source-timing))?', request.url.path)
        selectors = request.query_params.getlist('generation')
        return bool(match and len(selectors) <= 1 and
                    match[1] in curated_ids(selectors[0] if selectors else None))

    AuthStore(store, settings, public_read).install(app)
    app.include_router(coach_app.router)
    app.include_router(product_router(store))
    app.include_router(upload_router(store, settings, ensure_worker, quotas=quotas))
    app.include_router(correction_router(store, settings, ensure_worker, quotas=quotas))

    def job_or_404(analysis_id, owner_id, public=False, generation=None):
        job = store.get(analysis_id)
        if job is None or (job['owner_id'] != owner_id and not (public and analysis_id in curated_ids(generation))):
            raise HTTPException(404, 'Analysis not found.')
        return job

    @app.get('/api/health')
    def health():
        try:
            with FileLock(str(settings.root / 'worker.lock'), timeout=0):
                worker_available = False
        except Timeout:
            worker_available = True
        return {'status': 'ok', 'max_upload_bytes': settings.max_upload_bytes,
                'supported_formats': sorted(EXTENSIONS), 'max_duration_seconds': settings.max_duration_seconds,
                'worker_available': worker_available, 'auth_required': settings.require_auth,
                'resumable_uploads': True, 'upload_chunk_bytes': 8 * 1024 * 1024,
                'batch_max_files': 16, 'max_owner_pending_jobs': settings.max_owner_pending_jobs,
                'max_owner_storage_bytes': settings.max_owner_storage_bytes}

    @app.get('/api/usage')
    def usage(owner_id: str = Depends(current_owner)):
        with store.connect() as conn:
            return quotas.snapshot(conn, owner_id)

    @app.post('/api/usage/reconcile')
    def reconcile_usage(owner_id: str = Depends(current_owner)):
        return quotas.reconcile_owner(owner_id)

    def reserve(analysis_id, owner_id, size, kind):
        with store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            quotas.reserve(conn, analysis_id, owner_id, size, kind=kind, recovery_version=1)

    def release(analysis_id):
        with store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            quotas.release(conn, analysis_id)

    def check_storage(owner_id):
        with store.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            quotas.check_storage(conn, owner_id)

    async def upload_operation(function, *args, **kwargs):
        """Finish in-flight I/O before a cancelled request drops its lease.

        Native asyncio cancellation can leave an executor thread running even
        after run_in_threadpool unwinds. Shield and drain each operation so
        settlement cannot race a still-running queue insertion or file write.
        Request body receives themselves remain cancellable.
        """
        # An executor Future is not a separately cancellable asyncio Task.
        # Even event-loop shutdown cancelling all tasks must drain this actual
        # operation before request settlement or releasing its lifetime lock.
        task = asyncio.get_running_loop().run_in_executor(None, partial(function, *args, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()  # Retrieve failures; cancellation remains authoritative.
            raise

    def settle_reservation(analysis_id, extension, *, accepted=False):
        """Preserve durable acceptance and the request's original failure."""
        try:
            durable = accepted or store.get(analysis_id) is not None
        except sqlite3.Error:
            # Unknown queue state cannot authorize deletion or release. The
            # live reservation keeps these exact paths and capacity attributed.
            LOGGER.exception('Upload acceptance could not be reconciled: %s', analysis_id)
            return
        if not durable:
            directory = store.upload_dir(analysis_id)
            try:
                for name in ('upload.part', 'original' + extension):
                    (directory / name).unlink(missing_ok=True)
                if directory.exists():
                    directory.rmdir()
            except OSError:
                # Retain the ownership row even after release, so residue is
                # still charged and cannot become another owner's data.
                LOGGER.exception('Upload cleanup left retained bytes: %s', analysis_id)
        try:
            release(analysis_id)
        except (OSError, sqlite3.Error, HTTPException):
            # Existing analyses deduplicate a live reservation after acceptance.
            # An unaccepted reservation remains conservatively charged until
            # recovery, without replacing a 413/415 or losing an accepted job.
            LOGGER.exception('Upload reservation could not be released: %s', analysis_id)

    @app.get('/api/analyses')
    def list_analyses(owner_id: str = Depends(current_owner)):
        return [public_job(job) for job in store.list(owner_id)]

    @app.get('/api/demos')
    def demos():
        from src.api.demos import curated_demos
        return curated_demos(store, demos_manifest)

    @app.get('/api/runs')
    def list_runs(owner_id: str = Depends(current_owner)):
        summaries = []
        for job in store.list(owner_id):
            if job['status'] == 'COMPLETED':
                try:
                    summaries.append(result_summary(store.output_dir(job['id']), job['id']))
                except (OSError, ValueError, sqlite3.Error, TypeError, KeyError):
                    continue
        return summaries

    @app.post('/api/analyses', status_code=202)
    async def upload(request: Request, filename: str, match_id: str | None = None, owner_id: str = Depends(current_owner)):
        if match_id:
            with store.connect() as conn:
                draft = conn.execute("SELECT analysis_id FROM matches WHERE id=? AND owner_id=? AND source='upload'", (match_id, owner_id)).fetchone()
                if not draft:
                    raise HTTPException(404, 'Upload draft not found.')
                if draft['analysis_id']:
                    raise HTTPException(409, 'This upload draft is already used. Create a new draft before retrying.')
        # Never use a supplied filename as a filesystem path.
        filename = filename.replace('\\', '/').rsplit('/', 1)[-1].strip()
        extension = Path(filename).suffix.lower()
        if extension not in EXTENSIONS or not filename or len(filename) > 200 or any(ord(c) < 32 for c in filename):
            raise HTTPException(415, 'Choose an MP4, MOV or AVI video with a valid filename.')
        declared = request.headers.get('content-length')
        if declared is not None and (not declared.isdigit() or int(declared) > settings.max_upload_bytes):
            raise HTTPException(413, 'Video exceeds the upload size limit.')
        analysis_id = store.new_id()
        # Hold across streaming, durable queue insertion and final settlement.
        # Recovery takes this same OS lock nonblocking before releasing capacity.
        try:
            request_lock = quotas.reservation_lock(analysis_id)
            request_lock.acquire(timeout=0)
        except (OSError, ValueError, Timeout):
            raise HTTPException(503, 'Storage usage could not be checked. Please try again.') from None
        try:
            directory = store.upload_dir(analysis_id)
            temporary = directory / 'upload.part'
            size, header = 0, b''
            accepted = False
            reserved = False

            def reserve_request():
                nonlocal reserved
                reserve(analysis_id, owner_id,
                        int(declared) if declared is not None else settings.max_upload_bytes, 'legacy')
                reserved = True

            try:
                await upload_operation(reserve_request)
                directory.mkdir(parents=True)
                handle = None

                def open_file():
                    nonlocal handle
                    handle = temporary.open('xb')

                try:
                    await upload_operation(open_file)
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > settings.max_upload_bytes or declared is not None and size > int(declared):
                            raise HTTPException(413, 'Video exceeds the upload size limit.')
                        if len(header) < 32:
                            header = (header + chunk)[:32]
                        await upload_operation(handle.write, chunk)
                finally:
                    if handle is not None:
                        await upload_operation(handle.close)
                signature_ok = (header[:4] == b'RIFF' and header[8:12] == b'AVI ') if extension == '.avi' else (
                    header[4:8] in {b'ftyp', b'moov', b'mdat', b'wide', b'free', b'skip'})
                if size < 12 or not signature_ok:
                    raise HTTPException(415, 'The file is not a supported video container. Export an MP4, MOV or AVI and try again.')
                temporary.replace(directory / ('original' + extension))
                await upload_operation(check_storage, owner_id)
                job = await upload_operation(store.create, analysis_id, filename, extension,
                                             size, settings.config, match_id, owner_id)
                accepted = True
                ensure_worker()
                return public_job(job)
            except OSError as exc:
                raise HTTPException(507, 'Video could not be stored. Check available disk space and try again.') from exc
            finally:
                if reserved:
                    await upload_operation(settle_reservation, analysis_id, extension, accepted=accepted)
        finally:
            request_lock.release()

    @app.get('/api/analyses/{analysis_id}')
    def status(analysis_id: str, owner_id: str = Depends(current_owner)):
        return public_job(job_or_404(analysis_id, owner_id, public=True))

    @app.post('/api/analyses/{analysis_id}/cancel')
    def cancel(analysis_id: str, owner_id: str = Depends(current_owner)):
        return public_job(store.cancel(analysis_id, owner_id))

    @app.post('/api/analyses/{analysis_id}/retry', status_code=202)
    def retry(analysis_id: str, request: Request, owner_id: str = Depends(current_owner)):
        prior = job_or_404(analysis_id, owner_id)
        keys = request.headers.getlist('idempotency-key')
        key = keys[0] if len(keys) == 1 else None
        with retries.lock(owner_id, key):
            selected = retries.selected(owner_id, key, analysis_id)
            existing = retries.accepted(selected, owner_id)
            if existing is not None:
                ensure_worker()
                return public_job(existing)
            if prior['status'] != 'FAILED':
                raise HTTPException(409, 'Only a failed or cancelled analysis can be retried.')
            original = store.upload_dir(analysis_id) / ('original' + prior['extension'])
            if not original.is_file():
                raise HTTPException(409, 'The original upload is unavailable. Upload this video again.')
            size = original.stat().st_size
            if size != prior['size_bytes']:
                raise HTTPException(409, 'The retained original changed. Upload this video again.')
            new_id = retries.admit(owner_id, key, analysis_id, size)
            directory = store.upload_dir(new_id)
            destination = directory / original.name
            temporary = directory / 'upload.part'
            try:
                directory.mkdir(parents=True, exist_ok=True)
                # This identity has no accepted job under its request lock.
                # Replace only its own pre-queue crash leftovers, then rename
                # the complete input before inserting the durable queue row.
                temporary.unlink(missing_ok=True)
                destination.unlink(missing_ok=True)
                try:
                    os.link(original, temporary)
                except OSError:
                    shutil.copyfile(original, temporary)
                if temporary.stat().st_size != size:
                    raise HTTPException(409, 'The retained original changed while retrying. Upload the video again.')
                temporary.replace(destination)
                check_storage(owner_id)
                job = store.create(new_id, prior['filename'], prior['extension'], size, prior['config_path'],
                                   owner_id=owner_id, retry_of=analysis_id)
            except OSError as error:
                raise HTTPException(507, 'The retry could not be stored. Check available disk space and try again.') from error
            finally:
                settle_reservation(new_id, prior['extension'])
        ensure_worker()
        return public_job(job)

    def generation_selector(generation):
        if generation is not None and re.fullmatch('[0-9a-f]{64}', generation) is None:
            raise HTTPException(422, 'Invalid saved generation.')

    def saved_document(analysis_id, owner_id, generation=None):
        job = job_or_404(analysis_id, owner_id, public=True, generation=generation)
        if job['status'] != 'COMPLETED':
            raise HTTPException(409, job['error'] or 'Analysis is still processing.')
        generation_selector(generation)
        generation = permitted_generation(job, owner_id, generation)
        document = None
        try:
            directory = store.output_dir(analysis_id)
            document = ResultDocument(directory, analysis_id, generation=generation)
            return document, document.presented(analysis_id)
        except FileNotFoundError as exc:
            if document is not None:
                document._close_preserving(exc)
            raise HTTPException(404 if generation is not None else 503,
                'The saved generation is unavailable.' if generation is not None else
                'The saved analysis is unavailable. Upload the video again to retry.') from exc
        except (OSError, ValueError, sqlite3.Error, AttributeError, KeyError, TypeError) as exc:
            if document is not None:
                document._close_preserving(exc)
            raise HTTPException(503, 'The saved analysis is unavailable. Upload the video again to retry.') from exc

    @app.get('/api/analyses/{analysis_id}/result')
    def result(analysis_id: str, generation: str | None = None, owner_id: str = Depends(current_owner)):
        document, saved = saved_document(analysis_id, owner_id, generation)
        return ResultStreamingResponse(document, document.json_chunks(saved), media_type='application/json')

    def saved_asset(analysis_id, name, generation, *, legacy_path=None, legacy_revision=None):
        from src.api.analysis_generation import resolve_generation
        generation_selector(generation)
        try:
            directory = store.output_dir(analysis_id)
            snapshot = resolve_generation(directory, generation)
            if snapshot is not None:
                if snapshot.analysis_id != analysis_id:
                    raise ValueError('Saved asset belongs to a different analysis')
                if legacy_revision is not None:
                    raise HTTPException(409, 'Refresh the saved analysis to use its generation links.')
                return snapshot.resolve_file(name), snapshot
            if generation is not None:
                raise HTTPException(404, 'The saved generation is unavailable.')
            return (legacy_path if legacy_path is not None else directory / name), None
        except FileNotFoundError as exc:
            raise HTTPException(404 if generation is not None else 503, 'The saved generation is unavailable.') from exc
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise HTTPException(503, 'The saved analysis is unavailable. Upload the video again to retry.') from exc

    @app.get('/api/analyses/{analysis_id}/media/{kind}')
    def media(analysis_id: str, kind: str, generation: str | None = None, v: str | None = None,
              owner_id: str = Depends(current_owner)):
        job = job_or_404(analysis_id, owner_id, public=True, generation=generation)
        if kind not in MEDIA or job['status'] != 'COMPLETED':
            raise HTTPException(404, 'Video is not available.')
        generation = permitted_generation(job, owner_id, generation)
        path, snapshot = saved_asset(analysis_id, MEDIA[kind], generation, legacy_revision=v)
        if path is None or not path.is_file():
            raise HTTPException(404, 'Video is not available.')
        headers = {'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'}
        if snapshot is not None:
            headers['X-Analysis-Generation'] = snapshot.generation
        return FileResponse(path, media_type='video/mp4', headers=headers)

    @app.get('/api/analyses/{analysis_id}/download/{kind}')
    def download(analysis_id: str, kind: str, generation: str | None = None, v: str | None = None,
                 owner_id: str = Depends(current_owner)):
        job = job_or_404(analysis_id, owner_id, public=True, generation=generation)
        generation_selector(generation)
        generation = permitted_generation(job, owner_id, generation)
        if kind == 'report' and job['status'] == 'COMPLETED':
            document, saved = saved_document(analysis_id, owner_id, generation)
            return ResultStreamingResponse(document, document.json_chunks(saved), media_type='application/json', headers={
                'Content-Disposition': f'attachment; filename="{analysis_id}_report.json"',
                'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'})
        if kind in {'readable-report', 'csv'} and job['status'] == 'COMPLETED':
            from src.api.reporting import readable_report_chunks, telemetry_csv_chunks
            document, saved = saved_document(analysis_id, owner_id, generation)
            extension = 'html' if kind == 'readable-report' else 'csv'
            headers = {
                'Content-Disposition': f'attachment; filename="{analysis_id}_{kind}.{extension}"',
                'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': "default-src 'none'; img-src data:; style-src 'unsafe-inline'"}
            if kind == 'csv':
                return ResultStreamingResponse(document, telemetry_csv_chunks(saved), media_type='text/csv', headers=headers)
            try:
                annotation = document.asset_path('annotated_web.mp4')
                image_directory = document.asset_directory if annotation is not None else None
            except (OSError, ValueError, sqlite3.Error, TypeError, KeyError) as exc:
                document._close_preserving(exc)
                raise HTTPException(503, 'The saved analysis is unavailable. Upload the video again to retry.') from exc
            return ResultStreamingResponse(document, readable_report_chunks(saved, image_directory), media_type='text/html', headers=headers)
        if kind == 'original':
            original_name = 'original' + job['extension']
            original = store.upload_dir(analysis_id) / original_name
            if job['status'] != 'COMPLETED':
                if generation is not None:
                    raise HTTPException(404, 'The saved generation is unavailable.')
                path, snapshot = original, None
            else:
                path, snapshot = saved_asset(analysis_id, original_name, generation,
                    legacy_path=original, legacy_revision=v)
        elif kind == 'annotated' and job['status'] == 'COMPLETED':
            path, snapshot = saved_asset(analysis_id, 'annotated_web.mp4', generation, legacy_revision=v)
        elif kind == 'source-timing' and job['status'] == 'COMPLETED':
            path, snapshot = saved_asset(analysis_id, 'source_frame_mapping.jsonl', generation, legacy_revision=v)
        else:
            raise HTTPException(404, 'Download is not available.')
        if path is None or not path.is_file():
            raise HTTPException(404, 'Download is not available.')
        headers = {'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'}
        if snapshot is not None:
            headers['X-Analysis-Generation'] = snapshot.generation
        return FileResponse(path, filename=f'{analysis_id}_{kind}{path.suffix}',
                            media_type='application/x-ndjson' if kind == 'source-timing' else None,
                            headers=headers)

    # One-origin production serving, while Vite remains available for development.
    frontend = Path(frontend_root) if frontend_root is not None else REPO_ROOT / 'frontend/dist'

    @app.get('/{page_path:path}', include_in_schema=False)
    def website(page_path: str):
        if page_path == 'api' or page_path.startswith('api/'):
            raise HTTPException(404, 'API endpoint not found.')
        path = (frontend / page_path).resolve()
        if not path.is_relative_to(frontend.resolve()):
            raise HTTPException(404, 'Page not found.')
        if path.is_file():
            return FileResponse(path)
        if not (frontend / 'index.html').is_file():
            raise HTTPException(503, 'Build the frontend first, or run the Vite development server on port 4173.')
        if Path(page_path).suffix:
            raise HTTPException(404, 'Asset not found.')
        return FileResponse(frontend / 'index.html', headers={'Cache-Control': 'no-cache'})

    return app


app = create_app()
