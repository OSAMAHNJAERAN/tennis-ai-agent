"""Upload, job status and result API for the existing tennis pipeline."""

from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import aiofiles
from filelock import FileLock, Timeout
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import FileResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from src.api.coach import app as coach_app
from src.api.jobs import JobStore, REPO_ROOT, Settings, public_job
from src.api.product import product_router
from src.api.media_revision import presentation_urls

EXTENSIONS = {'.mp4', '.mov', '.avi'}
MEDIA = {'source': 'source.mp4', 'annotated': 'annotated_web.mp4'}


def create_app(settings=None, start_worker=True, demos_manifest=None):
    settings = settings or Settings.from_env()
    store = JobStore(settings.root)
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
               'TENNIS_MAX_DURATION_SECONDS': str(settings.max_duration_seconds)}
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
    app.include_router(coach_app.router)
    app.include_router(product_router(store))

    def job_or_404(analysis_id):
        job = store.get(analysis_id)
        if job is None:
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
                'worker_available': worker_available}

    @app.get('/api/analyses')
    def list_analyses():
        return [public_job(job) for job in store.list()]

    @app.get('/api/demos')
    def demos():
        from src.api.demos import curated_demos
        return curated_demos(store, demos_manifest)

    @app.get('/api/runs')
    def list_runs():
        summaries = []
        for job in store.list():
            if job['status'] == 'COMPLETED':
                try:
                    summaries.append(json.loads((store.output_dir(job['id']) / 'summary.json').read_text(encoding='utf-8')))
                except (OSError, ValueError):
                    continue
        return summaries

    @app.post('/api/analyses', status_code=202)
    async def upload(request: Request, filename: str, match_id: str | None = None):
        # Never use a supplied filename as a filesystem path.
        filename = filename.replace('\\', '/').rsplit('/', 1)[-1].strip()
        extension = Path(filename).suffix.lower()
        if extension not in EXTENSIONS or not filename or len(filename) > 200 or any(ord(c) < 32 for c in filename):
            raise HTTPException(415, 'Choose an MP4, MOV or AVI video with a valid filename.')
        declared = request.headers.get('content-length')
        if declared and (not declared.isdigit() or int(declared) > settings.max_upload_bytes):
            raise HTTPException(413, 'Video exceeds the upload size limit.')
        analysis_id = store.new_id()
        directory = store.upload_dir(analysis_id)
        try:
            directory.mkdir(parents=True)
        except OSError as exc:
            raise HTTPException(507, 'Video could not be stored. Check available disk space and try again.') from exc
        temporary = directory / 'upload.part'
        size, header = 0, b''
        accepted = False
        try:
            async with aiofiles.open(temporary, 'xb') as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(413, 'Video exceeds the upload size limit.')
                    if len(header) < 32:
                        header = (header + chunk)[:32]
                    await handle.write(chunk)
            signature_ok = (header[:4] == b'RIFF' and header[8:12] == b'AVI ') if extension == '.avi' else (
                header[4:8] in {b'ftyp', b'moov', b'mdat', b'wide', b'free', b'skip'})
            if size < 12 or not signature_ok:
                raise HTTPException(415, 'The file is not a supported video container. Export an MP4, MOV or AVI and try again.')
            temporary.replace(directory / ('original' + extension))
            job = await run_in_threadpool(store.create, analysis_id, filename, extension, size, settings.config, match_id)
            accepted = True
            ensure_worker()
            return public_job(job)
        except OSError as exc:
            raise HTTPException(507, 'Video could not be stored. Check available disk space and try again.') from exc
        finally:
            if not accepted:
                # Only paths created for this UUID in this request are removed.
                for name in ('upload.part', 'original' + extension):
                    (directory / name).unlink(missing_ok=True)
                directory.rmdir()

    @app.get('/api/analyses/{analysis_id}')
    def status(analysis_id: str):
        return public_job(job_or_404(analysis_id))

    @app.get('/api/analyses/{analysis_id}/result')
    def result(analysis_id: str):
        job = job_or_404(analysis_id)
        if job['status'] != 'COMPLETED':
            raise HTTPException(409, job['error'] or 'Analysis is still processing.')
        try:
            directory = store.output_dir(analysis_id)
            saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
            return presentation_urls(saved, directory, analysis_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, 'The saved analysis is unavailable. Upload the video again to retry.') from exc

    @app.get('/api/analyses/{analysis_id}/media/{kind}')
    def media(analysis_id: str, kind: str):
        job = job_or_404(analysis_id)
        if kind not in MEDIA or job['status'] != 'COMPLETED':
            raise HTTPException(404, 'Video is not available.')
        path = store.output_dir(analysis_id) / MEDIA[kind]
        if not path.is_file():
            raise HTTPException(404, 'Video is not available.')
        return FileResponse(path, media_type='video/mp4', headers={
            'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'})

    @app.get('/api/analyses/{analysis_id}/download/{kind}')
    def download(analysis_id: str, kind: str):
        job = job_or_404(analysis_id)
        if kind == 'report' and job['status'] == 'COMPLETED':
            return Response(json.dumps(result(analysis_id), allow_nan=False), media_type='application/json', headers={
                'Content-Disposition': f'attachment; filename="{analysis_id}_report.json"',
                'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'})
        if kind in {'readable-report', 'csv'} and job['status'] == 'COMPLETED':
            from src.api.reporting import readable_report, telemetry_csv_chunks
            saved = result(analysis_id)
            extension = 'html' if kind == 'readable-report' else 'csv'
            headers = {
                'Content-Disposition': f'attachment; filename="{analysis_id}_{kind}.{extension}"',
                'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': "default-src 'none'; img-src data:; style-src 'unsafe-inline'"}
            if kind == 'csv':
                return StreamingResponse(telemetry_csv_chunks(saved), media_type='text/csv', headers=headers)
            return Response(readable_report(saved, store.output_dir(analysis_id)), media_type='text/html', headers=headers)
        if kind == 'original':
            path = store.upload_dir(analysis_id) / ('original' + job['extension'])
        elif kind == 'annotated' and job['status'] == 'COMPLETED':
            path = store.output_dir(analysis_id) / 'annotated_web.mp4'
        elif kind == 'source-timing' and job['status'] == 'COMPLETED':
            path = store.output_dir(analysis_id) / 'source_frame_mapping.jsonl'
        else:
            raise HTTPException(404, 'Download is not available.')
        if not path.is_file():
            raise HTTPException(404, 'Download is not available.')
        return FileResponse(path, filename=f'{analysis_id}_{kind}{path.suffix}',
                            media_type='application/x-ndjson' if kind == 'source-timing' else None,
                            headers={'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-cache'})

    # One-origin production serving, while Vite remains available for development.
    frontend = REPO_ROOT / 'frontend/dist'

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
