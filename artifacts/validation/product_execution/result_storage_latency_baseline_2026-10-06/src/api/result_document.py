"""Coherent saved result generations and owned bounded response readers."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import anyio
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from src.api.result_storage import ResultStorage
from src.api.results import file_hash
from src.api.media_revision import presentation_urls

GENERATION_NAME = re.compile(r'^[0-9a-f]{32}\.sqlite3$')


def publish_result(storage, result, directory):
    """Close strict exports and database before committing the generation.

    Job completion is the caller's final action. A generation manifest is the
    authoritative reader gate; partial generations remain diagnostic artifacts.
    Compatibility JSON/summary files remain available to existing tooling.
    """
    directory = Path(directory)
    temporary = directory / 'result.tmp'
    with temporary.open('w', encoding='utf-8') as stream:
        storage.dump(result, stream)
    summary_temporary = directory / 'summary.tmp'
    with summary_temporary.open('w', encoding='utf-8') as stream:
        json.dump(result['summary'], stream, allow_nan=False)
    generations = directory / '.result'
    generations.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    database = generations / (generation + '.sqlite3')
    storage.set_root(result)
    storage.publish(database)
    manifest = {'schemaVersion': 1, 'database': database.name,
                'databaseBytes': database.stat().st_size, 'databaseSha256': file_hash(database),
                'analysisId': result['summary']['id'], 'frames': result['summary']['frames']}
    pending = generations / (generation + '.store.tmp')
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, allow_nan=False)
    exports = ((temporary, directory / 'result.json'),
               (summary_temporary, directory / 'summary.json'))
    previous = {}
    for _staged, destination in exports:
        if destination.exists():
            backup = generations / (generation + '.' + destination.name + '.previous')
            shutil.copyfile(destination, backup)
            previous[destination] = backup
    replaced = []
    try:
        for staged, destination in exports:
            staged.replace(destination)
            replaced.append(destination)
        pending.replace(generations / 'current.store')
    except BaseException as error:
        # Process interruption is handled by the durable job-status gate. For
        # caught publication failures, also restore the compatibility exports.
        for destination in reversed(replaced):
            try:
                if destination in previous:
                    previous[destination].replace(destination)
                else:
                    destination.unlink()
            except BaseException as rollback_error:
                error.add_note(f'Result export rollback also failed: {rollback_error}')
        raise
    # Previous generation files are retained alongside their database. Readers
    # never use these diagnostic copies as the authoritative current result.


def _reject_constant(value):
    raise ValueError('Nonfinite saved result constant: ' + value)


class ResultDocument:
    """One owned reader and its lazy result tree, opened before HTTP headers."""
    def __init__(self, directory, analysis_id=None):
        self.directory = Path(directory)
        self.storage = None
        self.value = None
        self.closed = False
        try:
            marker = self.directory / '.result/current.store'
            if marker.exists():
                if marker.stat().st_size > 8192:
                    raise ValueError('Invalid result generation manifest')
                manifest = json.loads(marker.read_text(encoding='utf-8'), parse_constant=_reject_constant)
                if (not isinstance(manifest, dict) or type(manifest.get('schemaVersion')) is not int or
                        manifest['schemaVersion'] != 1 or not isinstance(manifest.get('database'), str) or
                        GENERATION_NAME.fullmatch(manifest['database']) is None):
                    raise ValueError('Invalid result generation manifest')
                database = marker.parent / manifest['database']
                if (not database.is_file() or database.stat().st_size != manifest.get('databaseBytes') or
                        file_hash(database) != manifest.get('databaseSha256')):
                    raise ValueError('Saved result generation changed or is missing')
                self.storage = ResultStorage.open(database)
                self.storage.__enter__()
                self.value = self.storage.root()
                summary = self.value.get('summary') if isinstance(self.value, dict) else None
                if (not isinstance(summary, dict) or summary.get('id') != manifest.get('analysisId') or
                        summary.get('frames') != manifest.get('frames')):
                    raise ValueError('Saved result generation identity does not align')
            else:
                if marker.parent.exists():
                    raise ValueError('Saved result generation manifest is missing')
                # Historical complete JSON is parsed through EOF into one owned
                # temporary database. No saved analysis is rewritten/migrated.
                self.storage = ResultStorage(Path(tempfile.gettempdir()))
                self.storage.__enter__()
                self.value = self.storage.read_json(self.directory / 'result.json', parse_constant=_reject_constant)
                if not isinstance(self.value, dict):
                    raise ValueError('Saved result is not an object')
                # Reject numeric overflow and unsupported values before headers,
                # without assembling a second full serialized representation.
                for _ in self.storage.json_chunks(self.value):
                    pass
            summary = self.value.get('summary')
            if analysis_id is not None and (not isinstance(summary, dict) or summary.get('id') != analysis_id):
                raise ValueError('Saved result belongs to a different analysis')
        except BaseException as error:
            self._close_preserving(error)
            raise

    def presented(self, analysis_id):
        return presentation_urls(self.value, self.directory, analysis_id)

    def json_chunks(self, value=None):
        return self.storage.json_chunks(self.value if value is None else value)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.storage is not None:
                self.storage.__exit__(None, None, None)

    def __enter__(self):
        return self

    def _close_preserving(self, error):
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f'Result reader cleanup also failed: {cleanup_error}')

    def __exit__(self, error_type, error, _traceback):
        if error_type is None:
            self.close()
        else:
            self._close_preserving(error)


@contextmanager
def open_result(directory, analysis_id=None):
    with ResultDocument(directory, analysis_id) as document:
        yield document.value


class ResultStreamingResponse(StreamingResponse):
    """Close the owned reader even when sending fails or the client disconnects."""
    def __init__(self, document, content, **kwargs):
        self.document = document
        super().__init__(content, **kwargs)

    async def __call__(self, scope, receive, send):
        error = None
        try:
            return await super().__call__(scope, receive, send)
        except BaseException as primary_error:
            error = primary_error
            raise
        finally:
            with anyio.CancelScope(shield=True):
                if error is None:
                    await run_in_threadpool(self.document.close)
                else:
                    await run_in_threadpool(self.document._close_preserving, error)
