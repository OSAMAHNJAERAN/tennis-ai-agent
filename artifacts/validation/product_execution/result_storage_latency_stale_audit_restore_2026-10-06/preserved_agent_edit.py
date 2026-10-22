"""Read-only idle-reload gate and existing demo/export checks on localhost.

PREPARED helper: execution requires root dispatch. It never stops or starts a
process, changes a job, runs inference, or alters saved analysis/frontend files.
"""
from datetime import datetime, timezone
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
import urllib.request
import psutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = Path(__file__).parent
PREFLIGHT = BASE / 'result_storage_latency_runtime_preflight_2026-10-06.json'
RECEIPT = BASE / 'result_storage_latency_runtime_verification_2026-10-06.json'
DATABASE = ROOT / 'outputs/product/analyses.sqlite3'
ORIGINAL = BASE / 'verify_result_storage_runtime_2026-10-06.py'
ORIGINAL_SHA256 = '52ced1809e03381911aa397cfb7b85e15810496ad02b23aa6380a81054f8605a'
BASE_PYTHON = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe')
FRONTEND = ROOT / 'frontend/dist'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def snapshot():
    with sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    assert len(rows) == 10 and not any(row['status'] in ('PROCESSING', 'UPLOADED') for row in rows)
    paths = [path for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()]
    return {'jobs': rows, 'savedAnalysisFiles': {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}}


def source_bindings():
    assert Path(sys.executable).resolve() == BASE_PYTHON.resolve(), 'Use the reviewed base interpreter'
    assert digest(ORIGINAL) == ORIGINAL_SHA256, 'Original runtime verifier changed'
    names = [ORIGINAL, Path(__file__).resolve(), ROOT / 'scripts/run/serve_product.py',
             ROOT / 'scripts/run/restart_idle_product_server.py', ROOT / 'data/demos/sources.json']
    names += sorted((ROOT / 'src/api').glob('*.py'))
    names += sorted(path for path in FRONTEND.rglob('*') if path.is_file())
    return {'files': {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                      for path in names},
            'baseInterpreter': {'path': str(BASE_PYTHON.resolve()), 'sha256': digest(BASE_PYTHON)}}


def process_record(process):
    return {'pid': process.pid, 'createTimeUnixSeconds': process.create_time(),
            'parentPid': process.ppid(), 'name': process.name(), 'exe': process.exe(),
            'cmdline': process.cmdline(), 'cwd': process.cwd()}


def same_identity(first, second):
    return (first['pid'], first['createTimeUnixSeconds']) == (second['pid'], second['createTimeUnixSeconds'])


def runtime_processes():
    # Discover recognizable runtimes before checking ownership, so a wrong cwd,
    # interpreter, extra argument, duplicate or orphan is rejected, not omitted.
    candidates, wrappers = [], []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if ('src.api.worker' in args or
                any(Path(argument).name.lower() == 'serve_product.py' for argument in args[1:])):
            # The documented tool launch has an enclosing cmd /c process.
            # Its final argument is a complete command string, not a Python
            # script argument. Accept only that exact wrapper with its actual
            # server child; it never counts as or owns the server listener.
            if (Path(process.exe()).name.lower() == 'cmd.exe' and len(args) == 3 and
                    args[1].lower() == '/c' and
                    args[2] == str(BASE_PYTHON) + ' scripts/run/serve_product.py' and
                    Path(process.cwd()).resolve() == ROOT):
                children = process.children()
                servers = [child for child in children if child.name().lower() != 'conhost.exe']
                consoles = [child for child in children if child.name().lower() == 'conhost.exe']
                if (len(servers) == 1 and servers[0].cmdline() ==
                        [str(BASE_PYTHON), 'scripts/run/serve_product.py'] and
                        Path(servers[0].exe()).resolve() == BASE_PYTHON.resolve() and
                        Path(servers[0].cwd()).resolve() == ROOT and
                        all(Path(child.exe()).resolve() == Path('C:/Windows/System32/conhost.exe').resolve()
                            for child in consoles)):
                    wrappers.append(process_record(process))
                    continue
            candidates.append(process)
    listeners = [{'pid': row.pid, 'host': row.laddr.ip, 'port': row.laddr.port}
                 for row in psutil.net_connections(kind='tcp')
                 if row.status == 'LISTEN' and row.laddr.port == 8000]
    listeners.sort(key=lambda row: (row['pid'] or -1, row['host']))
    if not candidates:
        assert not listeners, 'Port 8000 belongs to an unowned runtime'
        assert not wrappers, 'A launch wrapper has no accepted server'
        return {'state': 'ABSENT', 'processes': [], 'port8000Listeners': [], 'ownedDescendants': [],
                'launchWrappers': []}
    assert len(candidates) == 2, 'Require exactly one owned server and one owned worker'
    server, worker = None, None
    expected_python = BASE_PYTHON.resolve()
    for process in candidates:
        record = process_record(process)
        args = record['cmdline']
        assert Path(record['cwd']).resolve() == ROOT, 'Runtime cwd must be the reviewed workspace'
        assert Path(record['exe']).resolve() == expected_python, 'Runtime executable must be the base interpreter'
        assert args and Path(args[0]).resolve() == expected_python, 'Runtime command interpreter changed'
        if (len(args) == 2 and
                (ROOT / args[1]).resolve() == ROOT / 'scripts/run/serve_product.py'):
            assert server is None, 'Duplicate owned server'
            server = process
        elif args == [str(expected_python), '-u', '-m', 'src.api.worker']:
            assert worker is None, 'Duplicate owned worker'
            worker = process
        else:
            raise AssertionError('Unrecognized server/worker command; preserve runtime')
    assert server is not None and worker is not None
    assert worker.ppid() == server.pid, 'Worker must be a direct child of the owned server'
    assert {process.pid for process in server.children()} == {worker.pid}, 'Unexpected direct server child'
    consoles = worker.children()
    assert all(process.name().lower() == 'conhost.exe' and process.ppid() == worker.pid
               for process in consoles), 'Worker has a non-console child; preserve runtime'
    assert {process.pid for process in worker.children(recursive=True)} == {process.pid for process in consoles}, 'Unexpected worker descendant'
    assert {process.pid for process in server.children(recursive=True)} == {worker.pid, *(process.pid for process in consoles)}, 'Unexpected server descendant'
    assert listeners and all(row['pid'] == server.pid and row['host'] in ('127.0.0.1', '::1')
                             for row in listeners), 'Port 8000 must be owned by the loopback server'
    records = [process_record(server), process_record(worker)]
    # Console-host cwd/command introspection can be denied on Windows. Their
    # permitted role is established by name, direct parent and exact identity.
    descendants = [{'pid': process.pid, 'createTimeUnixSeconds': process.create_time(),
                    'parentPid': process.ppid(), 'name': process.name()}
                   for process in sorted(consoles, key=lambda item: item.pid)]
    # Recheck identities and topology after collecting the complete observed tree.
    assert worker.ppid() == server.pid
    assert {process.pid for process in server.children(recursive=True)} == {worker.pid, *(record['pid'] for record in descendants)}
    for record in records + descendants:
        assert psutil.Process(record['pid']).create_time() == record['createTimeUnixSeconds'], 'Runtime identity changed during review'
    return {'state': 'OWNED_IDLE_PAIR', 'processes': records,
            'port8000Listeners': listeners, 'ownedDescendants': descendants,
            'launchWrappers': wrappers}


def previous_identities_retired(before):
    retired = []
    for record in before['processes'] + before['ownedDescendants']:
        try:
            gone = psutil.Process(record['pid']).create_time() != record['createTimeUnixSeconds']
        except psutil.NoSuchProcess:
            gone = True
        assert gone, 'The exact previous runtime identity remains alive; idle reload incomplete'
        retired.append({'pid': record['pid'], 'createTimeUnixSeconds': record['createTimeUnixSeconds'],
                        'retired': True})
    return retired


def request(path):
    with urllib.request.urlopen('http://127.0.0.1:8000' + path, timeout=45) as response:
        return response.status, {key.lower(): value for key, value in response.headers.items()}, response.read()


class IndexAssets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        path = attributes.get('src') if tag == 'script' else attributes.get('href') if tag == 'link' else None
        if isinstance(path, str) and path.startswith('/assets/'):
            self.paths.add(path)


def frontend_checks():
    expected_index = (FRONTEND / 'index.html').read_bytes()
    status, headers, body = request('/')
    assert status == 200 and body == expected_index, 'Served frontend index differs from saved build'
    assert headers.get('cache-control') == 'no-cache', 'SPA index must retain no-cache'
    parser = IndexAssets()
    parser.feed(expected_index.decode('utf-8'))
    assert parser.paths and any(path.endswith('.js') for path in parser.paths), 'Built index has no script asset'
    assets = []
    for path in sorted(parser.paths):
        local = (FRONTEND / path.lstrip('/')).resolve()
        assert local.is_relative_to(FRONTEND.resolve()) and local.is_file(), 'Invalid local index asset'
        expected = local.read_bytes()
        status, asset_headers, body = request(path)
        assert status == 200 and body == expected, path
        assets.append({'url': path, 'status': status, 'bytes': len(body),
                       'sha256': hashlib.sha256(body).hexdigest(),
                       'cacheControl': asset_headers.get('cache-control')})
    return {'index': {'status': 200, 'bytes': len(expected_index),
                      'sha256': hashlib.sha256(expected_index).hexdigest(),
                      'cacheControl': headers['cache-control']}, 'assets': assets}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        assert not PREFLIGHT.exists()
        sources = source_bindings()
        runtime = runtime_processes()
        before = snapshot()
        assert before == snapshot(), 'Saved jobs/files changed during preflight; preserve runtime'
        assert runtime == runtime_processes(), 'Runtime changed during preflight; preserve it'
        assert sources == source_bindings(), 'Reviewed code/frontend changed during preflight'
        save(PREFLIGHT, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'runtime': runtime,
                         'matchingProcesses': runtime['processes'],
                         'port8000Listeners': runtime['port8000Listeners'], 'snapshot': before,
                         'sourceBindings': sources, 'productStateChanged': False,
                         'acceptedWorkCancelled': False, 'runtimeStoppedOrStarted': False,
                         'scope': 'Read-only preflight before root guarded idle stop and documented startup.'})
        print(json.dumps({'readyForDocumentedIdleReload': True, 'runtimeState': runtime['state'],
                          'savedJobs': len(before['jobs']),
                          'savedAnalysisFiles': len(before['savedAnalysisFiles'])}))
        return
    assert not RECEIPT.exists()
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    before = preflight['snapshot']
    assert preflight['sourceBindings'] == source_bindings(), 'Reviewed code/frontend changed after preflight'
    assert before == snapshot(), 'Saved jobs/files changed before compatibility checks'
    runtime_before = runtime_processes()
    assert runtime_before['state'] == 'OWNED_IDLE_PAIR', 'Documented server and worker must be running'
    retired = previous_identities_retired(preflight['runtime'])
    from src.api.media_revision import presentation_urls
    from src.api.media_source import ATTRIBUTION_FIELDS
    from src.api import reporting
    status, _, health_bytes = request('/api/health')
    assert status == 200
    health = json.loads(health_bytes)
    assert health['status'] == 'ok' and health['worker_available']
    status, _, catalog_bytes = request('/api/demos')
    catalog = json.loads(catalog_bytes)
    # Catalog schema is inspected dynamically, never assume source order.
    entries = catalog if isinstance(catalog, list) else catalog.get('demos', [])
    assert status == 200 and len(entries) == 3
    exports = []
    credits = []
    media_headers = []
    for entry in entries:
        analysis_id = entry.get('analysisId') or entry.get('analysis_id')
        assert analysis_id
        directory = ROOT / 'outputs/product/analyses' / analysis_id
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        presented = presentation_urls(saved, directory, analysis_id)
        source = presented['mediaSource']
        assert entry['attribution'] == {key: source[key] for key in ATTRIBUTION_FIELDS}
        assert entry['qualificationEvidence'] is False
        credit = reporting.source_credit(source)
        assert credit, 'Public demo attribution must remain available'
        credits.append({'analysisId': analysis_id, 'attribution': entry['attribution'],
                        'presentedMediaSource': source, 'catalogMatchesPresentedCredit': True})
        expected = {'result': json.dumps(presented, allow_nan=False).encode(),
                    'download/report': json.dumps(presented, allow_nan=False).encode(),
                    'download/csv': ''.join(reporting.telemetry_csv_chunks(presented)).encode(),
                    'download/readable-report': reporting.readable_report(presented, directory).encode()}
        for suffix, content in expected.items():
            status, headers, body = request(f'/api/analyses/{analysis_id}/{suffix}')
            assert status == 200 and body == content, (analysis_id, suffix)
            if suffix.startswith('download/'):
                assert 'attachment;' in headers['content-disposition']
            if suffix == 'download/report':
                assert headers.get('cache-control') == 'no-cache'
            if suffix == 'download/readable-report':
                assert credit in body.decode('utf-8'), 'HTML footage credit missing'
            exports.append({'analysisId': analysis_id, 'kind': suffix, 'status': status,
                            'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(),
                            'cacheControl': headers.get('cache-control')})
        # Header-only HEAD would be routed differently by this API. A bounded
        # byte-range GET verifies the existing no-cache media contract without
        # downloading an entire source/annotated video. No source byte is written.
        for kind in ('source', 'annotated'):
            req = urllib.request.Request(f'http://127.0.0.1:8000/api/analyses/{analysis_id}/media/{kind}',
                                         headers={'Range': 'bytes=0-0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                assert response.status == 206
                assert response.headers.get('Cache-Control') == 'no-cache'
                local = directory / ('source.mp4' if kind == 'source' else 'annotated_web.mp4')
                with local.open('rb') as stream:
                    assert response.read(2) == stream.read(1)
                media_headers.append({'analysisId': analysis_id, 'kind': kind, 'status': response.status,
                                      'cacheControl': response.headers.get('Cache-Control'),
                                      'range': 'bytes=0-0', 'bytesRead': 1})
    status, _, analyses = request('/api/analyses')
    assert status == 200 and len(json.loads(analyses)) == 10
    frontend = frontend_checks()
    after = snapshot()
    assert before == after
    runtime_after = runtime_processes()
    assert runtime_before == runtime_after, 'Owned runtime changed during compatibility checks'
    assert preflight['sourceBindings'] == source_bindings(), 'Reviewed code/frontend changed during verification'
    assert len(exports) == 12
    save(RECEIPT, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
         'health': health, 'catalogEntries': len(entries), 'exports': exports,
         'credits': credits, 'mediaCacheHeaders': media_headers, 'frontend': frontend,
         'runtimeProcesses': runtime_after['processes'], 'runtime': runtime_after,
         'previousRuntimeIdentitiesRetired': retired,
         'savedJobsAndAnalysisFilesUnchanged': True, 'savedJobs': 10,
         'savedAnalysisFiles': len(before['savedAnalysisFiles']), 'snapshotBefore': before, 'snapshotAfter': after,
         'sourceBindings': preflight['sourceBindings'], 'originalVerifierSha256': digest(ORIGINAL),
         'preflightSha256': digest(PREFLIGHT), 'scriptSha256': digest(Path(__file__)),
         'scope': 'Existing public legacy-result idle-reload compatibility; no inference or scientific qualification.',
         'runtimeStoppedOrStartedByThisVerifier': False, 'productStateChanged': False})
    print(json.dumps({'passed': True, 'demos': 3, 'exports': len(exports),
                      'savedJobsUnchanged': 10, 'savedAnalysisFilesUnchanged': len(before['savedAnalysisFiles']),
                      'frontendAssetsExact': len(frontend['assets'])}))


if __name__ == '__main__':
    main()
