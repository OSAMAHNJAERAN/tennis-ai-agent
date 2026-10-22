"""Read-only startup gate and existing demo/export compatibility on localhost."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import urllib.request
import psutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = Path(__file__).parent
PREFLIGHT = BASE / 'result_storage_runtime_preflight_2026-10-06.json'
RECEIPT = BASE / 'result_storage_runtime_verification_2026-10-06.json'
DATABASE = ROOT / 'outputs/product/analyses.sqlite3'

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

def runtime_processes():
    result = []
    for process in psutil.process_iter(['pid', 'cmdline', 'cwd', 'create_time']):
        args = process.info['cmdline'] or []
        if not args or Path(args[0]).resolve() != Path(sys.executable).resolve():
            continue
        server = (len(args) == 2 and (ROOT / args[1]).resolve() == ROOT / 'scripts/run/serve_product.py')
        worker = args[1:] == ['-u', '-m', 'src.api.worker']
        if (server or worker) and Path(process.info['cwd'] or '').resolve() == ROOT:
            result.append(process.info)
    return result

def request(path):
    with urllib.request.urlopen('http://127.0.0.1:8000' + path, timeout=45) as response:
        return response.status, {key.lower(): value for key, value in response.headers.items()}, response.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        assert not PREFLIGHT.exists()
        processes = runtime_processes()
        listeners = [row.pid for row in psutil.net_connections(kind='tcp')
                     if row.status == 'LISTEN' and row.laddr.port == 8000]
        assert not processes and not listeners, 'Existing runtime detected; preserve it'
        before = snapshot()
        save(PREFLIGHT, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'matchingProcesses': processes,
                         'port8000Listeners': listeners, 'snapshot': before, 'productStateChanged': False})
        print(json.dumps({'readyForDocumentedStartup': True, 'savedJobs': len(before['jobs'])}))
        return
    assert not RECEIPT.exists()
    before = json.loads(PREFLIGHT.read_text(encoding='utf-8'))['snapshot']
    from src.api.media_revision import presentation_urls
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
    for entry in entries:
        analysis_id = entry.get('analysisId') or entry.get('analysis_id')
        assert analysis_id
        directory = ROOT / 'outputs/product/analyses' / analysis_id
        saved = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        presented = presentation_urls(saved, directory, analysis_id)
        expected = {'result': json.dumps(presented, allow_nan=False).encode(),
                    'download/report': json.dumps(presented, allow_nan=False).encode(),
                    'download/csv': ''.join(reporting.telemetry_csv_chunks(presented)).encode(),
                    'download/readable-report': reporting.readable_report(presented, directory).encode()}
        for suffix, content in expected.items():
            status, headers, body = request(f'/api/analyses/{analysis_id}/{suffix}')
            assert status == 200 and body == content, (analysis_id, suffix)
            if suffix.startswith('download/'):
                assert 'attachment;' in headers['content-disposition']
            exports.append({'analysisId': analysis_id, 'kind': suffix, 'status': status,
                            'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()})
    status, _, analyses = request('/api/analyses')
    assert status == 200 and len(json.loads(analyses)) == 10
    after = snapshot()
    assert before == after
    processes = runtime_processes()
    assert len(processes) == 2
    save(RECEIPT, {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
         'health': health, 'catalogEntries': len(entries), 'exports': exports, 'runtimeProcesses': processes,
         'savedJobsAndAnalysisFilesUnchanged': True, 'savedJobs': 10,
         'preflightSha256': digest(PREFLIGHT), 'scriptSha256': digest(Path(__file__)),
         'scope': 'Existing public legacy-result runtime compatibility; no inference or scientific qualification.'})
    print(json.dumps({'passed': True, 'demos': 3, 'exports': len(exports), 'savedJobsUnchanged': 10}))

if __name__ == '__main__':
    main()
