"""Bind actual retry process exits, current code, frontend checks and limits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def counts(path):
    suites = list(ET.parse(path).getroot().iter('testsuite'))
    return {key: sum(int(suite.get(key, 0)) for suite in suites)
            for key in ('tests', 'errors', 'failures', 'skipped')}


def main():
    if {path.name for path in OUT.iterdir()} != {'build_receipt.py'}:
        raise RuntimeError('Preserve existing receipt; use a new directory')
    bindings = {}

    def bind(path):
        bindings[path.relative_to(ROOT).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}

    backend_dir = BASE/'retry_intent_checks_v1'
    frontend_dir = BASE/'retry_frontend_checks_v2'
    backend = json.loads((backend_dir/'receipt.json').read_text(encoding='utf-8'))
    frontend = json.loads((frontend_dir/'receipt.json').read_text(encoding='utf-8'))
    assert backend['status'] == frontend['status'] == 'PASS'
    assert backend['counts'] == counts(backend_dir/'junit.xml') == dict(tests=171, errors=0, failures=0, skipped=0)
    assert frontend['counts'] == counts(frontend_dir/'junit.xml') == dict(tests=58, errors=0, failures=0, skipped=0)
    assert backend['source_before'] == backend['source_after'] and backend['source_stable']
    assert frontend['source_before'] == frontend['source_after'] and frontend['source_stable']
    assert not backend['inference_import_attempts'] and not backend['forbidden_runtimes_loaded']
    assert backend['actual_host_ram_gpu_probes'] == 0
    assert not frontend['host_ram_gpu_probes'] and not frontend['model_training_or_inference']
    assert all(check['exit_code'] == 0 for check in frontend['checks'])
    current = dict(backend['source_after'])
    current.update(frontend['source_after'])
    for name, expected in current.items():
        assert digest(ROOT/name) == expected, name
        bind(ROOT/name)
        target = OUT/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    for directory in (backend_dir, BASE/'retry_frontend_checks_v1', frontend_dir):
        for name in ('receipt.json', 'started.json', 'run_checks.py', 'junit.xml'):
            bind(directory/name)
        for path in directory.glob('*.log'):
            bind(path)
        if (directory/'pytest.txt').exists():
            bind(directory/'pytest.txt')
        record = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
        for name, expected in record['source_after'].items():
            snapshot = directory/'sources'/name
            assert digest(snapshot) == expected
            bind(snapshot)
    children = []
    started = json.loads((backend_dir/'started.json').read_text(encoding='utf-8'))
    for path in sorted((backend_dir/'temp').rglob('retry_child_*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['pid'] != started['pid']
        assert record['forbidden_import_attempts'] == record['forbidden_loaded'] == []
        assert record['queued_before_exit'] == (record['case'] == 'after')
        selected = record['selected']
        storage = path.parent/'storage'
        with sqlite3.connect(storage/'analyses.sqlite3') as conn:
            rows = conn.execute('SELECT id,status,owner_id,retry_of FROM analyses WHERE retry_of=?',
                                (selected['source_analysis_id'],)).fetchall()
            assert rows == [(selected['analysis_id'], 'UPLOADED', 'local', selected['source_analysis_id'])]
            active = conn.execute('SELECT active FROM resource_reservations WHERE analysis_id=?',
                                  (selected['analysis_id'],)).fetchone()
            assert active in ((0,), (1,))
            # A post-commit process exit precedes reservation acknowledgement.
            # Accepted replay need not rewrite that flag: the queue consumes
            # it through the existing NOT EXISTS/UNION accounting contract.
            pending = conn.execute("SELECT COUNT(*) FROM (SELECT id FROM analyses WHERE owner_id='local' AND status IN ('UPLOADED','PROCESSING') UNION SELECT analysis_id FROM resource_reservations r WHERE owner_id='local' AND active=1 AND NOT EXISTS(SELECT 1 FROM analyses a WHERE a.id=r.analysis_id))").fetchone()[0]
            assert pending == 1
        original = storage/'uploads'/selected['analysis_id']/'original.mp4'
        assert digest(original) == record['original_sha256']
        children.append({'record': path.relative_to(ROOT).as_posix(), 'case': record['case'],
                         'pid': record['pid'], 'one_child_queue_row': rows[0][0],
                         'reservation_active_flag': active[0], 'deduplicated_pending_jobs': pending,
                         'original_sha256': digest(original)})
    assert len(children) == 2 and {row['case'] for row in children} == {'before', 'after'}
    for path in sorted((backend_dir/'temp').rglob('*')):
        if path.is_file():
            bind(path)
    review_path = BASE/'retry_intent_review_v1/backend_static_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    for name, expected in review['reviewed_files'].items():
        assert digest(ROOT/name) == expected
    bind(review_path)
    bind(BASE/'retry_intent_closure_v1/build_receipt.py')
    bind(BASE/'retry_intent_closure_v1/interruption.json')
    for path in (BASE/'retry_intent_baseline_v1').rglob('*'):
        if path.is_file():
            bind(path)
    for name in ('docs/plans/RETRY_IDEMPOTENCY.md', 'docs/PRODUCT_WORKFLOW.md',
                 'docs/plans/OWNER_RESOURCE_QUOTAS.md', 'docs/plans/YOLO26_EXECUTION_PROGRESS.md',
                 'docs/research/INDEPENDENT_TENNIS_MATCH_NEW_LEADS_2026-10-22.md'):
        bind(ROOT/name)
    for path in (ROOT/'frontend/dist').rglob('*'):
        if path.is_file():
            bind(path)
    bind(Path(__file__).resolve())
    receipt = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'FAILED_ANALYSIS_RETRY_DELIVERY_VERIFIED_MASTER_INCOMPLETE',
        'previous_goal_turn_classification': 'PROGRESS', 'goal_status': 'active', 'master_complete': False,
        'backend': {'tests': 171, 'failures': 0, 'errors': 0, 'skipped': 0,
            'receipt': (backend_dir/'receipt.json').relative_to(ROOT).as_posix()},
        'frontend': {'tests': 58, 'failures': 0, 'errors': 0, 'skipped': 0,
            'full_typecheck_lint_production_build': True,
            'receipt': (frontend_dir/'receipt.json').relative_to(ROOT).as_posix()},
        'actual_child_process_cases': children,
        'current_source_pins': current, 'source_pin_count': len(current),
        'independent_backend_review': review_path.relative_to(ROOT).as_posix(),
        'historical_frontend_attempt': {'tests': 53, 'passes': 46, 'failures': 7,
            'cause': 'API test cleanup used Node global localStorage.clear; fixed with explicit browser storage fixture',
            'diagnostic_runner_print': 'cp1252 error occurred only while printing failure diagnostics after its FAIL receipt was durably saved; later runner uses UTF-8'},
        'bounds': {'no_model_imports_or_execution': True, 'no_actual_host_ram_gpu_probe': True,
            'no_actual_browser_visual_qualification': True, 'no_independent_accuracy_or_long_match_claim': True,
            'no_power_loss_durability_claim': True},
        'remaining_scope': ['Initial upload-session/batch creation idempotency after lost creation acknowledgement',
            'Owner orphan-reservation recovery, controlled expiry/deletion and public service qualification',
            'Exact original hash is not bound at retry admission; same-length out-of-band mutation and shared hardlinks remain storage-integrity limits',
            'Storage-blocked browser persistence lasts only for current runtime; distinct keys are distinct intents',
            'Specialist baselines/training await human RAM completion notice; independent accuracy, long matches and real live operation remain incomplete'],
        'bindings': bindings}
    for name, record in bindings.items():
        assert digest(ROOT/name) == record['sha256'], name
    target = OUT/'receipt.json'
    target.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'bindings': len(bindings),
                      'source_pins': len(current), 'sha256': digest(target)}, indent=2))


if __name__ == '__main__':
    main()
