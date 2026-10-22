"""Bind terminal quota evidence and current revision; no model/resource probe."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
BASE = ART.parent


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    if {p.name for p in ART.iterdir()} != {'build_receipt.py'}:
        raise RuntimeError('Preserve existing closure outputs; use a new folder')
    attempts = []
    evidence = {}

    def bind(path):
        evidence[path.relative_to(ROOT).as_posix()] = {'sha256': digest(path), 'bytes': path.stat().st_size}

    names = ['owner_quota_checks_v1', 'owner_quota_checks_v2',
             'owner_quota_cleanup_checks_v1', 'owner_quota_cleanup_checks_v2',
             'owner_quota_cleanup_checks_v3']
    for name in names:
        directory = BASE/name
        receipt = json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
        assert receipt['source_before'] == receipt['source_after']
        assert receipt['source_stable'] and not receipt['inference_import_attempts']
        assert not receipt['forbidden_runtimes_loaded'] and receipt['actual_host_ram_gpu_probes'] == 0
        suites = list(ET.parse(directory/'junit.xml').getroot().iter('testsuite'))
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
                  for key in ('tests', 'errors', 'failures', 'skipped')}
        assert counts == receipt['counts']
        delta = []
        for path, expected in receipt['source_after'].items():
            snapshot = directory/'sources'/path
            assert digest(snapshot) == expected, (name, path)
            bind(snapshot)
            if digest(ROOT/path) != expected:
                delta.append(path)
        for filename in ('receipt.json', 'junit.xml', 'started.json', 'pytest.txt', 'run_checks.py'):
            bind(directory/filename)
        if name == 'owner_quota_checks_v2':
            assert receipt['status'] == 'PASS' and counts == dict(tests=227, errors=0, failures=0, skipped=0)
            assert delta == ['src/api/app.py', 'src/api/uploads.py', 'tests/test_owner_quota_api.py']
        if name == 'owner_quota_cleanup_checks_v3':
            assert receipt['status'] == 'PASS' and counts == dict(tests=110, errors=0, failures=0, skipped=0)
            assert not delta
        attempts.append({'directory': directory.relative_to(ROOT).as_posix(), 'status': receipt['status'],
            'counts': counts, 'elapsed_seconds': receipt['elapsed_seconds'], 'delta_to_current': delta,
            'synthetic_resource_calls': receipt['synthetic_resource_calls']})
    latest = json.loads((BASE/'owner_quota_cleanup_checks_v3/receipt.json').read_text(encoding='utf-8'))
    for path, expected in latest['source_after'].items():
        assert digest(ROOT/path) == expected
        bind(ROOT/path)
        target = ART/'sources'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/path, target)
    review_path = BASE/'owner_quota_review_v1/static_review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    review_delta = [row['path'] for row in review['files'] if digest(ROOT/row['path']) != row['sha256']]
    assert review_delta == ['tests/test_owner_quota_api.py']
    bind(review_path)
    for path in (BASE/'owner_quota_baseline_v1').rglob('*'):
        if path.is_file():
            bind(path)
    for name in ('docs/plans/OWNER_RESOURCE_QUOTAS.md', 'docs/PRODUCT_WORKFLOW.md',
                 'docs/deployment/PRODUCT_INSTALLATION.md', 'docs/plans/YOLO26_EXECUTION_PROGRESS.md'):
        bind(ROOT/name)
    bind(Path(__file__).resolve())
    # Retain synthetic runtime evidence (including real child-process records,
    # codecs, queue databases and final generation bytes) rather than claiming
    # the JUnit count alone establishes process recovery or publication.
    for name in ('owner_quota_checks_v2', 'owner_quota_cleanup_checks_v3'):
        for path in sorted((BASE/name/'temp').rglob('*')):
            if path.is_file():
                bind(path)
    receipt = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'OWNER_ADMISSION_SOFTWARE_VERIFIED_MASTER_INCOMPLETE',
        'previous_goal_turn_classification': 'PROGRESS',
        'master_complete': False, 'goal_status': 'active',
        'attempts': attempts, 'overlapping_counts_not_summed': True,
        'current_source_pins': latest['source_after'],
        'production_python_files': sum(path.startswith('src/') for path in latest['source_after']),
        'independent_static_review': {'receipt': review_path.relative_to(ROOT).as_posix(),
            'review_delta_to_current': review_delta,
            'production_files_match': True,
            'followup': 'Reviewer re-read the later explicit async two-message body and loopback hostname; no separate execution or v2 reviewer receipt is claimed.'},
        'execution_scope': 'Actual HTTP/SQLite owner admission, retained bytes, concurrent requests, lost acknowledgements, correction generation final guard and existing synthetic native worker/process recovery with real codecs. Selected model results/resources are injected at documented boundaries.',
        'remaining_limits': ['Not a continuous filesystem quota; stage and transfer scratch growth can occur between checks',
            'Resumable temporary chunk bytes require headroom beyond reserved input',
            'Active orphan reservations and retained crash files require explicit operator recovery; no automatic expiry/deletion',
            'No owner deletion, public capacity, global rate/abuse qualification, backup restore or account recovery completion',
            'No actual RAM/GPU admission while awaiting human memory-clear notification',
            'Specialist model training, independent accuracy, long matches and live operation remain incomplete'],
        'actual_host_ram_gpu_probes': 0, 'model_training_or_inference': False,
        'bindings': evidence}
    for name, record in evidence.items():
        assert digest(ROOT/name) == record['sha256'], name
    target = ART/'receipt.json'
    target.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'bindings': len(evidence),
                      'source_pins': len(latest['source_after']), 'sha256': digest(target)}, indent=2))


if __name__ == '__main__':
    main()
