"""Read-only recheck of sealed completed evidence and the exact live stress owner."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
RESTORE = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_capacity_2026-10-05'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


seal = read(RESTORE / 'verification_seal.json')
for name, expected in {**seal['evidenceBindings'], **seal['changedSourceBindings']}.items():
    assert digest(ROOT / name) == expected, name
receipt = read(RESTORE / 'receipt.json')
target = ROOT / receipt['target']
for source, copied, bindings in ((ROOT / 'frontend', target, receipt['sourceBindings']), (ROOT, RESTORE, receipt['sharedBindings'])):
    for name, expected in bindings.items():
        assert digest(source / name) == digest(copied / name) == expected, name
for name, expected in seal['buildBindings'].items():
    assert digest(ROOT / 'frontend/dist' / name) == digest(target / 'dist' / name) == expected, name
host_path = ROOT / 'artifacts/validation/product_installation/capacity_source_restoration_host_verification_2026-10-05.json'
host = read(host_path)
manifest = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
assert host['checksPassed'] and len(host['checks']) == 75 and not host['warnings']
assert host['apiSmoke']['workerDisabled'] and not host['apiSmoke']['modelModulesImported']
assert host['manifestSha256'] == digest(manifest)
stress = HERE / 'repeated_source_capacity_uniform_clock_2026-10-05'
protocol = read(stress / 'protocol.json')
for name, expected in protocol['sourceFiles'].items():
    assert digest(ROOT / name) == expected, name
assert digest(ROOT / 'scripts/audit/run_repeated_source_capacity.py') == protocol['scriptSha256']
owner = read(stress / 'running_process.json')
process = psutil.Process(owner['pid'])
assert process.is_running() and Path(process.cwd()).resolve() == ROOT
command = process.cmdline()
assert str(ROOT / 'scripts/audit/run_repeated_source_capacity.py') in command
identity = read(stress / 'accepted_job.json')['analysisId']
from src.api.jobs import JobStore
job = JobStore(stress / 'isolated_product').get(identity)
samples = stress / 'resource_samples.jsonl'
with samples.open(encoding='utf-8') as stream:
    last = None
    for line in stream:
        last = json.loads(line)
report = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'scope': 'Scoped completed evidence recheck plus live owned stress process; pending inference is not qualified.',
          'evidenceFiles': len(seal['evidenceBindings']), 'changedSources': len(seal['changedSourceBindings']),
          'currentAndCopiedInputs': seal['currentAndCopiedInputs'], 'currentAndCopiedBuildFiles': len(seal['buildBindings']),
          'sealSha256': digest(RESTORE / 'verification_seal.json'), 'manifestSha256': digest(manifest),
          'hostVerifiedAt': host['verifiedAt'], 'hostReceiptSha256': digest(host_path),
          'liveStress': {'pid': process.pid, 'createTime': process.create_time(), 'command': command,
                         'analysisId': identity, 'status': job['status'], 'stage': job['stage'], 'lastResourceSample': last},
          'qualificationEvidence': False}
save_path = HERE / 'capacity_evidence_recheck_2026-10-05.json'
with save_path.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps(report))
