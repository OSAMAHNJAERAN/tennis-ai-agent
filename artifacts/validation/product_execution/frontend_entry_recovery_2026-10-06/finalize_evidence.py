"""Seal current frontend repair evidence without inference or product mutations."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]

def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    output = BASE / 'final_verification.json'
    assert not output.exists()
    docs = import_file('entry_document_contract', BASE / 'document_evidence.py')
    docs.track(Path(__file__))
    document_receipt = docs.load(BASE / 'documentation_receipt.json')
    assert document_receipt['passed'] and document_receipt['immutableEvidenceUnchanged']
    assert document_receipt['assetRecordsUnchanged'] and document_receipt['olderManifestRecordsUnchanged']
    docs.track(BASE / 'document_evidence.py', document_receipt['helperBinding'])
    docs.verify_map(document_receipt['sourceBindings'])
    docs.verify_map(document_receipt['evidenceBindings'])
    docs.verify_map(document_receipt['currentDocumentHashes'])
    old_seal = docs.load(docs.EVENT_SEAL, docs.EVENT_SEAL_SHA256)
    for name, record in document_receipt['preservedBeforeFiles'].items():
        docs.track(ROOT / record['path'], {'bytes': record['bytes'], 'sha256': record['sha256']})
        expected = old_seal['bindings'].get(name)
        if expected:
            assert {'bytes': record['bytes'], 'sha256': record['sha256']} == expected
    for name, expected in old_seal['bindings'].items():
        if name not in docs.DOCUMENTS and not name.startswith('frontend/'):
            docs.track(docs.resolve(name), expected)
    backend = docs.load(docs.EVENT / 'backend_final/receipt.json')
    assert backend['passed'] and backend['counts'] == {'tests': 1316, 'failures': 0, 'errors': 0, 'skipped': 0}
    backend_bindings = docs.load(docs.EVENT / 'backend_final/source_bindings.json')
    docs.verify_map(backend_bindings)
    assert {docs.name(path) for folder in (ROOT / 'src', ROOT / 'tests') for path in folder.rglob('*.py')}.issubset(backend_bindings)
    focused = docs.focused_gate()
    restore = docs.restore_gate()
    publication = docs.publication_gate(restore, old_seal)
    browser = docs.browser_gate(publication)
    runtime = docs.runtime_gate(publication)
    runtime_module = import_file('entry_current_runtime', BASE / 'verify_runtime.py')
    assert runtime_module.runtime_processes() == runtime['runtime']
    assert runtime_module.snapshot() == runtime['snapshotAfter']
    status, _, health_bytes = runtime_module.request('/api/health')
    assert status == 200 and json.loads(health_bytes) == runtime['health']
    host = docs.load(BASE / 'verify_installation.json')
    assert host['passed'] and host['inputsUnchanged'] and host['exitCode'] == 0 and host['checkCount'] == 75
    docs.verify_map(host['inputs'])
    host_report = docs.load(ROOT / host['report'], host['reportSha256'])
    docs.track(BASE / 'verify_installation.log', host['logSha256'])
    assert host_report['checksPassed'] and len(host_report['checks']) == 75 and all(check['passed'] for check in host_report['checks'])
    assert host_report['manifestSha256'] == document_receipt['currentDocumentHashes']['docs/deployment/PRODUCT_RESTORE_MANIFEST.json']['sha256']
    assert host_report['inferenceRun'] is False and not host_report['modelsImported'] and host_report['productionQualified'] is False
    current_manifest = docs.load(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
    previous_manifest = json.loads((ROOT / document_receipt['beforeDocumentBackupBindings']['docs/deployment/PRODUCT_RESTORE_MANIFEST.json']['path']).read_text(encoding='utf-8'))
    assert current_manifest['assets'] == previous_manifest['assets'] and current_manifest['productionQualified'] is False
    assert current_manifest['validationEvidence'][:len(previous_manifest['validationEvidence'])] == previous_manifest['validationEvidence']
    assert all(current_manifest['validation'][name] == value for name, value in previous_manifest['validation'].items())
    # Preserve every small owned unit artifact, including RED/frozen snapshots,
    # old build, screenshots, preparation variants and rollback before-images.
    for path in sorted(BASE.rglob('*')):
        if path.is_file() and path != output and '__pycache__' not in path.parts:
            docs.track(path)
    docs.unchanged()
    assert runtime_module.runtime_processes() == runtime['runtime'] and runtime_module.snapshot() == runtime['snapshotAfter']
    final = {
        'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'frontendTestsPassed': restore['testSummary']['passed'], 'frontendTestFiles': restore['testSummary']['executedFiles'],
        'frontendInputsExact': restore['copiedFiles'], 'frontendProductionFilesExact': len(restore['buildBindings']),
        'previousBuildFilesPreserved': len(publication['buildBefore']), 'previousBackendTestsPassedOnUnchangedSources': backend['counts']['tests'],
        'cleanHostChecksPassed': host['checkCount'], 'layoutAssertions': browser['layoutAssertions'], 'focused': focused,
        'existingDemoExportsExact': len(runtime['exports']), 'httpFrontendAssetsExact': len(runtime['frontend']['assets']),
        'savedJobsUnchanged': runtime['savedJobs'], 'savedAnalysisFilesUnchanged': runtime['savedAnalysisFiles'],
        'runtimeRetainedWithoutRestart': True, 'runtimeProcesses': runtime['runtimeProcesses'],
        'bindings': docs.TRACKED, 'inferenceRun': False, 'productionQualified': False, 'accuracyQualified': False,
        'fullMatchQualified': False, 'physicalSpeedAnd3DQualified': False,
        'scope': 'Verified current frontend repair, software restoration and actual ordinary browser layout/playback; existing data/API compatibility only.',
        'remainingWork': [
            'Current model-worker inference on event/source revision remains pending; previous guarded attempts declined before launch.',
            'Resident pipeline pose/racket/player/trajectory and browser arrays still constrain long-match capacity.',
            'Representative owner-authorized recordings, independently annotated untouched test split and physical speed/3D references remain unavailable.',
            'Ball, participant identities, pose, racket, masks, calibration, physical speed and reconstructed flight remain unqualified.',
            'Forced autoplay rejection and actual GPU-failure behavior were not measured by this browser unit.',
            'Full-match resource capacity, release/deployment gates and all remaining execution-prompt requirements must still be fulfilled.'
        ]
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(final, stream, indent=2, allow_nan=False)
    print(json.dumps({key: final[key] for key in ('passed', 'frontendTestsPassed', 'frontendInputsExact', 'frontendProductionFilesExact', 'cleanHostChecksPassed', 'savedJobsUnchanged', 'savedAnalysisFilesUnchanged')}))

if __name__ == '__main__':
    main()
