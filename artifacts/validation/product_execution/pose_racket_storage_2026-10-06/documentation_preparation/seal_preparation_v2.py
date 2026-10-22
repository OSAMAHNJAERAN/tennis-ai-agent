"""Static AST and hash seal only; no helper/product imports or workloads."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
ROOT = UNIT.parents[3]


def bind(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    target = HERE / 'preparation_v2.json'
    assert not target.exists(), 'Prepared seal is exclusively created once'
    static = {}
    for leaf in ('document_evidence_v2.py', 'verify_installation_v2.py', 'seal_preparation_v2.py'):
        path = HERE / leaf
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        static[leaf] = {'binding': bind(path), 'astParsed': True, 'topLevelStatements': len(tree.body)}
    originals = load(UNIT / 'before.json')
    docs = {name: row['before'] for name, row in originals['files'].items() if name.startswith('docs/')}
    assert len(docs) == 3 and all(bind(ROOT / name) == value for name, value in docs.items())
    immutable = {}

    def add(path, expected=None):
        path = path.resolve()
        value = bind(path)
        assert expected is None or value == expected
        key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        assert key not in immutable or immutable[key] == value
        immutable[key] = value

    # Freeze the integrated backend's complete source/test/config/helper inventory.
    backend = load(UNIT / 'backend_full/source_bindings_before.json')
    assert backend == load(UNIT / 'backend_full/source_bindings_after.json')
    for name, value in backend['files'].items():
        add(Path(name) if Path(name).is_absolute() else ROOT / name, value)
    for key in ('baseInterpreter', 'actualHelperInterpreter'):
        row = backend[key]
        add(Path(row['path']), {field: row[field] for field in ('bytes', 'sha256')})
    for row in originals['files'].values():
        add(ROOT / row['preservedPath'], row['before'])
    add(UNIT / 'before.json')
    for leaf in ('preparation_correction.json', 'worker_preparation_revision.json', 'verify_runtime.py', 'stop_idle_runtime.py'):
        add(UNIT / leaf)
    for directory in ('storage_focused', 'pipeline_red', 'pipeline_green', 'backend_full'):
        for path in (UNIT / directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and 'tmp' not in path.parts:
                add(path)
    # The full prior frontend evidence, exact current sources/build, selected
    # recipe, checkpoints, license metadata and public credits are protected.
    previous = ROOT / 'artifacts/validation/product_execution/frontend_entry_recovery_2026-10-06'
    for path in previous.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts:
            add(path)
    oldseal = originals['previousFrontendSeal']
    add(ROOT / oldseal['path'], {key: oldseal[key] for key in ('bytes', 'sha256')})
    restore = ROOT / 'artifacts/validation/product_installation/frontend_source_restore_entry_recovery_2026-10-06'
    front = load(restore / 'receipt.json')
    assert front['passed'] and front['boundCopiedInputsUnchanged'] and front['boundSelectedInputsUnchanged']
    assert len(front['sourceBindings']) + len(front['sharedBindings']) == 127 and len(front['buildBindings']) == 47
    add(restore / 'receipt.json')
    add(restore / 'tests.json')
    for name, expected in front['sourceBindings'].items():
        path = ROOT / 'frontend' / name
        assert bind(path)['sha256'] == expected
        add(path)
    for name, expected in front['sharedBindings'].items():
        path = ROOT / name
        assert bind(path)['sha256'] == expected
        add(path)
    for name, expected in front['buildBindings'].items():
        path = ROOT / 'frontend/dist' / name
        assert bind(path)['sha256'] == expected
        add(path)
    for asset in load(ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')['assets']:
        add(ROOT / asset['path'], {key: asset[key] for key in ('bytes', 'sha256')})
    for path in (ROOT / 'data/demos').rglob('*'):
        if path.is_file():
            add(path)
    for path in (UNIT / 'measurement').iterdir():
        if path.is_file():
            add(path)
    for path in (UNIT / 'measurement/contract_tests').iterdir():
        if path.is_file():
            add(path)
    if (UNIT / 'measurement/run/report.json').exists():
        add(UNIT / 'measurement/run/report.json')
    for path in (UNIT / 'worker_smoke_v2').iterdir():
        if path.is_file():
            add(path)
    for leaf in ('scripts/audit/verify_pose_racket_storage_worker_smoke_v2.py',
                 'scripts/setup/verify_product_installation.py',
                 'scripts/audit/measure_candidate_storage.py',
                 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'):
        add(ROOT / leaf)
    for leaf in static:
        add(HERE / leaf)
    for leaf in ('document_evidence.py', 'verify_installation.py', 'seal_preparation.py',
                 'preparation.json', 'initial_dry_run.json', 'revision_2_origin.json', 'prepare_revision_2.py'):
        add(HERE / leaf)
    value = {'preparedAt': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_NOT_EXECUTED',
        'documentationPublished': False, 'hostVerifierRun': False, 'modelsImported': False, 'inferenceRun': False,
        'scope': 'Static preparation only. Publisher requires terminal measurement, worker completed or RAM-declined status, and fresh idle-reload verification. Clean-host wrapper requires completed publication.',
        'originalDocuments': docs, 'immutableInputs': immutable, 'staticReview': static,
        'historicalFrontend': {'testsPassed': 169, 'sourceInputs': 127, 'productionFiles': 47,
            'seal': oldseal, 'newFrontendWorkloadRun': False},
        'commands': {
            'validate': ['C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe', '-B',
                         str(HERE / 'document_evidence_v2.py'), '--validate-only'],
            'publishAfterRootReview': ['C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe', '-B',
                         str(HERE / 'document_evidence_v2.py'), '--apply'],
            'freshHostAfterPublication': ['C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe', '-B',
                         str(HERE / 'verify_installation_v2.py')]},
        'futureEvidencePaths': [str(UNIT / name) for name in (
            'measurement/run/report.json', 'worker_smoke_v2/resource_receipt.json',
            'runtime_preflight.json', 'idle_stop_receipt.json', 'runtime_verification.json',
            'documentation_receipt.json', 'verify_installation.json')],
        'qualificationEvidence': False, 'productionQualified': False, 'fullMatchQualified': False,
        'accuracyQualified': False, 'physicalSpeedAnd3DQualified': False}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': value['status'], 'immutableInputs': len(immutable),
                      'helpers': static, 'documentationPublished': False, 'hostVerifierRun': False}))


if __name__ == '__main__':
    main()
