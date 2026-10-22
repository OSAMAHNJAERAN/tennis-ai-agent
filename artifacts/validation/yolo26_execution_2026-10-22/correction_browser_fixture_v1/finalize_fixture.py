"""Bind prepared fixture entry points without starting its server/worker."""
import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[3]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((ART/'fixture_manifest.json').read_text(encoding='utf-8'))
assert manifest['status'] == 'PREPARED_VERIFIED_SYNTHETIC_FIXTURE_SERVER_NOT_STARTED'
assert manifest['source_stable'] and manifest['source_before'] == manifest['source_after']
assert not manifest['inference_import_attempts'] and not manifest['inference_loaded']
assert manifest['native_rule_reproduction_verified'] and manifest['owner_result_review_media_verified']
assert manifest['guest_foreign_owner_rejected']
assert all(sha(ROOT/name) == digest for name, digest in manifest['source_after'].items())
runtime_receipt = ART/'runtime.json'
names = ['generate.py', 'serve.py', 'README.md', 'finalize_fixture.py', 'fixture_manifest.json',
    'initial_review_source.json', 'request_example.json', 'synthetic_credentials.json',
    'empty-demos.json', 'unused-original-inference.yaml', 'failed_generate_v1.py', 'generation_failure_v1.txt']
names += [str(path.relative_to(ART)) for path in sorted((ART/'factory_ast').glob('*.py'))]
result = {'status': 'PREPARED_VERIFIED_FIXTURE',
    'fixture_files_sha256': {name: sha(ART/name) for name in names},
    'current_product_python_source_pins': len(manifest['source_after']),
    'analysis_url': manifest['analysis_url'], 'historical_url': manifest['historical_url'],
    'storage_root': manifest['storage_root'], 'scope': manifest['scope'],
    'server_started_by_preparation': False, 'worker_started_by_preparation': False,
    'later_runtime_receipt_sha256': sha(runtime_receipt) if runtime_receipt.exists() else None,
    'later_process_liveness': 'NOT_VERIFIED_BY_THIS_PREPARATION_FINALIZER'}
(ART/'fixture_bundle_hashes.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': result['status'], 'source_pins': result['current_product_python_source_pins'],
    'fixture_manifest_sha256': sha(ART/'fixture_manifest.json'),
    'bundle_sha256': sha(ART/'fixture_bundle_hashes.json')}))
