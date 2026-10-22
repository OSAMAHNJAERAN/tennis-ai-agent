"""Static-only preparation receipt writer; never imports/runs either verifier."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / 'verify_result_storage_runtime_2026-10-06.py'
HELPER = BASE / 'verify_result_storage_latency_runtime_2026-10-06.py'
PROTOCOL = BASE / 'verify_result_storage_latency_runtime_PROTOCOL_2026-10-06.md'
RECEIPT = BASE / 'verify_result_storage_latency_runtime_PREPARED_2026-10-06.json'


def binding(path):
    payload = path.read_bytes()
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def shape(node):
    return ast.dump(node, include_attributes=False)


def children(tree):
    return {shape(node) for node in ast.walk(tree)}


original_source = ORIGINAL.read_text(encoding='utf-8')
source = HELPER.read_text(encoding='utf-8')
original = ast.parse(original_source, str(ORIGINAL))
candidate = ast.parse(source, str(HELPER))
compile(candidate, str(HELPER), 'exec')
assert binding(ORIGINAL)['sha256'] == '52ced1809e03381911aa397cfb7b85e15810496ad02b23aa6380a81054f8605a'
old_functions, new_functions = functions(original), functions(candidate)
preserved = ('digest', 'save', 'snapshot', 'request')
for name in preserved:
    assert shape(old_functions[name]) == shape(new_functions[name]), name
old_main, new_main = old_functions['main'], new_functions['main']
new_nodes = children(new_main)
retained_export_nodes = []
for node in ast.walk(old_main):
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ('saved', 'presented', 'expected') for target in node.targets):
        assert shape(node) in new_nodes
        retained_export_nodes.append(ast.get_source_segment(original_source, node))
    if isinstance(node, ast.Assert) and ast.get_source_segment(original_source, node).startswith('assert status == 200 and body == content'):
        assert shape(node) in new_nodes
        retained_export_nodes.append(ast.get_source_segment(original_source, node))
assert len(retained_export_nodes) == 4
assert not (BASE / 'result_storage_latency_runtime_preflight_2026-10-06.json').exists()
assert not (BASE / 'result_storage_latency_runtime_verification_2026-10-06.json').exists()
assert not RECEIPT.exists()
receipt = {
    'preparedAt': datetime.now(timezone.utc).isoformat(),
    'passed': True,
    'scope': 'Static AST parse/compile and exact original-node comparison only. Neither verifier nor production code is imported or executed.',
    'originalVerifier': {'path': ORIGINAL.relative_to(ROOT).as_posix(), **binding(ORIGINAL)},
    'newVerifier': {'path': HELPER.relative_to(ROOT).as_posix(), **binding(HELPER)},
    'protocol': {'path': PROTOCOL.relative_to(ROOT).as_posix(), **binding(PROTOCOL)},
    'preparationWriter': {'path': Path(__file__).resolve().relative_to(ROOT).as_posix(), **binding(Path(__file__))},
    'astParseAndCompilePassed': True,
    'preservedOriginalFunctionASTsExact': list(preserved),
    'preservedCompleteExportExpectedValuesAndByteAssertionASTs': retained_export_nodes,
    'intentionalChanges': [
        'Distinct preflight/result artifact names; original receipts never reused.',
        'Preflight permits only exact owned idle server/worker or absent/no-listener state.',
        'Complete descendant checks, exact PID/create-time ownership and previous-identity retirement.',
        'Two equal complete ten-row/all-analysis-file snapshots before idle reload.',
        'Source/base-interpreter/full-frontend bindings across preflight/postverification.',
        'Explicit presented attribution and declared no-cache checks.',
        'Bounded one-byte media range requests and exact served index/referenced assets.',
        'Full before/after state snapshots retained in successful postverification receipt.',
    ],
    'helperImportedOrExecuted': False,
    'liveProcessesOrQueueInspectedByPreparation': False,
    'savedAnalysisSnapshotHasNotBeenTakenByPreparation': True,
    'networkRequestsMade': False,
    'runtimeStoppedOrStarted': False,
    'inferenceLaunched': False,
    'requiresRootReviewAndDispatch': True,
}
with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'prepared': True, 'helper': binding(HELPER), 'receipt': binding(RECEIPT),
                  'helperImportedOrExecuted': False, 'networkRequestsMade': False}, sort_keys=True))
