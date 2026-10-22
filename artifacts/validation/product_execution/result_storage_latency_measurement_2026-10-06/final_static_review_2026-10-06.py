"""Static review and exclusive preparation seal; never imports the helper.

Authorized after root reported final production edits. This reads and hashes
sources, compiles ASTs without execution, preserves its receipt, then exclusively
writes the inventory expected by the parent benchmark. It launches no workload.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06'
ORIGINAL_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_measurement_2026-10-06'
EXPECTED_HELPER = 'b33cc0a93fa3ce3b6069b2cd297a29ffbb8d47797e41be0139c08095fee771cc'
EXPECTED_STORAGE = 'ac2dd55e91435562cf031aae47b0107cab13eb6a57f2f1acb62405817f97b343'


def binding(path):
    payload = path.read_bytes()
    return {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


def save_exclusive(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def constants(tree):
    values = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in ('MONITOR', 'MONITOR_SHA256', 'FROZEN', 'FROZEN_RECEIPT', 'CURRENT', 'HISTORICAL'):
            continue
        if isinstance(node.value, ast.Tuple):
            values[name] = tuple(values[item.id] if isinstance(item, ast.Name)
                                 else ast.literal_eval(item) for item in node.value.elts)
        else:
            values[name] = ast.literal_eval(node.value)
    return values


def function_nodes(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


helper = UNIT / 'helper.py'
helper_binding = binding(helper)
assert helper_binding['sha256'] == EXPECTED_HELPER, 'Prepared helper changed'
tree = ast.parse(helper.read_text(encoding='utf-8'), str(helper))
compile(tree, str(helper), 'exec')
values = constants(tree)
initial = json.loads((UNIT / 'static_sanity_receipt.json').read_text(encoding='utf-8'))
assert helper_binding == initial['helper']
assert binding(UNIT / 'protocol.json') == initial['protocol']
assert binding(BASELINE / 'receipt.json') == values['FROZEN_RECEIPT'] == initial['baselineReceipt']
assert len(values['FROZEN']) == 17
for relative, expected in values['FROZEN'].items():
    # The failed readback used ROOT here. The preserved baseline is separate.
    assert binding(BASELINE / relative) == expected, relative
assert binding(ROOT / values['MONITOR'])['sha256'] == values['MONITOR_SHA256']

original = json.loads((ORIGINAL_UNIT / 'protocol.json').read_text(encoding='utf-8'))
protocol = json.loads((UNIT / 'protocol.json').read_text(encoding='utf-8'))
guard_names = (
    'counts', 'modes', 'caseOrder', 'maximumChildWallSeconds', 'sampleSeconds',
    'minimumAvailableRamBytes', 'lowRamConsecutiveSamples',
    'maximumOwnedTreeRssBytes', 'preflightAvailableRamBytes',
    'minimumFreeDiskBytes', 'automaticRetry',
)
for name in guard_names:
    assert protocol[name] == original[name], name
assert protocol['automaticRetry'] is False
assert protocol['counts'] == [600, 18000]
assert protocol['caseOrder'] == ['600_old', '600_new', '18000_old', '18000_new']
assert protocol['frozenResultsSha256'] == values['FROZEN']['src/api/results.py']['sha256']
assert protocol['frozenReportingSha256'] == values['FROZEN']['src/api/reporting.py']['sha256']
assert protocol['frozenStorageSha256'] == values['FROZEN']['src/api/result_storage.py']['sha256']
assert protocol['frozenDocumentSha256'] == values['FROZEN']['src/api/result_document.py']['sha256']

old_source = (ORIGINAL_UNIT / 'helper.py').read_text(encoding='utf-8')
assert binding(ORIGINAL_UNIT / 'helper.py') == initial['originalHelper']
old_functions = function_nodes(ast.parse(old_source))
new_functions = function_nodes(tree)
for name in initial['preservedFixtureOracleGuardFunctionNames']:
    assert ast.dump(old_functions[name], include_attributes=False) == ast.dump(
        new_functions[name], include_attributes=False
    ), name

assert not (UNIT / 'run').exists(), 'A workload directory already exists'
assert not (UNIT / 'prepared_source_bindings.json').exists(), 'Prepared bindings already exist'
assert (UNIT / 'static_readback_failure_2026-10-06.py').is_file()
failure = json.loads((UNIT / 'static_readback_failure_2026-10-06.json').read_text(encoding='utf-8'))
assert failure['exitCode'] == 1 and failure['classification'].startswith('Preparation verifier')

# Replicate only the helper's reviewed source_bindings inventory from literals.
# No helper function or production code is imported or evaluated.
names = list(values['CURRENT']) + list(values['HISTORICAL'])
names += [(BASELINE / name).relative_to(ROOT).as_posix() for name in values['FROZEN']]
names += [(BASELINE / 'receipt.json').relative_to(ROOT).as_posix(),
          helper.relative_to(ROOT).as_posix(),
          (UNIT / 'protocol.json').relative_to(ROOT).as_posix(),
          (UNIT / 'PROTOCOL.md').relative_to(ROOT).as_posix()]
assert len(names) == len(set(names)), 'Duplicate inventory path'
bindings = {name: binding(ROOT / name) for name in names}
assert len(bindings) == 49
api_names = ('result_storage', 'results', 'result_document', 'reporting')
current_api = {name: bindings['src/api/' + name + '.py'] for name in api_names}
assert current_api['result_storage']['sha256'] == EXPECTED_STORAGE
assert current_api['results']['sha256'].startswith('01d8fb')
changed_api = [name for name in api_names if current_api[name] != values['FROZEN']['src/api/' + name + '.py']]
assert changed_api == ['result_storage', 'results'], changed_api

# A second read detects accidental drift during this static review.
assert bindings == {name: binding(ROOT / name) for name in names}, 'Source drift during static review'
receipt = {
    'reviewedAt': datetime.now(timezone.utc).isoformat(),
    'passed': True,
    'validationScope': 'Static AST/compile and exact hashes only; helper and production modules never imported or executed.',
    'astParseAndCompilePassed': True,
    'helper': helper_binding,
    'protocol': binding(UNIT / 'protocol.json'),
    'baselineReceipt': binding(BASELINE / 'receipt.json'),
    'frozenBaselineFilesVerified': 17,
    'inheritedFixtureOracleGuardFunctionASTsExact': True,
    'guardValuesExact': {name: protocol[name] for name in guard_names},
    'currentApi': current_api,
    'changedApiFromPreservedDiskBaseline': changed_api,
    'currentDependencyPaths': len(values['CURRENT']),
    'historicalEvidencePaths': len(values['HISTORICAL']),
    'preparedBindingFileCount': len(bindings),
    'sourceBindingsStableDuringReview': True,
    'failedReadbackRetained': failure,
    'rootFinalSourceEditAuthorizationReceived': True,
    'preparedBindingsWillBeWrittenExclusivelyAfterThisReceipt': True,
    'runExists': False,
    'helperImportedOrExecuted': False,
    'workloadsOrInferenceLaunched': False,
}
save_exclusive(UNIT / 'final_static_review_receipt.json', receipt)
save_exclusive(UNIT / 'prepared_source_bindings.json', bindings)
assert json.loads((UNIT / 'prepared_source_bindings.json').read_text(encoding='utf-8')) == bindings
assert bindings == {name: binding(ROOT / name) for name in names}, 'Source drift after exclusive binding write'
print(json.dumps({'staticReviewPassed': True, 'sourceBindingsFrozen': True,
                  'bindingFiles': len(bindings), 'helper': helper_binding,
                  'preparedBindingFile': binding(UNIT / 'prepared_source_bindings.json'),
                  'runExists': (UNIT / 'run').exists(),
                  'workloadsOrInferenceLaunched': False}, sort_keys=True))
