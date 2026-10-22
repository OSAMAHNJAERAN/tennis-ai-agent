"""Prepare a fresh guarded disk-to-disk comparison; never import or run it."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / 'artifacts/validation/product_execution/result_storage_measurement_2026-10-06'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06'
UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_measurement_2026-10-06'
ORIGINAL_HELPER_SHA = '8a1585543a904826fca354f5fa727debfd7b5071f330cae9cfe8c41eb0f446dc'


def binding(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open('x', encoding='utf-8', newline='') as stream:
        stream.write(value)


def save(path, value):
    write(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise AssertionError('Preparation replacement did not have exactly one match: ' + before[:80])
    return source.replace(before, after)


def main():
    original = (ORIGINAL / 'helper.py').read_text(encoding='utf-8')
    assert binding(ORIGINAL / 'helper.py')['sha256'] == ORIGINAL_HELPER_SHA
    baseline_receipt = json.loads((BASELINE / 'receipt.json').read_text(encoding='utf-8'))
    frozen = baseline_receipt['bindings']
    assert baseline_receipt['beforeLatencyChanges'] is True and len(frozen) == 17
    for name, expected in frozen.items():
        assert binding(BASELINE / name) == expected, 'Frozen baseline changed: ' + name
    UNIT.mkdir(exist_ok=False)
    source = replace_once(original,
        '"""One-shot synthetic result construction/publication/read/export comparison.',
        '"""One-shot synthetic preoptimization-disk/current-disk latency comparison.')
    source = replace_once(source,
        "BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_baseline_2026-10-06'",
        "BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06'")
    first, last = source.index('FROZEN = {'), source.index('CURRENT = (')
    source = (source[:first] + 'FROZEN = ' + repr(frozen) + '\n' +
              'FROZEN_RECEIPT = ' + repr(binding(BASELINE / 'receipt.json')) + '\n' + source[last:])
    source = replace_once(source,
        "    'src/api/results.py', 'src/api/result_storage.py', 'src/api/result_document.py',",
        "    'src/__init__.py', 'src/api/__init__.py',\n"
        "    'src/api/results.py', 'src/api/result_storage.py', 'src/api/result_document.py',")
    source = replace_once(source, 'COUNTS = (600, 18000)', '''HISTORICAL = (
    'artifacts/validation/product_execution/finalize_result_storage_evidence_2026-10-06.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/helper.py',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/protocol.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/run/report.json',
    'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit_correction.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/protocol.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/resource_receipt.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/child_receipt.json',
    'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/saved_result_comparison.json',
)
COUNTS = (600, 18000)''')
    source = replace_once(source,
        "    names = list(CURRENT) + [str((BASELINE / name).relative_to(ROOT).as_posix()) for name in FROZEN]",
        "    names = list(CURRENT) + list(HISTORICAL) + [str((BASELINE / name).relative_to(ROOT).as_posix()) for name in FROZEN]")
    source = replace_once(source, '''    for name, expected in FROZEN.items():
        if digest(BASELINE / name) != expected:
            raise AssertionError('Frozen baseline changed: ' + name)''', '''    receipt = BASELINE / 'receipt.json'
    if {'bytes': receipt.stat().st_size, 'sha256': digest(receipt)} != FROZEN_RECEIPT:
        raise AssertionError('Frozen baseline preservation receipt changed')
    for name, expected in FROZEN.items():
        path = BASELINE / name
        if {'bytes': path.stat().st_size, 'sha256': digest(path)} != expected:
            raise AssertionError('Frozen baseline changed: ' + name)''')
    source = replace_once(source, '\n\nclass Rows:', '''

def load_disk_implementation(mode):
    # Importing this package runs only its empty initializer, before production
    # dependencies. Frozen modules occupy their actual aliases so every imported
    # DiskSeries/ResultStorage/file_hash belongs to the selected implementation.
    import src.api as api_package
    names = ('result_storage', 'results', 'reporting', 'result_document')
    aliases = tuple('src.api.' + name for name in names)
    if any(name in sys.modules for name in aliases):
        raise AssertionError('Selected production aliases were imported before implementation selection')
    modules, bindings = {}, {}
    base = BASELINE if mode == 'old' else ROOT
    for name in names:
        path = base / 'src/api' / (name + '.py')
        module = load_file('src.api.' + name, path)
        setattr(api_package, name, module)
        modules[name] = module
        bindings[name] = {'moduleAlias': module.__name__,
                          'relativeSource': path.relative_to(ROOT).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)}
    storage, results = modules['result_storage'], modules['results']
    reporting, document = modules['reporting'], modules['result_document']
    assert results.DiskSeries is storage.DiskSeries
    assert reporting.DiskSeries is storage.DiskSeries
    assert document.ResultStorage is storage.ResultStorage
    assert document.file_hash is results.file_hash
    assert sys.modules['src.api.media_revision'].file_hash is results.file_hash
    return results, reporting, storage.ResultStorage, document.ResultDocument, document.publish_result, bindings


class Rows:''')
    source = replace_once(source, '''        if mode == 'old':
            results = load_file('frozen_results_20261006_measure', BASELINE / 'src/api/results.py')
            reporting = load_file('frozen_reporting_20261006_measure', BASELINE / 'src/api/reporting.py')
            media_revision = load_file('frozen_media_revision_20261006_measure', BASELINE / 'src/api/media_revision.py')
        else:
            from src.api import results, reporting
            from src.api.result_storage import ResultStorage
            from src.api.result_document import ResultDocument, publish_result''', '''        results, reporting, ResultStorage, ResultDocument, publish_result, loaded = load_disk_implementation(mode)
        report['implementation'] = 'FROZEN_PREOPTIMIZATION_DISK' if mode == 'old' else 'CURRENT_DISK_LATENCY_CANDIDATE'
        report['loadedProductionModules'] = loaded
        report['allSelectedStorageClassesConsistent'] = True''')
    source = replace_once(source,
        "        context = ResultStorage(analysis) if mode == 'new' else nullcontext()",
        '        context = ResultStorage(analysis)')
    source = replace_once(source,
        "            value = results.build_result(analysis, job, **({'_storage': storage} if mode == 'new' else {}))",
        '            value = results.build_result(analysis, job, _storage=storage)')
    source = replace_once(source, '''            if mode == 'old':
                # These three statements replay the preserved worker's exact
                # former strict JSON/replace/summary publication behavior.
                with (analysis / 'result.tmp').open('w', encoding='utf-8', newline='') as stream:
                    json.dump(value, stream, allow_nan=False)
                (analysis / 'result.tmp').replace(analysis / 'result.json')
                (analysis / 'summary.json').write_text(json.dumps(value['summary'], allow_nan=False), encoding='utf-8')
            else:
                publish_result(storage, value, analysis)''', '''            publish_result(storage, value, analysis)''')
    source = replace_once(source, '''        if mode == 'old':
            value = json.loads((analysis / 'result.json').read_text(encoding='utf-8'))
            document_context = nullcontext(value)
        else:
            document_context = ResultDocument(analysis, job['id'])''', '''        document_context = ResultDocument(analysis, job['id'])''')
    source = replace_once(source, '''        report['readerOpenScope'] = ('Complete JSON read_text/json.loads; no presentation decoration yet.' if mode == 'old' else
                                     'Production ResultDocument construction, including complete database SHA256 verification and root validation; no presentation decoration yet.')''', '''        report['readerOpenScope'] = 'Selected production ResultDocument construction, including database SHA256 verification and root validation; no presentation decoration yet. Shared file_hash cache follows the bound selected results.py implementation.' ''')
    source = replace_once(source, '''            value = (media_revision.presentation_urls(document, analysis, job['id']) if mode == 'old' else
                     document.presented(job['id']))''', '''            value = document.presented(job['id'])''')
    source = replace_once(source,
        "            chunks = json.JSONEncoder(allow_nan=False).iterencode(value) if mode == 'old' else document.json_chunks(value)",
        '            chunks = document.json_chunks(value)')
    source = replace_once(source, '''            # The frozen HTTP CSV path already streamed its rows, but retained
            # the entire saved result. Preserve that reference behavior.''', '''            # Both selected disk implementations stream the production CSV rows.''')
    source = replace_once(source, '''            html_chunks = (reporting.readable_report_chunks(value, None) if mode == 'new' else
                           iter((reporting.readable_report(value, None),)))''', '''            html_chunks = reporting.readable_report_chunks(value, None)''')
    source = replace_once(source, '''        if mode == 'new':
            marker = json.loads((analysis / '.result/current.store').read_text(encoding='utf-8'))
            database = analysis / '.result' / marker['database']
            assert marker['databaseSha256'] == digest(database) and marker['databaseBytes'] == database.stat().st_size
            assert marker['analysisId'] == job['id'] and marker['frames'] == count
            assert not (analysis / 'result.tmp').exists() and not (analysis / 'summary.tmp').exists()
            assert not tuple(analysis.glob('.result_storage_*.sqlite3*'))
            report['publicationManifest'] = marker
            report['publishedDatabaseBytes'] = database.stat().st_size
            report['ownedScratchAndPendingExportsAbsent'] = True''', '''        marker = json.loads((analysis / '.result/current.store').read_text(encoding='utf-8'))
        database = analysis / '.result' / marker['database']
        assert marker['databaseSha256'] == digest(database) and marker['databaseBytes'] == database.stat().st_size
        assert marker['analysisId'] == job['id'] and marker['frames'] == count
        assert not (analysis / 'result.tmp').exists() and not (analysis / 'summary.tmp').exists()
        assert not tuple(analysis.glob('.result_storage_*.sqlite3*'))
        assert not tuple((analysis / '.result').glob('*.store.tmp'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-journal'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-wal'))
        assert not tuple((analysis / '.result').glob('*.sqlite3-shm'))
        report['publicationManifest'] = marker
        report['publishedDatabaseBytes'] = database.stat().st_size
        report['ownedScratchAndPendingExportsAbsent'] = True''')
    source = replace_once(source,
        "              'currentDatabaseBytes': right.get('publishedDatabaseBytes'),",
        "              'baselineDatabaseBytes': left.get('publishedDatabaseBytes'),\n"
        "              'currentDatabaseBytes': right.get('publishedDatabaseBytes'),\n"
        "              'baselineImplementation': left['implementation'], 'currentImplementation': right['implementation'],\n"
        "              'noPerformanceQualificationGate': True,")
    # Keep all original fixture factories, semantic oracles, output comparisons,
    # ownership handshakes and monitoring code byte-for-byte at function level.
    old_tree, tree = ast.parse(original), ast.parse(source)
    old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    preserved = ('stamp', 'digest', 'save', 'publish_handshake', 'load_file', 'write_generated',
                 'camera_valid', 'selected', 'ball_record', 'motion_record', 'people_record',
                 'timing_valid', 'timing_intervals', 'generate_inputs', 'input_bindings',
                 'disk_snapshot', 'write_chunks', 'expected_location', 'expected_speed',
                 'check_contract', 'files_equal', 'parent', 'main')
    for name in preserved:
        assert ast.dump(functions[name], include_attributes=False) == ast.dump(old_functions[name], include_attributes=False), name
    compile(tree, str(UNIT / 'helper.py'), 'exec')  # Compilation is not execution.
    source = source.replace('from contextlib import nullcontext\n', '')
    write(UNIT / 'helper.py', source)
    protocol = json.loads((ORIGINAL / 'protocol.json').read_text(encoding='utf-8'))
    protocol['preparedDate'] = '2026-10-06'
    protocol['comparisonBaselineKind'] = 'FROZEN_PREOPTIMIZATION_DISK'
    protocol['frozenResultsSha256'] = frozen['src/api/results.py']['sha256']
    protocol['frozenReportingSha256'] = frozen['src/api/reporting.py']['sha256']
    protocol['frozenStorageSha256'] = frozen['src/api/result_storage.py']['sha256']
    protocol['frozenDocumentSha256'] = frozen['src/api/result_document.py']['sha256']
    protocol['frozenBaselineFileCount'] = 17
    protocol['baselinePath'] = ('Frozen preoptimization disk results/storage/document/reporting modules loaded in each old child '
        'under actual src.api.* aliases, storage first. Every imported DiskSeries/ResultStorage/file_hash class/function '
        'is asserted consistent. Both paths run build_result(_storage), publish_result, ResultDocument, presentation and streamed exports. '
        'No resident-list baseline replay or worker pipeline is run in this new experiment.')
    protocol['currentPath'] = ('Current four disk production modules under the same aliases and common bound dependencies. '
        'Candidate code changes are described by root final source review; preparation does not imply a latency improvement.')
    protocol['readTimingMeaning'] = ('Both paths separately time construction, publication, reader open, presentation, streamed JSON, '
        'CSV, HTML and independent checks. Reader-open follows its bound production hash validation/cache; each export times '
        'first chunk, all chunks written/closed and its separate full-output verification hash. Full export stages include that verification cost.')
    protocol['historicalEvidencePreserved'] = ('Earlier resident-list versus disk comparison, its initially failed timestamp auditor and '
        'corrected terminal audit, actual worker smoke, and the completed result-storage seal are retained and hash-bound. '
        'This is a separate preoptimization-disk versus current-disk comparison; prior receipts are not rewritten or rerun.')
    protocol['sourceFreezePendingAtPreparation'] = True
    protocol['sourceFreeze'] = ('Preparation intentionally does not create prepared_source_bindings.json. Root must finish all edits and '
        'tests, review the final source/helper/protocol bytes, then write the final bindings once. The parent refuses launch without '
        'that exact matching file. It freezes the binding record and all current common dependencies, all 17 preserved baseline '
        'files/receipt, prior seal/worker/synthetic evidence, helper, protocol and monitor through every child and completion.')
    protocol['latencyInterpretation'] = ('One old/new observation per count; report raw stage/child/monitor times separately. '
        'No variance estimate, throughput target or performance qualification gate is implied by exact-output success.')
    save(UNIT / 'protocol.json', protocol)
    write(UNIT / 'PROTOCOL.md', '''# Prepared disk-result latency comparison — not launched

This separate one-shot compares the sealed **preoptimization disk implementation**
with the current disk implementation at 600 and 18,000 generated frames. The old
resident-list versus disk experiment and its receipts remain historical evidence.
No worker, model, footage, FFmpeg or decoder is run here.

Both paths call their actual `build_result(_storage=storage)`, `publish_result`,
`ResultDocument`, presentation decoration and streamed JSON/CSV/HTML exports.
Old children load four frozen production modules under their actual `src.api.*`
aliases, storage first, before dependent production imports. Class/function identity
checks prevent mixing old and current storage. Common dependencies are final-bound
current code; the frozen 17-file baseline, preservation receipt and previous seal,
real worker smoke and resident/disk benchmark evidence are also bound.

Preparation does **not** create final `prepared_source_bindings.json` and does not
authorize execution. Root must finish edits/tests, review final code and fixtures,
then freeze that file. The launcher refuses missing or mismatched reviewed bindings.
The final candidate changes must be described from those frozen sources; this
protocol does not predeclare an improvement or allow automatic retry.

Inherited fixtures and all-frame semantic checks are unchanged. Synthetic index/60
timestamps and string PTS endpoints are explicitly generated, not decoded/native
camera timing. Null physical ball speed/xyz, disabled event authority, camera cuts,
timing/position gaps and unknown/null/empty people remain checked. Pose is a short
65-frame prefix, so this does not measure dense full-match pose.

Four clean children run sequentially: 600 old/new, then 18,000 old/new. Each has the
same predeclared 180-second wall guard, 1.25 GB aggregate owned-tree RSS guard, and
available RAM guard below 400 MiB for five consecutive 0.25-second samples.
Preflight requires 2.5 GB available RAM and 4 GB free disk before every child.
The child atomically publishes its exact PID/createTime identity and waits at most
10 seconds for sampled-identity acknowledgment before production imports or work.
The verified process monitor remembers exact owned identities and requires normal
or forced retirement before another child; only those owned processes are stopped.

Complete result JSON, reader JSON, summary JSON, CSV including CRLF and HTML must
match byte-for-byte, without runtime exclusions. Input hashes and independent
semantic checks must match. Closed generation manifest/hash, scratch and pending
export cleanup are verified for **both** disk paths. Any guard, source change,
child, retirement or output failure halts all later cases and preserves partials.

Child OS working-set and process-commit peaks include startup, imports, generated
inputs and verification. Sampled tree RSS can double-count shared pages and miss
short peaks; host RAM includes unrelated processes. Stage times, full child time
and parent-monitored wall time remain separate. Export stages include their
explicit full-file verification hash, with that cost separately recorded. Disk
figures are logical file bytes, not physical blocks or write amplification.

One observation per path/count does not establish stable performance, browser
capacity, representative long-video capacity or scientific accuracy. The browser
still holds full arrays. The 432,000 result case is outside this protocol.

After root review and final source freeze, launch from base Python313 with
`cmd.exe`, `login:false`, and the repository as working directory:

```bat
C:\\Users\\ac-98\\AppData\\Local\\Programs\\Python\\Python313\\python.exe artifacts\\validation\\product_execution\\result_storage_latency_measurement_2026-10-06\\helper.py
```

The launcher exclusively creates this unit's `run`. Existing output refuses reuse.
It preserves the exact executed helper, all input/identity/resource/child receipts,
samples, logs and exports. It never replaces old evidence or live product data.
''')
    save(UNIT / 'static_sanity_receipt.json', {
        'preparedAt': datetime.now(timezone.utc).isoformat(),
        'astParseAndCompilePassed': True, 'protocolParsed': True,
        'frozenBaselineFilesVerified': len(frozen),
        'preservedFixtureOracleGuardFunctionNames': preserved,
        'inheritedFunctionASTsExact': True,
        'originalHelper': binding(ORIGINAL / 'helper.py'),
        'helper': binding(UNIT / 'helper.py'), 'protocol': binding(UNIT / 'protocol.json'),
        'baselineReceipt': binding(BASELINE / 'receipt.json'),
        'finalPreparedBindingsWritten': False, 'sourceFreezePending': True,
        'runExists': (UNIT / 'run').exists(),
        'helperImportedOrExecuted': False, 'workloadsOrInferenceLaunched': False,
        'validationScope': 'Static AST/compile and exact inherited function comparisons only. No helper, fixture or child execution.'})
    print(json.dumps({'prepared': True, 'directory': str(UNIT), 'sourceFreezePending': True,
                      'helper': binding(UNIT / 'helper.py'), 'runExists': (UNIT / 'run').exists()}))


if __name__ == '__main__':
    main()
