"""Document terminal result-storage evidence without overwriting older receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
RECEIPT = Path(__file__).with_suffix('.json')
BASELINE = BASE / 'result_storage_baseline_2026-10-06'
assert not RECEIPT.exists()

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

suite = read(BASE / 'run_result_storage_backend_resumed_2026-10-06.json')
assert suite['passed'] and suite['sourceBindingsUnchanged'] and suite['counts']['tests'] == 1272
inventory = read(BASE / 'result_storage_backend_resumed_sources_2026-10-06.json')
assert all(digest(ROOT / name) == expected for name, expected in inventory.items())
measure = BASE / 'result_storage_measurement_2026-10-06'
benchmark = read(measure / 'run/report.json')
assert benchmark['passed'] and benchmark['sourcesUnchanged'] and benchmark['preparedBindingsUnchanged']
assert len(benchmark['comparisons']) == 2 and len(benchmark['runs']) == 4
audit = read(measure / 'terminal_audit_correction.json')
assert audit['passed']
old = read(measure / 'run/18000_old/child_receipt.json')
new = read(measure / 'run/18000_new/child_receipt.json')
old_resource = read(measure / 'run/18000_old/resource_receipt.json')
new_resource = read(measure / 'run/18000_new/resource_receipt.json')
smoke_dir = BASE / 'result_storage_worker_smoke_2026-10-06'
smoke = read(smoke_dir / 'resource_receipt.json')
child = read(smoke_dir / 'child_receipt.json')
assert smoke['passed'] and child['passed'] and child['isolatedValidationJobPublished']
assert smoke['allOwnedIdentitiesRetiredAfterNormalExit'] and smoke['productionJobStatesUnchanged']
assert not child['networkAttempts'] and not child['productionStatePublished']
comparison = read(smoke_dir / 'saved_result_comparison.json')
assert comparison['passed']
documents = ['docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
             'docs/deployment/PRODUCT_RESTORE_MANIFEST.json']
preserved = read(BASELINE / 'receipt.json')['bindings']
before = {}
for name in documents:
    assert digest(ROOT / name) == preserved[name]['sha256']
    assert digest(BASELINE / name) == preserved[name]['sha256']
    before[name] = digest(ROOT / name)

scope = ('Disk-backed result normalization and publication, owned immutable generation readers, '
         'and streamed JSON/CSV/HTML. Pipeline player/trajectory arrays and browser result arrays remain O(N). '
         'No new independent accuracy, physical speed/3D, full-match capacity or deployment qualification.')
memory = f"{old['windowsPeakWorkingSetBytes']:,} → {new['windowsPeakWorkingSetBytes']:,} bytes"
elapsed = f"{old_resource['monitoredChildWallSeconds']:.3f} → {new_resource['monitoredChildWallSeconds']:.3f} seconds"
stages = f"{old['measuredProductionStagesSeconds']:.3f} → {new['measuredProductionStagesSeconds']:.3f} seconds"
paragraph = f'''Latest result-storage unit (6 October): the production worker now parses complete raw JSON into owned disk-backed series and constructs the normalized result without full-video duplicate arrays. Small vectors stay resident. Missing samples, state/confidence, optional-file all-or-nothing failure, duplicate-key/EOF rules, null physical ball speed/3D, event authority and exact default JSON bytes are preserved. Speed-window checks use running gap indices. Publication closes strict result/summary exports and compacts a reachable result database, then commits a generation manifest before job completion. Caught later rename failures restore previous compatibility exports; abrupt interruption remains protected by the durable job-status gate. Historical JSON is read through EOF into owned temporary storage, without migration. JSON/CSV/HTML responses close readers after normal completion, send/iterator failure or disconnect, preserving primary errors if cleanup also fails. [All 1,272 backend tests](../../artifacts/validation/product_execution/result_storage_backend_RESUMED_2026-10-06.xml) passed in {suite['testElapsedSeconds']:.3f} seconds with source bindings unchanged. The first full run disappeared without a terminal receipt; its artifacts and [interruption readback](../../artifacts/validation/product_execution/result_storage_backend_interrupted_2026-10-06.json) remain incomplete evidence.

[The one-shot synthetic comparison](../../artifacts/validation/product_execution/result_storage_measurement_2026-10-06/run/report.json) passed four fresh children at 600/18,000 frames with identical complete result/read JSON, summary, CSV and HTML bytes and independent frame/gap/UNKNOWN-person/null/clock checks. At 18,000 frames actual workload-interpreter OS peak working set changed **{memory}**, while full child wall time changed **{elapsed}**; measured construction/publication/reader/presentation/export stages changed **{stages}**. The new closed database occupies {new['publishedDatabaseBytes']:,} logical bytes, in addition to compatibility JSON and raw artifacts. Sampled tree peak RSS was {old_resource['peakSampledOwnedTreeRssBytes']:,} → {new_resource['peakSampledOwnedTreeRssBytes']:,} bytes; actual process peak commit was {old['windowsPeakPrivateCommitBytes']:,} → {new['windowsPeakPrivateCommitBytes']:,} bytes. These are separate OS/sample metrics, not total host RAM or GPU memory. [Terminal audit](../../artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit_correction.json) verifies exact sources, exports and retirement. Its initial audit used an invalid timestamp-order assertion; that failed receipt remains preserved. The memory saving has a substantial elapsed-time cost. Dense full-match pose, 432,000 result construction, variance, real long-video capacity and browser arrays were not measured; synthetic index/60 timestamps are not decoded footage timing evidence.

[The fresh real worker smoke](../../artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/resource_receipt.json) ran the actual models, normalization, full-decode gates, disk-backed builder, publication and lazy reopening in one isolated JobStore. The licensed original source frames 390–419 retain exact input pixels and native 60-FPS relative PTS; normalized lossy H.264 pixels are independently decoded, not equated with the input. Complete JSON/CSV/HTML bytes match the frozen normalizer/reporting on the same new raw artifacts, with no runtime-field exclusions. Worker elapsed {child['actualWorkerElapsedSeconds']:.3f}s; full monitored child {smoke['monitoredChildWallSeconds']:.3f}s. Actual child OS peak working set {child['windowsPeakWorkingSetBytes']:,} bytes; sampled tree RSS {smoke['peakSampledOwnedTreeRssBytes']:,} bytes. Court/selected-role measurements remain unavailable, physical ball speed/3D null and event authority off. Zero network attempts, closed generation integrity, owned scratch cleanup, exact process retirement and all ten unchanged live analysis rows were verified. This is a half-second software check. The preparation observer is disabled before model inference; worker timing includes preparation/model-construction observation overhead and is not a performance comparison.

Current remaining capacity work: pipeline player/trajectory storage, API frame retrieval and browser loading, plus the measured result-path elapsed-time cost. The failed five-minute workload and failed large registration benchmark remain retained and unqualified. No frontend, model asset, selected configuration, scientific label or live saved result was changed in this unit. Preserve each analysis's complete hidden `.result` directory together with `result.json`, `summary.json` and the product queue; a missing/corrupt committed manifest or database is unavailable, never silently replaced with compatibility JSON.

'''
progress = ROOT / documents[0]
text = progress.read_text(encoding='utf-8')
anchor = 'Latest registration-storage unit (6 October):'
assert text.count(anchor) == 1
progress.write_text(text.replace(anchor, paragraph + 'Historical registration-storage unit (6 October):', 1), encoding='utf-8')
installation = ROOT / documents[1]
text = installation.read_text(encoding='utf-8')
anchor = '## Evidence and remaining checks\n\n'
assert text.count(anchor) == 1
instructions = f'''The result-storage update passes all 1,272 backend checks, paired 600/18,000-frame synthetic construction/publication/read/export comparisons and one actual isolated 30-frame worker smoke. All complete JSON/CSV/HTML bytes retain the frozen contract. At 18,000 synthetic frames actual interpreter OS peak working set changed {memory}; child wall time changed {elapsed}. The saving carries a substantial speed and disk cost. See [current measured progress](../plans/TENNIS_EXECUTION_PROGRESS.md) and the [terminal benchmark audit](../../artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit_correction.json) for scope and process retirement.

Back up the entire hidden `.result` directory in every new analysis, including `current.store`, its named immutable database and previous generation files, alongside compatibility result/summary JSON. The committed manifest is authoritative. Do not reconstruct, delete or silently fall back from a damaged new generation. Historical JSON analyses remain readable without migration; owned scratch is temporary reader storage. Job completion follows publication, and incomplete jobs remain unavailable. No extra runtime package is required; this uses Python's existing SQLite. Pipeline player/trajectory arrays, browser result arrays, representative long-video capacity and independent scientific accuracy remain unresolved.

The following registration/candidate receipts describe earlier revisions and remain historical after this result-storage update.

'''
installation.write_text(text.replace(anchor, anchor + instructions, 1), encoding='utf-8')
manifest_path = ROOT / documents[2]
manifest = read(manifest_path)
manifest['evidenceUpdatedAt'] = datetime.now(timezone.utc).isoformat()
manifest['validation']['backendResultStorageChecks'] = f"PASSED: all1272 backend tests in{suite['testElapsedSeconds']:.3f}s; exact source bindings unchanged. Prior incomplete full run retained."
manifest['validation']['resultStorageEngineering'] = (f"PASSED_SYNTHETIC_SOFTWARE_ONLY_2026-10-06: four fresh600/18000frame children; all result/readJSON,summary,CSV,HTML bytes exact. At18000 actualinterpreter OSpeakworking set {memory}; fullchildwall {elapsed}; productionstage total {stages}; database {new['publishedDatabaseBytes']}bytes. All exactownedprocesses retired. Memory improvement has substantial time/disk cost; no432000/densefullmatchpose/browser/inferencecapacity qualification.")
manifest['validation']['resultStorageRealWorkerSmoke'] = (f"PASSED_30_FRAME_SOFTWARE_ONLY_2026-10-06: actual offline selectedmodels+worker+committedresult+lazyreader on licensednative60FPS originalframes390-419; inputpixels/PTSexact, normalizedlossypixelsdecoded separately. JSON/CSV/HTML exactagainst frozen code on samenewrawartifacts, no exclusions. Worker {child['actualWorkerElapsedSeconds']:.3f}s/fullchild {smoke['monitoredChildWallSeconds']:.3f}s; tenlivejobsunchanged,zero network, allownedretired. Court/player measurements unavailable,speed3D null,authorityoff; no accuracy/fullmatch/deployment qualification.")
manifest['validation']['workerPublicationFailures'] = 'PASSED_SOFTWARE_ONLY: full1272-test suite includes10 original worker tests and31 reader/publication/export regressions. Strict writes, close/encoder and result/summary/manifest replacement failures preserve completed generation gates and prior compatibility bytes; interrupted streams close owned readers and retain primary errors.'
manifest['persistentState']['preserve'].append('Complete hidden .result directories: current.store, its named immutable SQLite result database, prior generation files and retained failure diagnostics; never filter to JSON/video only')
manifest['persistentState']['backupPolicy'] += ' Preserve the coherent committed result generation with every complete analysis directory; a missing or changed new-generation manifest/database is unavailable.'
for name, label in ((BASE / 'run_result_storage_backend_resumed_2026-10-06.json', 'Backend regression receipt;1272passed; exact sources.'),
                    (measure / 'terminal_audit_correction.json', 'Corrected synthetic result storage/export terminal audit; initial timestamp-assertion error preserved. No inference/accuracy/fullmatch qualification.'),
                    (smoke_dir / 'resource_receipt.json', 'Actual30-frame isolated worker compatibility; no long-video/accuracy qualification.')):
    manifest['validationEvidence'].append({'path': name.relative_to(ROOT).as_posix(), 'verifiedAt': datetime.now(timezone.utc).isoformat(),
                                          'scope': label, 'passed': True, 'sha256': digest(name)})
with manifest_path.open('w', encoding='utf-8') as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
    stream.write('\n')
bindings = [*documents, 'artifacts/validation/product_execution/run_result_storage_backend_resumed_2026-10-06.json',
            'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit.json',
            'artifacts/validation/product_execution/result_storage_measurement_2026-10-06/terminal_audit_correction.json',
            'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/resource_receipt.json',
            'artifacts/validation/product_execution/result_storage_worker_smoke_2026-10-06/saved_result_comparison.json',
            Path(__file__).relative_to(ROOT).as_posix()]
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True, 'scope': scope,
               'previousDocumentHashes': before, 'previousExactCopiesPreserved': True,
               'backendTests': 1272, 'benchmarkPassed': True, 'realWorkerSmokePassed': True,
               'productionQualified': False, 'fullMatchQualified': False, 'accuracyQualified': False,
               'bindings': {name: digest(ROOT / name) for name in bindings}}, stream, indent=2)
print(json.dumps({'documented': True, 'backendTests': 1272, 'scope': scope}))
