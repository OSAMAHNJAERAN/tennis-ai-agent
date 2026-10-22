"""One-shot evidence-aware documentation update for candidate storage."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def suite_count(path):
    suites = ET.parse(path).getroot().findall('testsuite')
    assert suites
    assert all(int(suite.get(key, '0')) == 0 for suite in suites for key in ('errors', 'failures', 'skipped'))
    return sum(int(suite.get('tests')) for suite in suites)


benchmark_dir = HERE / 'candidate_storage_measurement_monitored_2026-10-05'
protocol = read(benchmark_dir / 'protocol.json')
benchmark = read(benchmark_dir / 'report.json')
audit = read(benchmark_dir / 'terminal_receipt_audit.json')
assert benchmark['passed'] and benchmark['sourceFilesUnchanged'] and len(benchmark['runs']) == 8
assert audit['passed'] and audit['allEightActualWorkloadsSampledByExactIdentity']
assert audit['allRememberedOwnedIdentitiesRetiredAtTerminalAudit']
assert digest(benchmark_dir / 'executed_measure_candidate_storage.py') == protocol['sourceFiles']['scripts/audit/measure_candidate_storage.py']
for name, expected in protocol['sourceFiles'].items():
    if name != 'scripts/audit/measure_candidate_storage.py':
        assert digest(ROOT / name) == expected, name
comparison = read(benchmark_dir / 'tracker_5000_comparison.json')
assert comparison['passed'] and comparison['exactOutputBytesAndParsedValues']
assert [row['peakLive'] for row in comparison['candidateLifetimes']] == [13682, 16]
old, new = [next(row for row in audit['runs'] if row['kind'] == 'storage' and row['mode'] == mode and row['frames'] == 432000)
            for mode in ('old', 'new')]
assert old['windowsPeakWorkingSetBytes'] == 533958656 and new['windowsPeakWorkingSetBytes'] == 76079104
assert new['databaseCleanup']['logicalDatabaseBytesBeforeCleanup'] == 245608448
assert all(row['databaseCleanup'] is None or row['databaseCleanup']['allExactOwnedFilesAbsent'] for row in audit['runs'])
hardening_dir = HERE / 'candidate_monitor_hardening_2026-10-05'
guard = read(hardening_dir / 'redirector_monitor_regression/receipt.json')
assert guard['passed'] and not guard['remainingOwnedIdentities']
assert guard['sourceSha256'] == digest(ROOT / 'scripts/audit/measure_candidate_storage.py')
assert suite_count(hardening_dir / 'unit_regressions.xml') == 4
backend_path = HERE / 'candidate_storage_backend_FINAL_2026-10-05.xml'
assert suite_count(backend_path) == 1063
assert suite_count(HERE / 'candidate_producer_cleanup_GREEN_2026-10-05.xml') == 24
smoke_dir = HERE / 'candidate_storage_real_smoke_2026-10-05'
smoke = read(smoke_dir / 'real_pipeline_smoke.json')
resource = read(smoke_dir / 'resource_receipt.json')
artifact_compare = read(smoke_dir / 'artifact_comparison.json')
assert smoke['passed'] and resource['passed'] and artifact_compare['passed']
assert resource['status'] == 'COMPLETED' and resource['allOwnedIdentitiesRetiredAfterNormalExit']
assert resource['actualWorkloadWasSampled'] and not resource['remainingOwnedIdentitiesAfterNormalExit']
assert all(resource[name] for name in ('sourceBindingsUnchanged', 'historicalArtifactsUnchanged', 'productionJobStatesUnchanged'))
assert len(resource['productionJobStatesAfter']['jobs']) == 10
assert len(artifact_compare['jsonFiles']) == 26 and all(row['exactParsedValuesEqual'] for row in artifact_compare['jsonFiles'].values())
assert artifact_compare['volatilePaths'] == {'metrics.json': ['$.processing_fps', '$.total_time_s']}
assert artifact_compare['runConfigBytesEqual'] and not artifact_compare['unpublishedScratchFiles']
assert artifact_compare['annotatedDecodedPixelsEqual']
assert artifact_compare['annotatedVideos']['old']['sha256'] == artifact_compare['annotatedVideos']['new']['sha256']
assert not smoke['networkAttempts'] and not smoke['productionQualified'] and not smoke['productStatePublished']
for name, binding in resource['sourceBindingsAfter'].items():
    assert digest(ROOT / name) == binding['sha256'], name

now = datetime.now(timezone.utc).isoformat()
manifest_path = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
baseline = HERE / 'candidate_storage_document_baseline_2026-10-05'
document_baseline = read(baseline / 'receipt.json')
for name, expected in document_baseline['bindings'].items():
    assert digest(ROOT / name) == digest(baseline / name) == expected, name
manifest = read(manifest_path)
manifest['validation']['candidateStorageEngineering'] = (
    'PASSED_SYNTHETIC_ONLY_2026-10-05: disk-backed candidate storage and early streamed person export preserve frame values/order. '
    'Eight sequential clean-environment children at 600/18000/432000 candidate frames and 5000 tracker frames passed exact comparisons. '
    'At 432000 frames old/new actual interpreter peak working set was 533958656/76079104 bytes; '
    f'child elapsed seconds {old["totalChildElapsedSeconds"]:.3f}/{new["totalChildElapsedSeconds"]:.3f}, owned database logical size 245608448 bytes before cleanup. '
    'Every actual interpreter identity was sampled and all remembered identities were retired at terminal audit. '
    'Tracker output preserved every field; peak live candidates changed 13682 to 16. '
    'The first wrapper-only monitoring run remains historical with an explicit guard limitation; corrected and hardened monitor proofs are separate. '
    'Full calibration, player, trajectory, result/API and frontend arrays remain O(N); no full-match or accuracy qualification.'
)
manifest['validation']['modelLoadAndInferenceInCleanEnvironment'] = (
    'PASSED_SHORT_SOFTWARE_COMPATIBILITY_2026-10-05: fresh offline selected pipeline on 30 licensed original frames at 1920x1080/60 FPS; '
    '26 JSON artifacts retain exact parsed values except metrics.processing_fps and metrics.total_time_s; run_config bytes and annotated video bytes match the historical run. '
    f'Helper elapsed {smoke["elapsedSeconds"]:.3f}s, pipeline {smoke["metrics"]["total_time_s"]:.3f}s, '
    f'full verification child {resource["monitoredChildWallSeconds"]:.3f}s. '
    'No network attempts or product-state publication; source/history hashes and all10 stored analysis rows unchanged. '
    'Court calibration rejected, selected roles/meaningful pose/racket absent, physical speed/3D and event authority withheld. '
    'This half-second software compatibility check does not qualify accuracy, full-match capacity or deployment.'
)
manifest['validation']['backendCandidateStorageChecks'] = 'PASSED: all1063 backend tests after production candidate/cleanup changes, plus4 later owned-process teardown tests; no frontend source changes.'
manifest['evidenceUpdatedAt'] = now
evidence = [
    (backend_path, 'All1063 maintained backend tests after production candidate and producer-cleanup changes.'),
    (benchmark_dir / 'report.json', 'Eight synthetic candidate/tracker children with exact output equivalence; bounded scope.'),
    (benchmark_dir / 'terminal_receipt_audit.json', 'Actual interpreter sampling, exact remembered identity retirement and owned scratch cleanup; terminal audit only.'),
    (hardening_dir / 'unit_regressions.xml', 'Four isolated teardown regressions after audit helper hardening; no inference.'),
    (hardening_dir / 'redirector_monitor_regression/receipt.json', 'Controlled64MiB real child verifies actual descendant RSS stop and exact retirement with hardened helper.'),
    (smoke_dir / 'real_pipeline_smoke.json', 'Fresh half-second offline clean-runtime pipeline execution; no accuracy/capacity qualification.'),
    (smoke_dir / 'artifact_comparison.json', 'Exact26JSON compatibility with only2 measured runtime exclusions; unchanged run_config and annotated media.'),
    (smoke_dir / 'resource_receipt.json', 'Full owned tree sampled; source/history/product rows unchanged, all owned processes retired.'),
]
for path, scope in evidence:
    value = read(path) if path.suffix == '.json' else {}
    manifest['validationEvidence'].append({'path': path.relative_to(ROOT).as_posix(), 'verifiedAt': value.get('completedAt', now),
                                           'scope': scope, 'sha256': digest(path), 'passed': True, 'historicalReceiptPreserved': True})
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

addition = (
    'Latest candidate-storage unit: completed software checks on 5 October. The pipeline writes candidate frames to an exclusively owned SQLite scratch database '
    '(2MiB page cache, transactions of256 records, no mmap/full candidate index), retaining exact floats, boxes, gaps and ordering. '
    'Existing passes read snapshots and persist explicit indexed replacements. Person records are exported immediately after unchanged whole-segment participant selection, '
    'so unselected raw boxes are released before ball inference; exact BBox identity and role assignment are preserved. Temporal bounds filtering reads only requested frames. '
    'Storage failure closes the original detector generator even through tqdm, preserves the primary error and retains the closed diagnostic database. '
    '[Injected cleanup RED](../../artifacts/validation/product_execution/candidate_producer_cleanup_RED_2026-10-05.xml) failed both cleanup cases, '
    '[focused GREEN](../../artifacts/validation/product_execution/candidate_producer_cleanup_GREEN_2026-10-05.xml) passed24 checks, '
    'and [the complete backend suite](../../artifacts/validation/product_execution/candidate_storage_backend_FINAL_2026-10-05.xml) passed1063 tests in31.38s. '
    'Four subsequent [audit-guard teardown tests](../../artifacts/validation/product_execution/candidate_monitor_hardening_2026-10-05/unit_regressions.xml) passed separately. '
    'The first candidate utility test failures were test-owned open SQLite handles on Windows, not a proven implementation regression; that receipt is retained.\n\n'
    '[Corrected paired measurement](../../artifacts/validation/product_execution/candidate_storage_measurement_monitored_2026-10-05/report.json) completed eight sequential clean children. '
    '600/18,000/432,000 candidate-frame comparisons preserve every forward byte/parsed value, full reverse digest, neighbor and slice reads. '
    f'At432,000 frames actual interpreter peak working set was533,958,656 →76,079,104 bytes; elapsed18.339 →32.803s; '
    'the new scratch database required245,608,448 logical bytes before exact owned cleanup. Old/new process commit peaks were1,288,814,592/829,706,240 bytes, including imports; commit is not resident RAM. '
    'The real frozen/current5000-frame Kalman trackers preserve all11 TemporalBallPoint fields with identical1,367,179-byte output, '
    'while peak live BallObservation snapshots fell13,682 →16. Additional decoding increases work. These are synthetic candidate-only measurements, without models or representative match behavior.\n\n'
    '[The terminal audit](../../artifacts/validation/product_execution/candidate_storage_measurement_monitored_2026-10-05/terminal_receipt_audit.json) '
    'checks all8 actual interpreter identities sampled, exact aggregate resource sums and every remembered process retired. Windows clean-env Python uses a redirector; '
    '[the first benchmark](../../artifacts/validation/product_execution/candidate_storage_measurement_2026-10-05/terminal_monitor_audit.json) '
    'watched only that wrapper, so its output/child OS peak evidence is valid but its owned-RSS safeguard is unverified. '
    'Both exact executed benchmark scripts are preserved. The corrected run still lacked retirement checks at each child transition and a fallback after descendant teardown errors; '
    'post-run hardening adds those checks, preserves cleanup errors and handles process-exit races. '
    '[A controlled64MiB proof](../../artifacts/validation/product_execution/candidate_monitor_hardening_2026-10-05/redirector_monitor_regression/receipt.json) '
    'sampled the actual85.96MB child, stopped only remembered descendants deepest first and retired the wrapper; no third full benchmark ran.\n\n'
    '[Fresh real-pipeline compatibility](../../artifacts/validation/product_execution/candidate_storage_real_smoke_2026-10-05/resource_receipt.json) '
    'completed with the selected assets unchanged and zero network attempts.30 original source frames390–419, their retained/excerpt pixels, exact relative PTS,1920×1080 and native60FPS were verified again. '
    '[All26 JSON artifacts](../../artifacts/validation/product_execution/candidate_storage_real_smoke_2026-10-05/artifact_comparison.json) '
    'preserve exact parsed values except two declared runtime fields; run_config and annotated MP4 bytes match the historical run, all30 annotated frames decode with exact normalized index/60 PTS. '
    f'Pipeline10.895s at2.754FPS, helper26.760s, full monitored verification57.775s; actual child OS peak working set1,827,860,480 bytes, '
    'sampled owned-tree peak1,830,543,360 bytes, CUDA allocator486,753,792 allocated/578,813,952 reserved bytes. '
    'All candidate scratch files are absent, all owned processes retired, historic files unchanged and all10 production analysis rows unchanged. '
    'The court is rejected and roles/meaningful pose/racket remain unavailable; physical speed/3D and authority remain withheld. This is half-second software evidence, not accuracy or capacity qualification.\n\n'
    'The previous1063 backend run predates only the4 audit-helper tests, with no later production edits. The unchanged frontend161-test/restored build evidence remains applicable. '
    'The single progress record and installation manifest are updated, while their exact preceding versions and the three preunit production sources are preserved. '
    'Delegated final reviews hit an account usage limit; root commands continued to work and completed the remaining checks without bypass. '
    'Next meaningful capacity work is disk-backed calibration/player/trajectory results and bounded API/frontend frame retrieval, followed by a fresh prespecified real-worker workload with sufficient headroom. '
    'The failed18000-frame workload remains failed; no new full-match claim or accuracy promotion follows from this unit.'
)
progress = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
text = progress.read_text(encoding='utf-8')
anchor = 'Latest verification and resume notes:\n\n'
assert text.count(anchor) == 1
progress.write_text(text.replace(anchor, anchor + addition + '\n\n', 1), encoding='utf-8')

guide = ROOT / 'docs/deployment/PRODUCT_INSTALLATION.md'
text = guide.read_text(encoding='utf-8')
anchor = 'The final October5 [capacity-stop host receipt]'
index = text.index(anchor)
guide_addition = (
    'The [candidate-storage update](../../artifacts/validation/product_execution/candidate_storage_measurement_monitored_2026-10-05/report.json) '
    'passed all1063 backend tests plus4 later process-guard regressions. Candidate storage now uses owned bounded-cache SQLite, '
    'with early person export and lazy temporal bounds filtering. Synthetic432,000-frame candidate peak working set changed533.96 →76.08MB, '
    'elapsed18.339 →32.803s and scratch database245.61MB before cleanup; these are candidate-only costs. '
    '[Fresh30-frame clean-runtime compatibility](../../artifacts/validation/product_execution/candidate_storage_real_smoke_2026-10-05/resource_receipt.json) '
    'preserves26JSON artifact values (excluding only two measured runtime fields), exact run_config and annotated media bytes, original input pixels/relativePTS and all10 saved analyses. '
    'Actual interpreter/descendant monitoring and retirement are verified. Court rejection and unavailable scientific sections persist. '
    'Selected configuration/assets and frontend source/build remain unchanged; no new dependency is required. '
    'The preceding capacity-stop host/manifest seal below is historical after this evidence metadata update. '
    'Full calibration/player/trajectory/result/API/frontend arrays remain O(N); the prior five-minute failure and unqualified full-match ceiling still apply.\n\n'
)
guide.write_text(text[:index] + guide_addition + text[index:], encoding='utf-8')
receipt = {'recordedAt': now, 'softwareChecksPassed': True, 'productionQualified': False, 'fullMatchQualified': False,
           'manifestSha256': digest(manifest_path), 'backendTestsPassed': 1063, 'additionalAuditTestsPassed': 4,
           'syntheticCandidatePeakWorkingSetBytes': [old['windowsPeakWorkingSetBytes'], new['windowsPeakWorkingSetBytes']],
           'syntheticCandidateElapsedSeconds': [old['totalChildElapsedSeconds'], new['totalChildElapsedSeconds']],
           'real30FrameCompatibilityPassed': True, 'priorFiveMinuteWorkloadPassed': False,
           'documents': {path.relative_to(ROOT).as_posix(): digest(path) for path in (manifest_path, progress, guide)},
           'evidence': {path.relative_to(ROOT).as_posix(): digest(path) for path, _ in evidence}}
with (HERE / 'candidate_storage_documentation_2026-10-05.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2, allow_nan=False)
print(json.dumps({'recordedAt': now, 'backendTestsPassed': 1063, 'additionalAuditTestsPassed': 4,
                  'manifestSha256': receipt['manifestSha256'], 'real30FrameCompatibilityPassed': True}))
