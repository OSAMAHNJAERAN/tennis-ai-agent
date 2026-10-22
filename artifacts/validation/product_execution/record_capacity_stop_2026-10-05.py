"""Update the single progress record and restoration metadata with terminal evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


failure_path = HERE / 'repeated_source_capacity_STOP_verified_2026-10-05.json'
failure = json.loads(failure_path.read_text(encoding='utf-8'))
assert failure['evidenceVerified'] is True and failure['workloadPassed'] is False
assert digest(failure_path) == '4b13d66e23a936a9b3c1c3a9ea55555368fc8df6e755cf07eaf71ed36be7713e'
test_path = ROOT / 'tests/test_worker_result_publication.py'
test_receipt = HERE / 'worker_publication_capacity_2026-10-05.xml'
assert digest(test_path) == '5999b21601dfb7c200ac5599f9a0bbd19855e24d912f049d8b6aa415062febb5'
assert digest(test_receipt) == 'e70b98c5c6ff54324b55f73f2f3546de7c6f9aca7ca30d40657820c8b09a74b0'
suite = ET.parse(test_receipt).getroot().find('testsuite')
assert int(suite.get('tests')) == 10 and all(int(suite.get(key)) == 0 for key in ('failures', 'errors', 'skipped'))
manifest_path = ROOT / 'docs/deployment/PRODUCT_RESTORE_MANIFEST.json'
assert digest(manifest_path) == 'b435d04ce280bd12086979c52217a93f0abce08568fa4adf08793c8722d9cf6f'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
now = datetime.now(timezone.utc).isoformat()
manifest['validation']['repeatedSourceWorkerCapacity'] = (
    'FAILED_SYSTEM_RAM_SAFEGUARD_2026-10-05: one isolated 18000-frame, 300-second, 1920x1080, '
    '60 FPS repeated-source workload stopped during WASB detection after 1024.308 monitored seconds. '
    'Available system RAM was below 400000000 bytes for five consecutive samples; minimum 296972288 bytes, '
    'peak sampled owned-tree RSS 1854242816 bytes and private commit 4055429120 bytes. '
    'No completed result/annotation/CSV or child CUDA peak receipt was produced. '
    'Partial artifacts preserved, all 18000 synthetic N/60 normalization records checked, '
    'source/code/config hashes unchanged and live product states unchanged. '
    'Neither five-minute nor full-match processing capacity is qualified.'
)
manifest['validation']['workerPublicationFailures'] = (
    'PASSED_SOFTWARE_ONLY: 10 isolated inference-free tests cover exact successful JSON bytes and '
    'write, nonfinite encoder, close and replacement failures with new/prior files. '
    'Failed jobs retain prior bytes and cannot expose partial completed results through API/media/download routes.'
)
manifest['evidenceUpdatedAt'] = now
manifest['validationEvidence'].extend([
    {'path': str(failure_path.relative_to(ROOT)).replace('\\', '/'), 'verifiedAt': failure['verifiedAt'],
     'scope': 'Terminal failure audit: evidence verified, workload failed at predeclared available-system-RAM safeguard; no capacity or accuracy qualification.',
     'sha256': digest(failure_path), 'passed': False, 'evidenceVerified': True, 'historicalReceiptPreserved': True},
    {'path': str(test_receipt.relative_to(ROOT)).replace('\\', '/'), 'verifiedAt': now,
     'scope': 'Ten inference-free worker publication regressions only; production sources unchanged.',
     'sha256': digest(test_receipt), 'passed': True, 'historicalReceiptPreserved': True},
])
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

progress = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
lines = progress.read_text(encoding='utf-8').splitlines()
replacement = (
    '**The five-minute capacity job is terminal and failed; do not resume or automatically rerun it.** '
    '[The first preparation](../../artifacts/validation/product_execution/repeated_source_capacity_2026-10-05/resource_receipt.json) '
    'failed in FFmpeg before any inference or accepted job; its omitted stderr leaves that exact cause unresolved. '
    '[The revised protocol](../../artifacts/validation/product_execution/repeated_source_capacity_uniform_clock_2026-10-05/protocol.json) '
    'froze 18,000 frames, 300 seconds, 1920×1080, 60 FPS and 30 licensed repetitions with an explicit synthetic N/60 clock. '
    'Session **22734** exited **1**. Isolated analysis **match_63a83c719d1345ae96fa48646402595f** was marked FAILED at '
    '`2026-10-05T03:00:23.874560+00:00` during Step 4 WASB ball detection. '
    '[The parent receipt](../../artifacts/validation/product_execution/repeated_source_capacity_uniform_clock_2026-10-05/resource_receipt.json) '
    'records 1,024.308 monitored seconds, 509 samples, peak sampled owned-tree RSS **1,854,242,816 bytes**, '
    'private commit **4,055,429,120 bytes** and minimum available system RAM **296,972,288 bytes**. '
    'The available-RAM safeguard triggered after five consecutive samples below 400,000,000 bytes; '
    'the 4 GB owned-tree RSS and 10,800-second wall thresholds were not reached. '
    'Tree RSS can double-count shared pages and miss brief peaks; private commit is not resident RAM. '
    'System RAM pressure alone does not establish a worker leak or identify other host allocations. '
    'No child completion/CUDA peak receipt, completed inference result, annotation, report or CSV survived. '
    'All partial files, including the unpublished camera-writer temporary file, remain retained; parent logical footprint was **763,646,007 bytes**. '
    'Five-minute and representative full-match capacity remain unqualified.'
)
index = next(i for i, line in enumerate(lines) if line.startswith('**Resume the running capacity job'))
lines[index] = replacement
index = next(i for i, line in enumerate(lines) if line.startswith('Delegated edits stopped on the account usage limit.'))
lines[index] = (
    'Earlier delegated edits stopped on the account usage limit; root completed them. After the owner’s restoration reply, '
    'the focused read-only CSV/publication and stress-protocol reviews ran successfully. '
    '[Ten new publication regressions](../../artifacts/validation/product_execution/worker_publication_capacity_2026-10-05.xml) '
    'passed in 2.45 seconds: exact prior-serializer bytes on success; injected write, nonfinite encoder, close and replacement failures '
    'with/without a previous result; real failed-state persistence and blocked result/media/download exposure. '
    'FFmpeg, models and inference were stubbed; real filesystem and SQLite were used. '
    'No production source, model, label or selected configuration changed. '
    'The 974-test full backend receipt and 161-test frontend receipt remain the prior complete-suite checks; these ten are additional focused checks.'
)
index = next(i for i, line in enumerate(lines) if line.startswith('The [current clean-host check]'))
lines[index] = (
    'The [pre-stop clean-host check](../../artifacts/validation/product_installation/capacity_source_restoration_host_verification_2026-10-05.json) '
    'passed all 75 checks at `2026-10-05T02:51:19.238970+00:00`, binding then-current manifest '
    '`b435d04ce280bd12086979c52217a93f0abce08568fa4adf08793c8722d9cf6f`. '
    'Its isolated API had health/library 200, invalid-upload 415 and no inference/model imports. '
    '[The 02:52 live snapshot](../../artifacts/validation/product_execution/capacity_evidence_recheck_2026-10-05.json) is historical. '
    '[The terminal failure audit](../../artifacts/validation/product_execution/repeated_source_capacity_STOP_verified_2026-10-05.json) '
    'SHA256 `4b13d66e23a936a9b3c1c3a9ea55555368fc8df6e755cf07eaf71ed36be7713e` verifies the stop trigger, '
    'retired PIDs 12604/13880/18584, all 18,000 ordered unique prepared-source mapping records and exact rational normalized/source PTS N/60, '
    'unchanged selected sources/configuration/licensed input, correct credits and all ten live saved states unchanged. '
    'Twenty retained stress files are hash-bound. This verifies normalization of synthetic prepared timestamps, '
    'not original quantized source PTS or final media publication. The retained browser at 03:05:33 UTC still serves '
    '`index-BWMRwJe_.js`, paused at decoded/shared frame285, readyState4, source1280×720. '
    'Next capacity work requires measured memory headroom and bounded disk-backed records/result retrieval; '
    'a rerun must use a fresh prespecified directory and cannot reuse this failed evidence as a passing claim.'
)
progress.write_text('\n'.join(lines) + '\n', encoding='utf-8')

guide = ROOT / 'docs/deployment/PRODUCT_INSTALLATION.md'
text = guide.read_text(encoding='utf-8')
old = ('A monitored five-minute real-worker run of repeated licensed footage is active in isolated storage; its protocol, artificial boundaries and synthetic N/60 timestamps are recorded in [the progress record](../plans/TENNIS_EXECUTION_PROGRESS.md). Its full publication/resource checks are pending, and repetition cannot qualify representative match behavior or accuracy.')
new = ('The isolated five-minute real-worker stress **failed** during ball detection after 1,024.308 monitored seconds: available system RAM stayed below 400 MB for five samples (minimum297.0 MB); sampled owned-tree RSS peaked at1,854.2 MB and private commit at4,055.4 MB. No completed result/annotation/CSV or child CUDA peak receipt was produced. [The terminal audit](../../artifacts/validation/product_execution/repeated_source_capacity_STOP_verified_2026-10-05.json) verifies all18,000 prepared N/60 normalization records, unchanged sources/configuration and preserved live analyses. This failure does not establish five-minute/full-match capacity or a worker leak. Repetition and synthetic timing cannot qualify representative match behavior or accuracy; see [the progress record](../plans/TENNIS_EXECUTION_PROGRESS.md).')
assert old in text
guide.write_text(text.replace(old, new), encoding='utf-8')
print(json.dumps({'recordedAt': now, 'manifestSha256': digest(manifest_path), 'workloadPassed': False,
                  'publicationTestsPassed': 10, 'sourcesChanged': False}))
