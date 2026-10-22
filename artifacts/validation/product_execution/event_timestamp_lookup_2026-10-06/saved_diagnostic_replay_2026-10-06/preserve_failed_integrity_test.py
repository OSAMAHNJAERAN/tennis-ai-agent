"""Preserve the exact failed historical provenance test before its correction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
unit = Path(__file__).resolve().parent
root = unit.parents[4]
source = root / 'tests/test_phase6_4_recovery_artifacts.py'
target = unit / 'preserved/tests/test_phase6_4_recovery_artifacts.py'
report = root / 'artifacts/validation/phase6_4_event_recovery_results.json'
baseline = unit.parent / 'src/events/event_detector.py'
event_source = root / 'src/events/event_detector.py'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert 'test_recovery_input_hashes_match_current_sources_and_exclude_pristine_media' in source.read_text(encoding='utf-8')
assert digest(event_source) == 'ccc00b450aa144d59a5fff13ede8a675749dc62fdc500143d3c114af715eff63'
assert digest(baseline) == '078cd52f5ff55a7256f7b7d5e2506352dbd59df4d328ca1bd6f2f6ee73a79552'
target.parent.mkdir(parents=True, exist_ok=False)
with source.open('rb') as incoming, target.open('xb') as outgoing:
    outgoing.write(incoming.read())
assert digest(target) == digest(source)
record = {'preservedAt': datetime.now(timezone.utc).isoformat(),
    'test':source.relative_to(root).as_posix(), 'testSha256':digest(source),
    'preservedTest':target.relative_to(root).as_posix(), 'preservedTestSha256':digest(target),
    'historicalReport':report.relative_to(root).as_posix(),'historicalReportSha256':digest(report),
    'historicalEventDetector':baseline.relative_to(root).as_posix(),
    'historicalEventDetectorSha256':digest(baseline),'currentEventDetectorSha256':digest(event_source),
    'failedBackendReceipt':'artifacts/validation/product_execution/event_timestamp_lookup_2026-10-06/backend/receipt.json',
    'failedBackendReceiptSha256':digest(unit.parent / 'backend/receipt.json'),
    'sourceEdited':False,'testEdited':False,'historicalArtifactsEdited':False,'replayRun':False,
    'interpretation':'Historical diagnostic report binds the original detector, not every future compatible production revision. Preserve both provenance and the failed suite.'}
with (unit / 'preserved_integrity_test.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(record,stream,indent=2,allow_nan=False)
print(json.dumps(record),flush=True)

