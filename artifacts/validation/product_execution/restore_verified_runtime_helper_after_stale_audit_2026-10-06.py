"""Preserve a stale audit edit and restore only its exact previously sealed bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
UNIT = BASE / 'result_storage_latency_stale_audit_restore_2026-10-06'
SEAL = BASE / 'finalize_result_storage_latency_evidence_2026-10-06.json'
TARGET = BASE / 'verify_result_storage_latency_runtime_2026-10-06.py'
UNIT.mkdir()

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def path_for(name):
    return Path(name) if Path(name).is_absolute() else ROOT / name

report = {'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False,
          'scope': 'Restore an accidentally changed audit helper to the exact previously executed and sealed bytes. No production source, tests, documentation, historical receipts, server or jobs are changed.',
          'inferenceRun': False, 'historicalReceiptsRewritten': False}
try:
    seal = json.loads(SEAL.read_text(encoding='utf-8'))
    name = TARGET.relative_to(ROOT).as_posix()
    expected = seal['bindings'][name]
    data = TARGET.read_bytes()
    (UNIT / 'preserved_agent_edit.py').write_bytes(data)
    report.update({'sealedHelperSha256': expected, 'editedHelperSha256': digest(TARGET),
                   'priorSealSha256': digest(SEAL)})
    mismatches = {item: digest(path_for(item)) for item, value in seal['bindings'].items()
                  if digest(path_for(item)) != value}
    report['sealedBindingMismatchesBefore'] = mismatches
    assert set(mismatches) == {name}, 'More than the reported audit helper changed'
    edited = (
        '    # Console-host cwd/command introspection can be denied on Windows. Their\n'
        '    # permitted role is established by name, direct parent and exact identity.\n'
        "    descendants = [{'pid': process.pid, 'createTimeUnixSeconds': process.create_time(),\n"
        "                    'parentPid': process.ppid(), 'name': process.name()}\n"
        '                   for process in sorted(consoles, key=lambda item: item.pid)]\n'
    )
    original = '    descendants = [process_record(process) for process in sorted(consoles, key=lambda item: item.pid)]\n'
    text = data.decode('utf-8').replace('\r\n', '\n')
    assert text.count(edited) == 1, 'The edit differs from the reported isolated replacement'
    restored_text = text.replace(edited, original, 1)
    candidates = [restored_text.encode('utf-8'), restored_text.replace('\n', '\r\n').encode('utf-8')]
    matches = [candidate for candidate in candidates if hashlib.sha256(candidate).hexdigest() == expected]
    assert len(matches) == 1, 'No reconstructed candidate matches the historical executed bytes'
    temporary = TARGET.with_name(TARGET.name + '.verified_restore.tmp')
    with temporary.open('xb') as stream:
        stream.write(matches[0])
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, TARGET)
    assert digest(TARGET) == expected and digest(SEAL) == report['priorSealSha256']
    assert all(digest(path_for(item)) == value for item, value in seal['bindings'].items())
    report.update({'passed': True, 'restoredHelperSha256': digest(TARGET),
                   'allPriorSealedBindingsMatchAfter': True,
                   'restoredOnlyReportedReplacementAndOriginalLineEndings': True,
                   'restorationScriptSha256': digest(Path(__file__)),
                   'preservedAgentEditSha256': digest(UNIT / 'preserved_agent_edit.py')})
except BaseException as error:
    report['error'] = type(error).__name__ + ': ' + str(error)
finally:
    report['completedAt'] = datetime.now(timezone.utc).isoformat()
    with (UNIT / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
print(json.dumps({key: report.get(key) for key in ('passed', 'allPriorSealedBindingsMatchAfter', 'error')}))
raise SystemExit(0 if report['passed'] else 1)
