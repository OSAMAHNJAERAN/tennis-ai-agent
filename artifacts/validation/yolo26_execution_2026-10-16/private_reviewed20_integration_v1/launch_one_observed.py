"""One owned bounded private run; the unchanged runner retains admission checks."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent
OUTPUT = ROOT/'outputs/private_yolo26_reviewed20_candidate_run1_2026-10-16'
CONFIG = 'configs/evaluation/yolo26.private-reviewed20-match172-2026-10-16.yaml'
EXPECTED = '6f90ce4b90a6a9d0323f20337deb55491f365b50b7945195c382700298fd0c57'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUTPUT.exists() or sha(ROOT/CONFIG) != EXPECTED:
        raise ValueError('Fresh output and exact prepared config required')
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    observer = BASE/'run_observed.py'
    argv = [sys.executable, '-u', str(observer), '--config', CONFIG, '--source-id', 'match172',
            '--output', str(OUTPUT), '--frames', '30', '--device', '0', '--execute']
    receipt = {'status': 'RUNNING_ONE_BOUNDED_PRIVATE_ATTEMPT', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'argv': argv, 'observer_sha256': sha(observer), 'source_binding_before': sources,
               'max_wall_seconds': 180, 'qualification_evidence': False, 'final_test_used': False,
               'no_automatic_retry': True}
    start = time.monotonic()
    with (BASE/'execution_console.log').open('x', encoding='utf-8') as console:
        child = subprocess.Popen(argv, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['owned_pid'] = child.pid
        with (BASE/'execution_start.json').open('x', encoding='utf-8') as out:
            json.dump(receipt, out, indent=2)
        try:
            code = child.wait(timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], timeout=15,
                           stdout=console, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            code = child.wait(timeout=15)
            receipt['timeout_terminated_owned_process_tree'] = True
    after = {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    changes = {'changed': sorted(k for k in sources.keys() & after.keys() if sources[k] != after[k]),
               'added': sorted(after.keys()-sources.keys()), 'removed': sorted(sources.keys()-after.keys())}
    receipt.update(status='PROCESS_TERMINAL_CHECK_ACTUAL_RUN_RECEIPT', exit_code=code,
                   completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-start, source_changes_during_execution=changes,
                   console_sha256=sha(BASE/'execution_console.log'))
    run_receipt = OUTPUT/'evaluation_receipt.json'
    if run_receipt.is_file():
        value = json.loads(run_receipt.read_text(encoding='utf-8-sig'))
        receipt.update(actual_run_status=value['status'], evaluation_receipt_sha256=sha(run_receipt))
    with (BASE/'execution_process_receipt.json').open('x', encoding='utf-8') as out:
        json.dump(receipt, out, indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_binding_before'}))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
