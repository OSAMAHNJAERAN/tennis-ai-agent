"""One retained full backend check with source hashes and owned-process monitoring."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/validation/yolo26_execution_2026-10-09/backend_full_master_v1'


def snapshot():
    paths = [p for directory in ('src', 'tests', 'configs', 'scripts') for p in (ROOT / directory).rglob('*')
             if p.is_file() and p.suffix in {'.py', '.yaml', '.json'} and '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if not OUT.is_relative_to(ROOT):
        raise ValueError('Validation evidence must remain inside the workspace')
    OUT.mkdir(parents=True, exist_ok=False)
    started, before = datetime.now(timezone.utc).isoformat(), snapshot()
    (OUT / 'sources_start.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    command = [sys.executable, '-m', 'pytest', 'tests', '-q', '-p', 'no:cacheprovider',
               '--basetemp', str(OUT / 'tmp'), '--junitxml', str(OUT / 'junit.xml')]
    minimum_free, peak_rss, floor_samples = psutil.virtual_memory().available, 0, 0
    begin = time.monotonic()
    with (OUT / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        owned = psutil.Process(process.pid)
        stopped = None
        while process.poll() is None:
            free = psutil.virtual_memory().available
            minimum_free = min(minimum_free, free)
            floor_samples = floor_samples + 1 if free < 200_000_000 else 0
            try:
                peak_rss = max(peak_rss, sum(p.memory_info().rss for p in [owned, *owned.children(recursive=True)] if p.is_running()))
            except psutil.Error:
                pass
            if floor_samples >= 3:
                stopped = 'available_system_ram_below_200MB_three_consecutive_samples'
                for child in owned.children(recursive=True):
                    try:
                        child.terminate()
                    except psutil.Error:
                        pass
                process.terminate()
                break
            time.sleep(1)
        exit_code = process.wait()
    after = snapshot()
    suites = ET.parse(OUT / 'junit.xml').getroot() if (OUT / 'junit.xml').is_file() else None
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites.iter('testsuite')) if suites is not None else None
              for key in ('tests', 'failures', 'errors', 'skipped')}
    receipt = {'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'command': command, 'exit_code': exit_code, 'counts': counts, 'elapsed_seconds': time.monotonic() - begin,
               'changed_during_run': [key for key in before if before[key] != after.get(key)],
               'added_during_run': sorted(set(after) - set(before)), 'source_sha256_at_completion': after,
               'minimum_available_ram_bytes': minimum_free, 'peak_sampled_owned_tree_rss_bytes': peak_rss,
               'resource_stop': stopped, 'scientific_qualification': False,
               'status': 'PASS' if exit_code == 0 and not stopped else 'FAILED'}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key not in {'command', 'source_sha256_at_completion'}}))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
