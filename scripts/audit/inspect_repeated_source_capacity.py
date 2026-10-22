"""Print a read-only snapshot of one already-running isolated capacity audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(directory):
    directory = directory.resolve()
    allowed = ROOT / 'artifacts/validation/product_execution'
    if not directory.is_relative_to(allowed.resolve()) or directory == allowed.resolve():
        raise ValueError('Expected an existing isolated capacity directory')
    protocol = read(directory / 'protocol.json')
    owner = read(directory / 'running_process.json')
    expected_script = ROOT / 'scripts/audit/run_repeated_source_capacity.py'
    bindings = {name: digest(ROOT / name) == expected for name, expected in protocol['sourceFiles'].items()}
    bindings['scripts/audit/run_repeated_source_capacity.py'] = digest(expected_script) == protocol['scriptSha256']
    processes = []
    owner_matches = False
    try:
        process = psutil.Process(owner['pid'])
        command = process.cmdline()
        owner_matches = (len(command) == 5 and Path(command[1]).resolve() == expected_script.resolve()
                         and command[2] == '--directory' and Path(command[3]).resolve() == directory
                         and command[4] == '--child' and Path(process.cwd()).resolve() == ROOT)
        if owner_matches:
            for item in [process, *process.children(recursive=True)]:
                try:
                    memory = item.memory_info()
                    cpu = item.cpu_times()
                    processes.append({'pid': item.pid, 'createTime': item.create_time(), 'command': item.cmdline(),
                                      'rssBytes': memory.rss, 'privateBytes': getattr(memory, 'private', memory.vms),
                                      'userCpuSeconds': cpu.user, 'systemCpuSeconds': cpu.system})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    last = None
    count = peak_rss = peak_private = 0
    minimum_available = None
    with (directory / 'resource_samples.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # A concurrently appended partial record is not evidence.
            sample = json.loads(line)
            last = sample
            count += 1
            peak_rss = max(peak_rss, sample['ownedTreeRssBytes'])
            peak_private = max(peak_private, sample['ownedTreePrivateBytes'])
            minimum_available = sample['availableRamBytes'] if minimum_available is None else min(minimum_available, sample['availableRamBytes'])
    job = None
    accepted = directory / 'accepted_job.json'
    if accepted.is_file():
        identity = read(accepted)['analysisId']
        db = directory / 'isolated_product/analyses.sqlite3'
        with sqlite3.connect(db.as_uri() + '?mode=ro', uri=True, timeout=15) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute('SELECT * FROM analyses WHERE id=?', (identity,)).fetchone()
            job = dict(row) if row else None
    receipts = {name: read(directory / name) if (directory / name).is_file() else None
                for name in ('worker_receipt.json', 'resource_receipt.json')}
    return {'checkedAt': datetime.now(timezone.utc).isoformat(), 'directory': str(directory),
            'sourceBindingsUnchanged': all(bindings.values()), 'sourceBindingChecks': bindings,
            'ownerPid': owner['pid'], 'ownerIdentityMatches': owner_matches, 'processes': processes,
            'job': job, 'observedCompletedSamples': count, 'lastResourceSample': last,
            'peakSampledOwnedTreeRssBytesSoFar': peak_rss,
            'peakSampledOwnedTreePrivateBytesSoFar': peak_private,
            'minimumSampledAvailableRamBytesSoFar': minimum_available,
            'terminalReceipts': receipts, 'qualificationEvidence': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect((ROOT / args.directory).resolve()), allow_nan=False))
