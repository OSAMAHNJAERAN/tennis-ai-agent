"""One fresh, read-only authoritative capacity/runtime observation."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import psutil

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt-name', required=True)
    args = parser.parse_args()
    name = args.receipt_name
    assert name == Path(name).name and name.endswith('.json'), 'Owned receipt filename only'
    path = BASE / name
    assert not path.exists(), 'Observation receipts are exclusive'
    rows = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        argv = process.info['cmdline'] or []
        if not ('src.api.worker' in argv or
                any(Path(arg).name.lower() == 'serve_product.py' for arg in argv[1:])):
            continue
        rows.append({'pid': process.pid, 'createTimeUnixSeconds': process.create_time(),
                     'parentPid': process.ppid(), 'exe': process.exe(), 'cmdline': argv,
                     'cwd': process.cwd(), 'rssBytes': process.memory_info().rss})
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        jobs = connection.execute('SELECT id,status FROM analyses ORDER BY id').fetchall()
    with Path(__file__).open('rb') as stream:
        helper_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {'observedAt': datetime.now(timezone.utc).isoformat(),
              'readOnly': True, 'availableRamBytes': psutil.virtual_memory().available,
              'diskBytes': shutil.disk_usage(ROOT)._asdict(), 'runtimeCandidates': rows,
              'jobs': jobs, 'activeWork': any(state in ('PROCESSING', 'UPLOADED') for _, state in jobs),
              'helperSha256': helper_hash,
              'scope': 'Instant host observation; no workload, no launch authorization, no performance measurement.'}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ('availableRamBytes', 'activeWork', 'runtimeCandidates')}))


if __name__ == '__main__':
    main()
