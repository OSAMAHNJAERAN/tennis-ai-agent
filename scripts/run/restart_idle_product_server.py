"""Stop exact owned idle server/worker processes; leave every stored job intact.

The caller restarts scripts/run/serve_product.py after this bounded stop.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import psutil

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / 'outputs/product/analyses.sqlite3'


def snapshot():
    with sqlite3.connect(DATABASE.as_uri() + '?mode=ro', uri=True) as connection:
        rows = connection.execute('SELECT id,status FROM analyses ORDER BY id').fetchall()
    if any(state in {'UPLOADED', 'PROCESSING'} for _, state in rows):
        raise RuntimeError('Accepted analysis work exists; keep the server and worker running.')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    server, worker = psutil.Process(args.server_pid), psutil.Process(args.worker_pid)
    expected_python = str(Path(server.exe()).resolve())
    server_command = server.cmdline()
    server_script_matches = (len(server_command) == 2 and server_command[0] == expected_python
                             and (ROOT / server_command[1]).resolve() == ROOT / 'scripts/run/serve_product.py')
    if (Path(server.cwd()).resolve() != ROOT or not server_script_matches
            or Path(worker.cwd()).resolve() != ROOT or worker.ppid() != server.pid
            or worker.cmdline() != [expected_python, '-u', '-m', 'src.api.worker']):
        raise RuntimeError('The exact owned server/worker command and workspace must match.')
    server_children = server.children()
    if {child.pid for child in server_children} != {worker.pid}:
        raise RuntimeError('Unexpected server child process; do not stop.')
    if any(child.name().lower() != 'conhost.exe' for child in worker.children()):
        raise RuntimeError('Worker has a non-console child; do not stop.')
    before = snapshot()
    identities = {'server_pid': server.pid, 'server_create_time': server.create_time(),
                  'worker_pid': worker.pid, 'worker_create_time': worker.create_time()}
    if before != snapshot():
        raise RuntimeError('Job state changed; do not stop.')
    # Stop supervision first so it cannot spawn a replacement during this reload.
    server.terminate()
    server.wait(timeout=10)
    worker.terminate()
    worker.wait(timeout=10)
    after = snapshot()
    if before != after:
        raise RuntimeError('Persisted job state changed; inspect before restarting.')
    receipt = {'checked_at': datetime.now(timezone.utc).isoformat(), **identities,
               'jobs_before': before, 'jobs_after': after,
               'action': 'Stopped only exact idle owned runtime processes; restart the documented server command.',
               'accepted_work_cancelled': False, 'persistent_files_removed': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'stopped': identities, 'saved_jobs_preserved': len(after), 'receipt': str(args.receipt)}))


if __name__ == '__main__':
    main()
