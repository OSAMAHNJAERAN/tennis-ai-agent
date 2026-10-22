"""Verify this exact hidden Windows launch, including both native consoles."""
from pathlib import Path
import psutil

def runtime_processes(checks, expected):
    assert expected['state'] == 'OWNED_IDLE_PAIR'
    records = expected['processes']
    assert len(records) == 2
    server_record, worker_record = records
    expected_python = checks.BASE_PYTHON.resolve()
    server = psutil.Process(server_record['pid'])
    worker = psutil.Process(worker_record['pid'])
    recognized = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if ('src.api.worker' in args or
                any(Path(argument).name.lower() == 'serve_product.py' for argument in args[1:])):
            recognized.append(process.pid)
    assert sorted(recognized) == sorted([server.pid, worker.pid]), 'Unexpected/duplicate/orphan runtime'
    actual_records = [checks.process_record(process) for process in (server, worker)]
    assert actual_records == records, 'Declared native runtime identities or commands changed'
    for record in actual_records:
        assert Path(record['exe']).resolve() == expected_python
        assert Path(record['cwd']).resolve() == checks.ROOT
    assert server.cmdline() == [str(expected_python), 'scripts/run/serve_product.py']
    assert worker.cmdline() == [str(expected_python), '-u', '-m', 'src.api.worker']
    assert worker.ppid() == server.pid
    consoles = expected['ownedDescendants']
    assert len(consoles) == 2, 'Require exactly the two independently reviewed consoles'
    server_consoles = [row for row in consoles if row['parentPid'] == server.pid]
    worker_consoles = [row for row in consoles if row['parentPid'] == worker.pid]
    assert len(server_consoles) == len(worker_consoles) == 1
    actual_consoles = []
    for record in consoles:
        process = psutil.Process(record['pid'])
        actual = checks.process_record(process)
        assert actual == record, 'Exact console identity/topology changed'
        assert record['name'].lower() == 'conhost.exe'
        assert Path(record['exe']).resolve() == Path('C:/Windows/System32/conhost.exe').resolve()
        assert record['cmdline'] == ['\\??\\C:\\Windows\\system32\\conhost.exe', '0x4']
        assert Path(record['cwd']).resolve() == Path('C:/Windows').resolve()
        assert process.children() == [], 'Console owns an unexpected process'
        actual_consoles.append(actual)
    assert {process.pid for process in server.children()} == {worker.pid, server_consoles[0]['pid']}
    assert {process.pid for process in worker.children()} == {worker_consoles[0]['pid']}
    assert {process.pid for process in worker.children(recursive=True)} == {worker_consoles[0]['pid']}
    assert {process.pid for process in server.children(recursive=True)} == {worker.pid, *(row['pid'] for row in consoles)}
    listeners = [{'pid': row.pid, 'host': row.laddr.ip, 'port': row.laddr.port}
                 for row in psutil.net_connections(kind='tcp')
                 if row.status == 'LISTEN' and row.laddr.port == 8000]
    listeners.sort(key=lambda row: (row['pid'] or -1, row['host']))
    assert listeners == expected['port8000Listeners']
    assert listeners and all(row['pid'] == server.pid and row['host'] in ('127.0.0.1', '::1') for row in listeners)
    assert expected['launchWrappers'] == []
    for record in records + consoles:
        assert psutil.Process(record['pid']).create_time() == record['createTimeUnixSeconds']
    return {'state': 'OWNED_IDLE_PAIR', 'processes': actual_records,
            'port8000Listeners': listeners, 'ownedDescendants': actual_consoles,
            'launchWrappers': []}
