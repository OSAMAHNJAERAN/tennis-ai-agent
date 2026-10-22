"""Read current exact workload identities and completed measurement receipts."""
import json
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent / 'registration_storage_measurement_2026-10-05'
rows = []
for directory in sorted(path for path in HERE.iterdir() if path.is_dir()):
    resource = directory / 'resource_receipt.json'
    row = {'case': directory.name, 'resourceReceiptPresent': resource.exists()}
    if resource.exists():
        value = json.loads(resource.read_text(encoding='utf-8'))
        row.update({'passed': value['passed'], 'samples': value['samples'], 'error': value.get('error')})
    identity_path = directory / 'actual_identity.json'
    if identity_path.exists():
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        try:
            process = psutil.Process(identity['pid'])
            exact = process.create_time() == identity['createTimeUnixSeconds']
            row['exactOwnedWorkloadLive'] = exact and process.is_running()
            if exact:
                row['memoryBytes'] = process.memory_info()._asdict()
                row['cpuSeconds'] = process.cpu_times()._asdict()
        except psutil.NoSuchProcess:
            row['exactOwnedWorkloadLive'] = False
    rows.append(row)
print(json.dumps({'parentTerminalReportPresent': (HERE / 'report.json').exists(), 'cases': rows}))
