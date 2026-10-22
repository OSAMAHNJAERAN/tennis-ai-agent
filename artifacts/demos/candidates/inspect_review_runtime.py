"""Read-only process progress diagnostic for a pending media review."""

import json
import time

import psutil


rows = []
for process in psutil.process_iter():
    try:
        name = process.name().lower()
        if name not in {"python.exe", "ffmpeg.exe", "node.exe"}:
            continue
        cpu = process.cpu_times()
        rows.append({
            "pid": process.pid,
            "name": name,
            "ageSeconds": round(time.time() - process.create_time()),
            "cpuSeconds": round(cpu.user + cpu.system, 2),
        })
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        continue
print(json.dumps(rows, indent=2))
