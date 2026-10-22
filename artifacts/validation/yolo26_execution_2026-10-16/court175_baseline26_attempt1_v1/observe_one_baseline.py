from pathlib import Path
from datetime import datetime, timezone
import ctypes,json,subprocess,sys,time
sys.path.insert(0, str(Path.cwd()))
from scripts.train import train_yolo26_court_repair as training
root=Path.cwd(); artifact=root/"artifacts/validation/yolo26_execution_2026-10-16/court175_baseline26_attempt1_v1"
entry=artifact/"run_one_baseline.py"; output=root/"outputs/yolo26_court175_baseline26_attempt1_2026-10-16"
if output.exists(): raise FileExistsError("Fresh baseline output required")
class Memory(ctypes.Structure):
    _fields_=[("cb",ctypes.c_ulong),("faults",ctypes.c_ulong)]+[(key,ctypes.c_size_t) for key in ("peak_rss","rss","peak_paged","paged","peak_nonpaged","nonpaged","pagefile","peak_pagefile","private")]
ctypes.windll.kernel32.OpenProcess.restype=ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
ctypes.windll.kernel32.CloseHandle.argtypes=[ctypes.c_void_p]
def owned_memory(pid):
    handle=ctypes.windll.kernel32.OpenProcess(0x1000|0x10,False,pid)
    if not handle: return None
    try:
        value=Memory();value.cb=ctypes.sizeof(value)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle,ctypes.byref(value),value.cb): return None
        return {"rss_bytes":value.rss,"peak_working_set_bytes":value.peak_rss,"private_bytes":value.private}
    finally: ctypes.windll.kernel32.CloseHandle(handle)
receipt={"status":"OWNED_BASELINE_ATTEMPT_STARTING","created_utc":datetime.now(timezone.utc).isoformat(),"root_approval":"Explicit root message: ONE guarded baseline26 diagnostic only, no training; unchanged4.5GiBpre/1.5GiBpost/2GiBVRAM/1GiBdisk; max180s ownedchild, no retry/fallback/otherprocessstop", "entry":training._binding(entry),"output":str(output),"maximum_owned_wall_seconds":180,"poll_interval_seconds":2,"training_performed":False,"qualification_evidence":False,"sampling_errors":[],"samples":[],"source_bindings_before":training.source_bindings(),"runtime_bindings_before":training.runtime_bindings()}
start=time.perf_counter()
with (artifact/"stdout.txt").open("xb") as stdout,(artifact/"stderr.txt").open("xb") as stderr:
    child=subprocess.Popen([sys.executable,"-u",str(entry)],cwd=root,stdout=stdout,stderr=stderr)
    receipt["owned_child_pid"]=child.pid
    while child.poll() is None:
        if time.perf_counter()-start>180:
            child.kill();receipt["terminated_owned_child_for_wall_limit"]=True;break
        try:
            receipt["samples"].append({"elapsed_seconds":time.perf_counter()-start,"available_ram_bytes":training._resources(artifact)["free_ram_bytes"],"owned_child_memory":owned_memory(child.pid)})
        except Exception as exc: receipt["sampling_errors"].append({"type":type(exc).__name__,"reason":str(exc)})
        time.sleep(2)
    receipt["child_exit_code"]=child.wait(timeout=15)
receipt.update(ended_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,status="OWNED_BASELINE_ATTEMPT_TERMINAL")
receipt["source_changes"]=training.binding_changes(receipt["source_bindings_before"],training.source_bindings())
receipt["runtime_changes"]=training.binding_changes(receipt["runtime_bindings_before"],training.runtime_bindings())
path=output/"diagnostic.json"
if path.exists():
    document=training.read_json(path);receipt["diagnostic"]=training._binding(path);receipt["diagnostic_status"]=document["status"];receipt["rows"]=len(document["rows"]);receipt["runtime_probe_performed"]=document["runtime_probe_performed"]
training.write_json(artifact/"observer_receipt.json",receipt)
print(json.dumps({key:receipt.get(key) for key in ("status","owned_child_pid","child_exit_code","elapsed_seconds","diagnostic_status","rows","runtime_probe_performed","source_changes","runtime_changes")}))
