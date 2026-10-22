"""Measure resources while polling the exact persistent person-export job."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.api.jobs import JobStore, Settings

settings = Settings.from_env()
store = JobStore(settings.root)
record = json.loads((ROOT/'artifacts/validation/product_execution/people_evidence_run.json').read_text(encoding='utf-8'))
analysis_id = record['analysisId']
samples = []
previous = None
while True:
    job = store.get(analysis_id)
    if not job:
        raise RuntimeError('The exact saved job disappeared.')
    processes = []
    for process in psutil.process_iter(['pid','cmdline','memory_info']):
        try:
            command = process.info['cmdline'] or []
            if 'src.api.worker' in command and Path(process.cwd()).resolve() == ROOT:
                related = [process] + process.children(recursive=True)
                processes = [{'pid':p.pid,'name':p.name(),'rssBytes':p.memory_info().rss} for p in related]
        except (psutil.NoSuchProcess,psutil.AccessDenied):
            continue
    sample = {'at':datetime.now(timezone.utc).isoformat(),'status':job['status'],'stage':job['stage'],
              'availableRamBytes':psutil.virtual_memory().available,'workerAndChildrenRssBytes':sum(p['rssBytes'] for p in processes),
              'processes':processes}
    gpu = shutil.which('nvidia-smi')
    if gpu:
        try:
            sample['gpu'] = subprocess.check_output([gpu,'--query-gpu=name,memory.total,memory.used','--format=csv,noheader,nounits'],
                                                  text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW).strip()
        except (OSError,subprocess.SubprocessError):
            pass
    samples.append(sample)
    state=(job['status'],job['stage'])
    if state != previous:
        print(json.dumps({**{k:v for k,v in sample.items() if k!='processes'},'analysisId':analysis_id}),flush=True)
        previous=state
    if job['status'] in {'COMPLETED','FAILED'}:
        report={'analysisId':analysis_id,'status':job['status'],'startedAt':job['started_at'],'completedAt':job['completed_at'],
                'sampleIntervalSeconds':2,'peakSampledWorkerAndChildrenRssBytes':max(s['workerAndChildrenRssBytes'] for s in samples),
                'minimumSampledAvailableRamBytes':min(s['availableRamBytes'] for s in samples),
                'measurementScope':'Polling began after job acceptance; sampled RSS includes shared-page double counts and is not peak allocator/VRAM measurement or long-clip capacity qualification.',
                'samples':samples}
        output=ROOT/'artifacts/validation/product_execution/people_evidence_runtime.json'
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({'output':str(output),'status':job['status'],'error':job.get('error')}),flush=True)
        break
    time.sleep(2)
