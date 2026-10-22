"""Synthetic copy/hash seam only; bytes are deliberately not playable media."""
import builtins
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tracemalloc

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'PIL', 'pandas'}
SOURCES = ['src/api/analysis_generation.py', 'src/api/result_storage.py']
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
attempts = []
original_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.')[0] in FORBIDDEN:
        attempts.append(name)
        raise AssertionError('Copy/hash capacity cannot import runtime: ' + name)
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pins():
    return {name: digest(ROOT / name) for name in SOURCES}


def process_peak():
    class Memory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peakWorkingSet', 'workingSet', 'peakPaged',
            'paged', 'peakNonpaged', 'nonpaged', 'pagefile', 'peakPagefile')]
    info = Memory()
    info.cb = ctypes.sizeof(info)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Memory), ctypes.c_ulong]
    if not query(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return info.peakWorkingSet


def child(size):
    from src.api import analysis_generation as generation
    from src.api.result_storage import ResultStorage
    before = pins()
    case = RUN / ('case_' + str(size))
    case.mkdir()
    source = case / 'synthetic_input.bytes'
    block = bytes(range(256)) * 4096
    expected_hash = hashlib.sha256()
    with source.open('xb') as stream:
        for _ in range(size // len(block)):
            stream.write(block)
            expected_hash.update(block)
    del block
    expected_hash = expected_hash.hexdigest()
    files = {name: source for name in ('source.mp4', 'annotated_web.mp4', 'original.mp4')}
    result = {'summary': {'id': 'synthetic-copy-capacity', 'frames': 1}, 'analysisRevision': 0,
              'scope': 'SYNTHETIC_NONPLAYABLE_COPY_HASH_ONLY'}
    (case / '.result').mkdir()
    gc.collect()
    tracemalloc.start()
    begin = time.perf_counter()
    with ResultStorage(case / '.result') as storage:
        token = generation.publish_analysis_generation(storage, result, case, files=files)
    publish_seconds = time.perf_counter() - begin
    publish_retained, publish_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    generation._cached_hash.cache_clear()
    gc.collect()
    tracemalloc.start()
    begin = time.perf_counter()
    snapshot = generation.resolve_generation(case, token)
    copied = [snapshot.resolve_file(name) for name in files]
    with ResultStorage.open(snapshot.database) as stored:
        assert stored.root() == result
    assert all(path.stat().st_size == size and digest(path) == expected_hash for path in copied)
    validation_seconds = time.perf_counter() - begin
    validation_retained, validation_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    source_stat = source.stat()
    copied_stats = [path.stat() for path in copied]
    assert len({(entry.st_dev, entry.st_ino) for entry in [source_stat, *copied_stats]}) == 4
    assert all(entry.st_nlink == 1 for entry in [source_stat, *copied_stats])
    assert digest(source) == expected_hash
    after = pins()
    present = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    assert before == after and not attempts and not present
    return {'status': 'PASS', 'pid': os.getpid(), 'input_bytes': size,
        'copied_media_bytes': 3 * size, 'input_sha256': expected_hash, 'generation': token,
        'independent_file_inodes': True, 'source_unchanged': True,
        'publish_seconds': publish_seconds, 'publish_traced_retained_bytes': publish_retained,
        'publish_traced_peak_bytes': publish_peak, 'cold_validation_seconds': validation_seconds,
        'cold_validation_traced_retained_bytes': validation_retained,
        'cold_validation_traced_peak_bytes': validation_peak,
        'peak_process_working_set_bytes_including_fixture_setup': process_peak(),
        'retained_case_disk_bytes': sum(path.stat().st_size for path in case.rglob('*') if path.is_file()),
        'source_before': before, 'source_after': after, 'forbidden_import_attempts': attempts,
        'forbidden_modules_present': present, 'hash_cache_limit': generation._cached_hash.cache_info().maxsize}


def main():
    destination = RUN / 'receipt.json'
    if destination.exists():
        raise FileExistsError('Preserve retained capacity receipt')
    before = pins()
    for name in SOURCES:
        target = RUN / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    cases = []
    for size in (8 * 1024**2, 64 * 1024**2):
        result = subprocess.run([sys.executable, str(Path(__file__)), '--child', str(size)],
            capture_output=True, text=True, timeout=60, cwd=ROOT,
            env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        (RUN / (str(size) + '.stdout.txt')).write_text(result.stdout, encoding='utf-8')
        (RUN / (str(size) + '.stderr.txt')).write_text(result.stderr, encoding='utf-8')
        assert result.returncode == 0, (result.returncode, result.stderr)
        cases.append(json.loads(result.stdout))
    after = pins()
    assert before == after
    receipt = {'status': 'PASS', 'observed_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'Two actual isolated processes; synthetic nonplayable 8/64 MiB input each copied into three independent generation assets; no decoder/inference/long-match/live qualification',
        'measurement': 'Tracemalloc excludes input fixture setup; publish includes copied assets, JSON, database, manifest and pointer; validation includes cold asset/database hashes plus repeated digest equality checks. Native SQLite/OS caches are outside tracemalloc. Process working-set peak includes fixture setup.',
        'cases': cases, 'source_before': before, 'source_after': after, 'sources_unchanged': True,
        'retention': 'Each generation independently copies all selected assets. Inputs, history and failed private diagnostics are retained; storage grows with attempts and generations.',
        'correction_seams_remaining': ['review command and audit binding',
            'context-dependent event/line-call recomputation', 'derived analytics and prediction rebuild',
            'fresh corrected annotated rendering', 'expected-old-token commit of complete outputs'],
        'runner_sha256': digest(Path(__file__)), 'actual_video_used': False, 'gpu_used': False,
        'independent_accuracy_validated': False}
    destination.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'cases': cases}))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        print(json.dumps(child(int(sys.argv[2]))))
    else:
        main()
