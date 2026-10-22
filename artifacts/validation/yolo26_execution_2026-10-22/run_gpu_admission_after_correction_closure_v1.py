"""One pure host admission after terminal correction regression and real-worker closure; no execution release."""
import builtins
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]
REFERENCE = DIRECTORY.parent / 'yolo26_execution_2026-10-17/gpu_admission_after_rule_replay_v1.json'
DESTINATION = DIRECTORY / 'gpu_admission_after_correction_closure_v1.json'
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'PIL', 'pandas'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if DESTINATION.exists():
        raise FileExistsError('Preserve the single admission receipt; do not repeat this probe')
    present_before = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    if present_before:
        raise AssertionError('Pure admission started with forbidden runtimes: ' + repr(present_before))
    blocked = []
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in FORBIDDEN:
            blocked.append(name)
            raise AssertionError('Runtime import forbidden in pure admission: ' + name)
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    sys.path.insert(0, str(ROOT))
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    expected = reference['training_helpers_expected']
    if (type(expected) is not dict or len(expected) != 10 or
            any(type(name) is not str or type(value) is not str or
                re.fullmatch('[0-9a-f]{64}', value) is None
                for name, value in expected.items())):
        raise ValueError('Require all ten frozen helper bindings from the prior receipt')
    before = {name: digest(ROOT / name) for name in expected}
    from scripts.train.train_yolo26_author_ball_repair import resource_snapshot, resource_guard
    observed_utc = datetime.now(timezone.utc)
    # This is the only resource sample. No numerical/image runtime, CUDA or
    # nvidia-smi is involved in the unchanged host-only adapter.
    resources = resource_snapshot(ROOT)
    guard_error = None
    try:
        resource_guard(resources, 'before_gpu')
        status = 'HOST_ADMISSION_PASSED_NO_EXECUTION_RELEASE'
        admitted = True
    except ValueError as error:
        guard_error = str(error)
        status = 'RESOURCE_REFUSED_BEFORE_TORCH'
        admitted = False
    after = {name: digest(ROOT / name) for name in expected}
    present_after = sorted(name for name in sys.modules if name.split('.')[0] in FORBIDDEN)
    helpers_unchanged = before == after == expected
    receipt = {
        'observed_utc': observed_utc.isoformat(),
        'observed_local': observed_utc.astimezone(timezone(timedelta(hours=8), 'Asia/Singapore')).isoformat(),
        'local_timezone': 'Asia/Singapore', 'local_utc_offset': '+08:00',
        'status': status, 'host_only_admission_passed': admitted, 'resources': resources,
        'resource_snapshot_calls': 1, 'resource_guard_stage': 'before_gpu',
        'before_gpu_ram_floor_bytes': 4608 * 1024 ** 2,
        'disk_floor_bytes': 1024 ** 3, 'resource_guard_error': guard_error,
        'forbidden_import_attempts': blocked,
        'forbidden_modules_present_before': present_before,
        'forbidden_modules_present': present_after,
        'training_helper_reference': str(REFERENCE),
        'training_helper_reference_sha256': digest(REFERENCE),
        'training_helpers_expected': expected, 'training_helpers_before': before,
        'training_helpers_observed': after, 'training_helpers_unchanged': helpers_unchanged,
        'runner_sha256': digest(Path(__file__)),
        'full_source_runtime_release_created': False, 'training_release_created': False,
        'torch_imported': False, 'cuda_initialized': False,
        'gpu_execution_performed': False, 'training_performed': False,
        'scope': 'Single fresh host RAM/disk admission and frozen-helper comparison only; ROOT must separately decide every next stage.',
    }
    with DESTINATION.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: receipt[key] for key in (
        'observed_utc', 'observed_local', 'status', 'resources',
        'training_helpers_unchanged', 'forbidden_modules_present',
        'training_release_created')}))
    return 0 if helpers_unchanged and not blocked and not present_after else 1


if __name__ == '__main__':
    sys.exit(main())

