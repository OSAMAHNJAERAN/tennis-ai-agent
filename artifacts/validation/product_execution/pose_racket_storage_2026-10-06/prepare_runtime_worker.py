"""Prepare fresh reviewed runtime/worker checks; never launch work or reload."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def save_source(path, source):
    ast.parse(source, filename=str(path))
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)


def main():
    originals = {
        'artifacts/validation/product_execution/frontend_entry_recovery_2026-10-06/verify_runtime.py':
            '094302f8d625cd63b1639c8bd482fa0eb3f536750db54a81bde0f392a3657657',
        'scripts/audit/verify_result_storage_latency_headroom_worker_smoke.py':
            '99285464e48ed647a8886ee49a860c91ecc72d23f343381dbf5c3e67f288fee5',
    }
    for name, expected in originals.items():
        assert digest(ROOT / name) == expected, name
    assert digest(ROOT / 'src/pipeline/phase6_pipeline.py') == 'ce88b62b44613509d5f4217d0d11fe8c13d9136c9f32c8bf97813d3bdc922811'
    assert digest(ROOT / 'src/utils/identity_frame_sequence.py') == 'f2d2118ccbe68493cd191de79c4b674ea2d6e36c7f81bb726bef6ef0deb17f04'
    runtime = (ROOT / next(iter(originals))).read_text(encoding='utf-8')
    runtime = replace_once(runtime,
        "    assert runtime_before == preflight['runtime'], 'Retain the exact current idle runtime without restart'\n    event_runtime = json.loads((BASE.parent / 'event_timestamp_lookup_2026-10-06/runtime_verification.json').read_text(encoding='utf-8'))\n    assert runtime_before == event_runtime['runtime'], 'The previously verified server/worker must remain unchanged'",
        "    retired = previous_identities_retired(preflight['runtime'])\n    assert runtime_before != preflight['runtime'], 'Require fresh documented runtime after guarded idle reload'")
    runtime = replace_once(runtime, "         'runtimeRetainedWithoutRestart': True,",
        "         'runtimeRetainedWithoutRestart': False,\n         'previousExactRuntimeIdentitiesRetired': retired,")
    runtime = replace_once(runtime,
        'Existing public result compatibility after reversible frontend publication; retained runtime, no inference or scientific qualification.',
        'Existing public result compatibility after pose/racket storage source integration and guarded idle reload; no inference or scientific qualification.')
    runtime_path = UNIT / 'verify_runtime.py'
    save_source(runtime_path, runtime)

    worker = (ROOT / 'scripts/audit/verify_result_storage_latency_headroom_worker_smoke.py').read_text(encoding='utf-8')
    worker = replace_once(worker,
        "UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_headroom_worker_smoke_2026-10-06'",
        "UNIT = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/worker_smoke'")
    worker = replace_once(worker,
        "CONFIG = ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'",
        "SECOND_ATTEMPT = ROOT / 'scripts/audit/verify_result_storage_latency_headroom_worker_smoke.py'\nSECOND_UNIT = ROOT / 'artifacts/validation/product_execution/result_storage_latency_headroom_worker_smoke_2026-10-06'\nSECOND_SHA256 = '99285464e48ed647a8886ee49a860c91ecc72d23f343381dbf5c3e67f288fee5'\nCONFIG = ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml'")
    worker = replace_once(worker, 'def origin_record():', '''def verify_second_attempt():
    if digest(SECOND_ATTEMPT) != SECOND_SHA256:
        raise ValueError('The consumed second helper changed')
    receipt = strict_json(SECOND_UNIT / 'resource_receipt.json')
    protocol = strict_json(SECOND_UNIT / 'protocol.json')
    if (receipt['status'] != 'NOT_STARTED' or receipt['passed'] is not False or
            receipt['inferenceChildStarted'] is not False or receipt['samples'] != 0 or
            receipt['minimumSampledAvailableRamBytes'] != 2_227_232_768 or
            protocol['scriptSha256'] != SECOND_SHA256 or 'sourceBindingsBefore' in protocol or
            (SECOND_UNIT / 'run').exists() or (SECOND_UNIT / 'actual_child_identity.json').exists()):
        raise ValueError('The second attempt must remain declined before model launch')
    return {path.relative_to(ROOT).as_posix(): {'sha256': digest(path), 'bytes': path.stat().st_size}
            for path in [SECOND_ATTEMPT] + sorted(path for path in SECOND_UNIT.rglob('*') if path.is_file())}


def origin_record():''')
    worker = replace_once(worker,
        "            'firstLatencyAttemptAvailableRamBytes': 2_355_306_496,",
        "            'firstLatencyAttemptAvailableRamBytes': 2_355_306_496,\n            'secondLatencyAttemptBindings': verify_second_attempt(),\n            'secondLatencyAttemptStatus': 'NOT_STARTED',\n            'secondLatencyAttemptAvailableRamBytes': 2_227_232_768,")
    worker = replace_once(worker,
        "        raise ValueError('The first declined attempt history changed before this conditional launch')",
        "        raise ValueError('The first declined attempt history changed before this conditional launch')\n    if verify_second_attempt() != strict_json(UNIT / 'origin.json')['secondLatencyAttemptBindings']:\n        raise ValueError('The second declined attempt history changed before launch')")
    worker = replace_once(worker,
        "        protocol['firstLatencyAttemptHistoryBefore'] = verify_first_attempt()",
        "        protocol['firstLatencyAttemptHistoryBefore'] = verify_first_attempt()\n        protocol['secondLatencyAttemptHistoryBefore'] = verify_second_attempt()")
    worker = replace_once(worker,
        "                ('firstLatencyAttemptHistory', 'firstLatencyAttemptHistoryBefore', verify_first_attempt)):",
        "                ('firstLatencyAttemptHistory', 'firstLatencyAttemptHistoryBefore', verify_first_attempt),\n                ('secondLatencyAttemptHistory', 'secondLatencyAttemptHistoryBefore', verify_second_attempt)):")
    worker = replace_once(worker,
        "path.name.startswith(('ball_candidates_', 'frame_registration_'))",
        "path.name.startswith(('ball_candidates_', 'frame_registration_', 'identity_frames_'))")
    worker = replace_once(worker, 'scripts\\\\audit\\\\verify_result_storage_latency_headroom_worker_smoke.py --run',
        'scripts\\\\audit\\\\verify_pose_racket_storage_worker_smoke.py --run')
    worker_path = ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke.py'
    save_source(worker_path, worker)
    receipt = {
        'preparedAt': datetime.now(timezone.utc).isoformat(), 'status': 'PREPARED_NOT_EXECUTED',
        'originals': originals,
        'created': {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
                    for path in (runtime_path, worker_path)},
        'runtimeStoppedOrStarted': False, 'workerInvoked': False,
        'scope': 'Fresh helpers preserve consumed histories and unchanged guards; root dispatch remains separate.'
    }
    with (UNIT / 'runtime_worker_preparation.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
