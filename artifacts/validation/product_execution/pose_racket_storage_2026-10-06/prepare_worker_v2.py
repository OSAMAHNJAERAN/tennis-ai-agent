"""Preserve the prepared generic smoke and require both enabled raw series."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
OLD = ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke.py'
NEW = ROOT / 'scripts/audit/verify_pose_racket_storage_worker_smoke_v2.py'
OLD_HASH = '1fee98a45597e5744aa2e22cb4a343b15e929b94bcb36fa23d9d65df5e78734a'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def main():
    assert digest(OLD) == OLD_HASH
    assert not (UNIT / 'worker_smoke/launch_marker.json').exists(), 'Original preparation must remain unlaunched'
    original_preparation = json.loads((UNIT / 'worker_smoke/preparation.json').read_text(encoding='utf-8'))
    assert original_preparation['scriptSha256'] == OLD_HASH
    source = OLD.read_text(encoding='utf-8')
    source = replace_once(source,
        "UNIT = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/worker_smoke'",
        "UNIT = ROOT / 'artifacts/validation/product_execution/pose_racket_storage_2026-10-06/worker_smoke_v2'")
    source = replace_once(source, 'scripts\\\\audit\\\\verify_pose_racket_storage_worker_smoke.py --run',
        'scripts\\\\audit\\\\verify_pose_racket_storage_worker_smoke_v2.py --run')
    source = replace_once(source, "    for name in ('imagePeople', 'poseTracking', 'racketTracking'):",
        "    for name in ('poseTracking', 'racketTracking'):\n        if value.get(name) is None or len(value[name]['frames']) != 30:\n            raise ValueError('Enabled pose/racket section is missing or incomplete: ' + name)\n    for name in ('imagePeople', 'poseTracking', 'racketTracking'):")
    source = replace_once(source, 'def verify_saved_result(output, claimed_job, source_credit):', '''def verify_raw_pose_racket(output):
    pose = strict_json(output / 'player_motion.json')
    racket = strict_json(output / 'racket_tracking.json')
    if pose.get('pose_enabled') is not True or racket.get('enabled') is not True:
        raise ValueError('This selected recipe must export both enabled observation sections')
    checked = {}
    for name, values in (('pose_frames', pose.get('pose_frames')), ('racket_frames', racket.get('frames'))):
        if not isinstance(values, list) or len(values) != 30 or not all(isinstance(frame, dict) for frame in values):
            raise ValueError('Enabled raw pose/racket frame dictionaries missing or incomplete: ' + name)
        checked[name] = len(values)
    return {'enabled': True, 'rawFrameCounts': checked,
            'positiveObservationCoverageRequired': False,
            'scope': 'Complete enabled raw series; missing selected roles and null measurements remain legitimate.'}


def verify_saved_result(output, claimed_job, source_credit):''')
    source = replace_once(source, "    raw = RUN / 'raw_artifact_view'", "    raw_pose_racket = verify_raw_pose_racket(output)\n    raw = RUN / 'raw_artifact_view'")
    source = replace_once(source, "         'comparisons': comparisons, 'normalizedSectionCounts': counts,",
        "         'comparisons': comparisons, 'normalizedSectionCounts': counts,\n         'enabledRawPoseRacket': raw_pose_racket,")
    ast.parse(source, filename=str(NEW))
    with NEW.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(source)
    receipt = {'preparedAt': datetime.now(timezone.utc).isoformat(), 'workerInvoked': False,
        'status': 'PREPARED_NOT_EXECUTED', 'oldPreparedHelper': {'path': str(OLD), 'sha256': OLD_HASH},
        'oldPreparationSha256': digest(UNIT / 'worker_smoke/preparation.json'),
        'newHelper': {'path': str(NEW), 'bytes': NEW.stat().st_size, 'sha256': digest(NEW)},
        'reason': 'Independent review found generic smoke allowed optional null pose/racket sections. Require enabled complete raw and normalized 30-frame series, while accepting zero detector coverage.',
        'oldPreparationPreservedUnlaunched': True, 'guardsChanged': False,
        'earlierCliDiagnostic': {'commandOmittedRequiredPrepareFlag': True, 'exitCode': 2,
            'prepareWritesOrChildLaunch': False, 'correctedInvocationUsedPrepareFlag': True}}
    with (UNIT / 'worker_preparation_revision.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
