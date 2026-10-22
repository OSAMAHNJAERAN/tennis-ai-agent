"""Count actual observed pose/racket entries separately from complete series length."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
WORKER = ROOT / 'artifacts/validation/product_execution/pose_racket_worker_smoke_retry_2026-10-07'
child = json.loads((WORKER / 'child_receipt.json').read_text(encoding='utf-8'))
assert child['passed']
directory = ROOT / child['outputDirectory']
pose_path, racket_path = directory / 'player_motion.json', directory / 'racket_tracking.json'
pose, racket = [json.loads(path.read_text(encoding='utf-8')) for path in (pose_path, racket_path)]
assert pose['pose_enabled'] and racket['enabled']
assert len(pose['pose_frames']) == len(racket['frames']) == 30
counts = {}
for role in ('1', '2'):
    counts[role] = {'poseObservations': sum(frame.get(role) is not None for frame in pose['pose_frames']),
                    'racketObservations': sum(frame[role]['bbox_xyxy'] is not None for frame in racket['frames']),
                    'racketStates': {state: sum(frame[role]['state'] == state for frame in racket['frames'])
                                     for state in sorted({frame[role]['state'] for frame in racket['frames']})}}
record = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'frames': 30, 'countsByCourtRole': counts,
          'qualificationEvidence': False, 'scope': 'Observed coverage on the actual reused half-second worker input; complete enabled output series alone do not imply observed pose/rackets.',
          'bindings': {path.relative_to(ROOT).as_posix(): {'bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in (pose_path, racket_path, Path(__file__))}}
with (UNIT / 'worker_observation_coverage.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(counts))
