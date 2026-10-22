"""Append current outcomes while retaining the exact complete prior progress bytes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
PROGRESS = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
MOTION = ROOT / 'artifacts/validation/product_execution/participant_motion_uvy_2026-10-07'

def binding(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

worker = json.loads((UNIT / 'worker_review.json').read_text(encoding='utf-8'))
runtime = json.loads((UNIT / 'runtime_verification_v2.json').read_text(encoding='utf-8'))
coverage = json.loads((UNIT / 'worker_observation_coverage.json').read_text(encoding='utf-8'))
motion = json.loads((MOTION / 'run/report.json').read_text(encoding='utf-8'))
resource = json.loads((MOTION / 'run/resource_receipt.json').read_text(encoding='utf-8'))
visual = json.loads((MOTION / 'visual_review.json').read_text(encoding='utf-8'))
audit = json.loads((MOTION / 'audit/receipt.json').read_text(encoding='utf-8'))
assert worker['passed'] and worker['actualWorkerCompletionVerified'] and runtime['passed']
assert motion['complete'] and resource['passed'] and visual['reviewedImages'] == 26
assert audit['passed']
assert motion['pooledCounts']['proposalMatched'] == 2479 and motion['pooledCounts']['selectedMatched'] == 710
assert all(row['poseObservations'] == row['racketObservations'] == 0 for row in coverage['countsByCourtRole'].values())
before = PROGRESS.read_bytes()
assert len(before) == 106000 and hashlib.sha256(before).hexdigest() == '568600bf45df94aa12753895831f513aeed7a353e9ed431985cd9b48eb6e7e75', 'Preserve any newer progress edits'
with (UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md').open('xb') as stream:
    stream.write(before)
addition = '''

## Actual worker recovery and motion-selection rejection — 7 October 2026

The [fresh current-revision real-worker smoke](../experiments/WORKER_STORAGE_SMOKE_2026-10-07.md) **COMPLETED** after independently observed RAM recovery, with unchanged resource guards. The private job processed30 native1920×1080/60FPS frames, observed all four model constructions and completed actual pipeline/publication. Reopened lazy/published JSON plus CSV/HTML equal frozen normalizer/reporting outputs on the same new raw bytes, with no removed fields. Root terminal audit verifies exact actual interpreter sampling/retirement and source/weight/history/live-product preservation. All complete normalized sections contain30frames, but actual pose/racket observations remain **zero for both court roles**. This replaces the pending current-revision worker compatibility milestone; it supplies no participant/pose/racket accuracy or full-match capacity evidence. All consumed v2/v3 and earlier declined attempts remain unchanged.

The prior website was absent; documented hidden startup restores it. Original post-start check retains **FAILED_TOPOLOGY_CHECK**, because it excluded the server's additional native Windows console child. Independent exact-PID/create-time/native-executable review plus a separate v2 adapter verifies both consoles, server/worker topology and exclusive loopback listener without altering the original helper. Fresh HTTP compatibility passes3demos,12exports,6media-rangechecks and7indexassets, preserving all10jobs/319files. Evidence is under `artifacts/validation/product_execution/resume_checks_2026-10-07/`.

The [motion-selector expansion](../experiments/PARTICIPANT_MOTION_UVY_2026-10-07.md) is **rejected for production**. Its single frozen unchanged-selector run completes2230frames across three reused UVY clips (at most two additional match groups), without model inference or label-guided camera sampling. All historical proposal counts replay exactly. Motion retains710 of2479cached publisher player matches (V01:1329→0; V02:631→330; V03:519→380), loses1769matches, and emits1620publisher-unmatched selections. Unmatched is not validated participant FP/precision: reviewed V02 mismatches include an actual near player. Root reviews every13native source/overlay pair; V01 andV03 visibly select foreground spectators while missing court participants. Known V01frame604/V03frame140 publisher label defects remain unchanged. Separate pure-stdlib arithmetic/source/identity audit passes; no ORB, selector, model or source-video rerun is used by that audit.

All motion-trial owned identities retire; production sources/config and all10jobs/319files remain unchanged. The motion-only prototype stays research-only, with null court roles and no physical metrics. Further work must add stronger participant evidence and representative controls, separate detector coverage from selector loss, and keep tracked identity distinct from court role. Untouched supported recordings, independent speed/3D references, downstream observation accuracy and full-match capacity remain open. Continue the full original prompt; neither software compatibility nor this rejected idea completes the scientific/product goal.
'''.encode('utf-8')
PROGRESS.write_bytes(before + addition)
assert PROGRESS.read_bytes() == before + addition
with (UNIT / 'progress_append_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump({'appendedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'originalPrefixPreservedExactly': True, 'before': binding(UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md'),
               'after': binding(PROGRESS), 'additionBytes': len(addition), 'scriptBinding': binding(Path(__file__))}, stream, indent=2)
print(json.dumps({'passed': True, 'beforeBytes': len(before), 'afterBytes': len(before + addition)}))
