"""Seal the completed diagnostic and actual root visual review once."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
PROGRESS = ROOT / 'docs/plans/TENNIS_EXECUTION_PROGRESS.md'
EXPECTED_PROGRESS = 'da41eb69aef17bc3b4d9b153b5004404fb9225c45ec563c8bb23c6732534d428'
NOTES = {
    ('tennis_V01', 0): 'Both visible players receive support; large foreground spectator fragments are rejected or unknown because their complete sampling patches extend outside the image.',
    ('tennis_V01', 409): 'The left player remains supported. The valid right-player proposal is rejected at the surface boundary: this is the one lost original class1 match.',
    ('tennis_V01', 604): 'Visible players have no cached proposal in this view. Foreground spectator boxes receive rejection or unknown. The original missing-player-label defect remains unchanged.',
    ('tennis_V01', 818): 'The zoomed player and four nearby crew/official fragments receive support. Two of those supported crew proposals also overlap original class4 labels at IoU >= 0.5.',
    ('tennis_V02', 0): 'The near-player fragment4 and kneeling on-grass official fragment9 receive support. The distant left player fragment12 is rejected. The supported near player is publisher-unmatched because its localization fails the fixed IoU threshold; unmatched does not mean false participant.',
    ('tennis_V02', 99): 'Both visible player proposals receive support, along with a kneeling official and two distant boundary officials. Two supported officials overlap original class4 annotations.',
    ('tennis_V02', 484): 'Near-player fragment4 and a distant boundary figure receive support. Another distant figure is rejected. The kneeling official is not proposed at this frame; proposal absence is not evidence that the surface filter rejected it.',
    ('tennis_V02', 967): 'The downward foreground view has no supported cached proposal. Visible court people have no current proposal here; foreground patches are rejected or unknown.',
    ('tennis_V03', 0): 'No eligible surface component exists, so all proposals abstain. Both actual players remain visible. This mask failure cannot establish nonparticipation.',
    ('tennis_V03', 140): 'The near player fragment4 and the right boundary official fragment51 receive support. Fragment51 overlaps original class4 identity5. The known class1 identity9 chair-umpire label remains uncorrected.',
    ('tennis_V03', 221): 'Only the proposed near player receives support. The visible far player has no cached proposal. Foreground boxes are rejected or unknown.',
    ('tennis_V03', 230): 'Both visible player proposals receive support. Cached motion still selects only the near player and a foreground spectator; their surface intersection therefore retains only one player.',
    ('tennis_V03', 442): 'Both visible player proposals receive support after their source fragment identities changed. The cached motion selector is empty, so its intersection loses both. No fragment identities are merged.'
}

def binding(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}

def exclusive_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

report_path = UNIT / 'run/report.json'
audit_path = UNIT / 'audit/receipt.json'
resource_path = UNIT / 'resource_receipt.json'
report = json.loads(report_path.read_text(encoding='utf-8'))
audit = json.loads(audit_path.read_text(encoding='utf-8'))
resource = json.loads(resource_path.read_text(encoding='utf-8'))
assert report['complete'] and report['sourcesUnchanged'] and report['savedProductUnchanged']
assert audit['passed'] and not audit['qualificationEvidence'] and not audit['productionPromoted']
assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == '4310c4aec293eb075c62e6ffa8a5495276e66c5f178b3cf8a1e2e9316a4c36da'
old_progress = PROGRESS.read_bytes()
assert len(old_progress) == 109229 and hashlib.sha256(old_progress).hexdigest() == EXPECTED_PROGRESS
assert len(report['frames']) == len(NOTES) == 13
visual = {'reviewer': 'ROOT_AI_AGENT', 'method': 'Actual native context overlays viewed with view_image(detail=original)',
          'qualificationEvidence': False, 'productionPromoted': False,
          'decision': 'NO_PROMOTION_OF_SURFACE_ALONE_OR_MOTION_INTERSECTION',
          'reviewedAt': datetime.now(timezone.utc).isoformat(), 'frames': [],
          'limitations': ['Thirteen retrospective images selected after the motion experiment, at most two source match groups.',
                          'Native originals were reviewed in the preceding motion experiment and the audit verifies exact current copies.',
                          'Overlay green tint shows actual closed component support, not a convex hull or court calibration.',
                          'Publisher-unmatched boxes include valid players with insufficient IoU and unlabeled people; they are not validated false positives.',
                          'All participant statuses remain UNKNOWN and all court roles remain null.',
                          'Four proposals repeat reduced-grid pixels; their sample multiplicity is preserved, not treated as independent evidence.',
                          'No segmentation model, video decode, detector inference, identity recovery or physical measurement was evaluated.']}
for frame in report['frames']:
    key = (frame['sequence'], frame['frame_index'])
    directory = ROOT / frame['directory']
    visual['frames'].append({'sequence': key[0], 'frameIndex': key[1], 'observation': NOTES[key],
                             'overlay': binding(directory / 'native_context_overlay.png'),
                             'original': binding(directory / 'native_original.png'),
                             'componentMask': binding(directory / 'component_mask_01.png'),
                             'counts': frame['counts']})
visual_path = UNIT / 'visual_review.json'
exclusive_json(visual_path, visual)
note_path = ROOT / 'docs/experiments/PARTICIPANT_SURFACE_PROBE_2026-10-07.md'
note = '''# Participant surface-support diagnostic — 7 October 2026

Decision: keep the fixed surface-only filter and its intersection with the rejected motion selector out of production. The actual overlays admit on-grass officials/crew, exclude a valid boundary player and sometimes provide no usable mask. This is development evidence, not scientific qualification.

The single completed run used the thirteen already reviewed native stills from the UVY motion failure experiment. It reused cached tracked detections and the unchanged surface extractor; it did not decode video, run models, train, alter thresholds, rasterize a convex hull, merge masks or assign participant identities. Every pixel decision was saved before publisher labels were parsed. The fixed fifteen-point bbox-foot patch used nearest-integer reduced pixels, required all samples in bounds, and required at least eight supported pixels. No eligible component and out-of-bounds patches abstained.

| Fixed diagnostic group | Proposals | Original class1 matches | Publisher-unmatched | Original class1 misses |
| --- | ---: | ---: | ---: | ---: |
| Cached detector | 128 | 13 | 115 | 11 |
| Cached rejected motion selector | 14 | 4 | 10 | 20 |
| Surface supported detector proposals | 24 | 12 | 12 | 12 |
| Motion and surface intersection | 6 | 4 | 2 | 20 |

These are counts on thirteen retrospective frames, not full-clip metrics. The fixed matcher uses IoU 0.5. Publisher-unmatched includes visibly valid players whose boxes miss that threshold, unlabeled people and officials; it does not establish false-positive counts or participant precision. Known V01 frame604 missing labels and V03 frame140 chair-umpire class1 labeling are retained.

Actual visual review found the one lost match at V01 frame409, where the right player falls outside the selected surface. V01 frame818 supports a player plus four nearby crew/official fragments; two crew boxes also match original class4 labels. V02 frame99 supports both players plus three other people, including two original class4 matches. V03 frame140 supports the near player and a boundary official matching class4 identity5. V03 frame0 has no eligible component despite both players being visible. V03 frame442 supports both players while motion selects neither. These failures show why surface context can contribute evidence but cannot establish participant identity or repair missing detector proposals.

Of 128 proposals, 24 were supported, 25 rejected, 75 unknown because a sampling patch crossed image bounds, and four unknown because no surface component existed. Four proposals repeat reduced-grid pixels; multiplicity is retained and does not imply fifteen independent pixel samples. Roles remain null and participant status UNKNOWN throughout.

The actual guarded trial completed in 5.8877039 seconds including process overhead. Its twenty monitor samples observed peak owned RSS 140,779,520 bytes and minimum available system memory 2,564,644,864 bytes. The workload and its exact owned native console were retired. This small still-image diagnostic does not measure model throughput or full-match capacity.

An independent stdlib audit decoded the 0/1 grayscale masks without rerunning the extractor, reconstructed every sampling location/status, independently matched all groups, and checked thirteen native images, 181 input bindings, 643 loaded module files, staged semantic read order, resource acknowledgements and process retirement. Ten saved product jobs and 319 product files were preserved. Audit receipt: artifacts/validation/product_execution/participant_surface_probe_2026-10-07/audit/receipt.json. Root visual notes and image bindings: visual_review.json in the same unit. All attempts and source evidence are retained.

The original execution objective remains incomplete. Accurate automatic player selection, independent qualified camera/session data, full-match capacity and physical ball speed/3D validation remain separate outstanding requirements. The explicitly permitted assisted court-correction feature is being prepared separately and cannot replace automatic accuracy evaluation.
'''
with note_path.open('x', encoding='utf-8', newline='\n') as handle:
    handle.write(note)
backup = UNIT / 'TENNIS_EXECUTION_PROGRESS_before.md'
with backup.open('xb') as handle:
    handle.write(old_progress)
addition = b'\n\n## 7 October 2026: completed participant surface-support diagnostic\n\nThe single thirteen-frame pixel-context run completed and its independent audit passed. Surface support retains twelve of thirteen original detector/label matches and admits on-grass officials/crew. Motion plus surface retains four matches and still loses twenty original class1 boxes. Root reviewed all thirteen native overlays; neither diagnostic is promoted. Counts are retrospective publisher-label diagnostics, not participant precision or held-out accuracy. All181 bound inputs and ten saved jobs/319 files were preserved; no video decode or model inference ran. See docs/experiments/PARTICIPANT_SURFACE_PROBE_2026-10-07.md and artifacts/validation/product_execution/participant_surface_probe_2026-10-07/final_verification.json.\n\nThe brief-authorized assisted court-correction path is in isolated preparation, with exact source/frame bindings, visible landmark input and explicit annotation provenance. It is not integrated or verified yet and will not count as automatic accuracy or unlock unvalidated physical ball speed/3D. The full execution objective remains active and incomplete.\n'
PROGRESS.write_bytes(old_progress + addition)
assert PROGRESS.read_bytes().startswith(old_progress)
final = {'passed': True, 'status': 'REVIEWED_SURFACE_DIAGNOSTIC_NOT_PROMOTED',
         'qualificationEvidence': False, 'productionPromoted': False, 'fullExecutionObjectiveComplete': False,
         'trial': binding(report_path), 'resourceReceipt': binding(resource_path),
         'independentAudit': binding(audit_path), 'visualReview': binding(visual_path),
         'note': binding(note_path), 'preservedProgressPrefix': binding(backup),
         'currentProgress': binding(PROGRESS), 'pooledCounts': report['pooledCounts'],
         'completedAt': datetime.now(timezone.utc).isoformat()}
exclusive_json(UNIT / 'final_verification.json', final)
print(json.dumps({'passed': True, 'status': final['status'], 'fullExecutionObjectiveComplete': False,
                  'progressBytes': final['currentProgress']['bytes']}))
