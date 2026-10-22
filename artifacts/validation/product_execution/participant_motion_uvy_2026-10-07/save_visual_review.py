"""Bind the 13 native originals and overlays visually inspected by root."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RUN = UNIT / 'run'
report = json.loads((RUN / 'report.json').read_text(encoding='utf-8'))
assert report['complete'] and not report['productionPromoted']

def binding(path):
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}

notes = {
 'tennis_V01': {
  0: 'Both active players are visible on the court; selected green fragment3 encloses a foreground bald spectator. No selected box encloses either player.',
  409: 'Both court players remain visible. The sole selected green fragment3 is the same foreground spectator, not either active participant.',
  604: 'Court players are still visible, with the right player near the edge. Original class1 annotations are absent as previously documented; selected fragment3 still encloses the foreground spectator.',
  818: 'The camera is zoomed into a court player beside a ball attendant and officials; no moving fragment is selected. This changed framing is not an independently verified camera cut.'},
 'tennis_V02': {
  0: 'Selected green fragment4 encloses the visible nearer court player; the distant participant is not selected. Crowded foreground boxes remain grey.',
  99: 'Selected fragment4 still encloses the nearer participant. The first publisher-unmatched selection includes a visible participant, so a failed IoU agreement is not proof of a nonparticipant false positive. The distant participant is unselected.',
  484: 'Selected fragment4 follows the nearer participant during a lateral lunge; the distant participant remains unselected. This sample supplies neither identity continuity nor pose accuracy ground truth.',
  967: 'The camera has panned down across spectators and cropped most of the court; no fragment is selected. Changed framing is not a verified cut/occlusion annotation.'},
 'tennis_V03': {
  0: 'Selected fragment4 encloses the nearer active player while selected large fragment5 encloses the extreme-left foreground spectator. The far active player is unselected.',
  140: 'The same near-player and foreground-spectator selections persist. Original class1 identity9 marks the seated chair umpire; that known publisher defect remains visible and uncorrected.',
  221: 'Near player and left spectator are selected; the far participant is visible and unselected. The additional spectator selection is a clear participant error in this reviewed sample.',
  230: 'Near player and foreground spectator remain selected, with the far participant unselected. Losing a cached matched player proposal is distinct from failing to detect that person.',
  442: 'Both players are visible after the exchange; no fragment is selected. The foreground spectator is no longer selected in this frame.'}
}
images, frames = {}, []
for sequence, per_frame in notes.items():
    for index, note in per_frame.items():
        paths = [RUN / sequence / f'review_{index:06}_{kind}.png' for kind in ('original', 'overlay')]
        for path in paths:
            assert path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
            images[path.relative_to(ROOT).as_posix()] = binding(path)
        frames.append({'sequence': sequence, 'frameIndexZeroBased': index, 'observation': note,
                       'nativeOriginal': paths[0].relative_to(ROOT).as_posix(),
                       'nativeOverlay': paths[1].relative_to(ROOT).as_posix()})
assert len(frames) == 13 and len(images) == 26
assert set(images) == {path.relative_to(ROOT).as_posix() for path in RUN.rglob('*.png')}
record = {'reviewedAt': datetime.now(timezone.utc).isoformat(), 'reviewer': 'root',
          'allSavedNativeOriginalsAndOverlaysViewed': True, 'reviewedFrames': 13,
          'reviewedImages': 26, 'frames': frames, 'imageBindings': images,
          'reportBinding': binding(RUN / 'report.json'), 'scriptBinding': binding(Path(__file__)),
          'decision': 'REJECT_MOTION_ONLY_PARTICIPANT_PROMOTION',
          'reason': 'Visible spectators qualify while active players are missed; retained cached matched proposals also lose recall.',
          'qualificationEvidence': False, 'productionPromoted': False,
          'limitations': ['AI visual review of reused source images; no independent human ground truth.',
                         'Review-selected frames are diagnostic and do not estimate clip-wide participant precision.',
                         'Publisher-unmatched selections include localization failures and label omissions, not only spectators.',
                         'No complete camera-motion/cut, occlusion, identity, physical speed or 3D validation.']}
with (UNIT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
print(json.dumps({'reviewedFrames': 13, 'reviewedImages': 26, 'decision': record['decision']}))
