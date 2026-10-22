"""Extract exact decoded synthetic render frames for local visual inspection."""
import hashlib
import json
from pathlib import Path
import cv2

ART = Path(__file__).resolve().parent
case = ART/'temp/pytest/test_actual_two_encoded_videos0/private-rebuild'
seals = list(case.glob('corrected-render-*/corrected_render_manifest.json'))
assert len(seals) == 1
directory = seals[0].parent
manifest = json.loads(seals[0].read_text(encoding='utf-8'))
video = directory/'annotated_web.mp4'
with video.open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['artifacts'][video.name]['sha256']
destination = ART/'previews'
destination.mkdir(exist_ok=False)
capture = cv2.VideoCapture(str(video))
assert capture.isOpened()
saved, count = [], 0
try:
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if count in (0, 4, 8):
            path = destination/f'frame_{count}.png'
            assert cv2.imwrite(str(path), frame)
            saved.append(str(path))
        count += 1
finally:
    capture.release()
assert count == manifest['frames'] == 12
record = {'video': str(video), 'video_sha256': manifest['artifacts'][video.name]['sha256'],
    'decoded_frames': count, 'previews': saved, 'synthetic_source': True, 'independent_accuracy_validated': False}
(destination/'receipt.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
