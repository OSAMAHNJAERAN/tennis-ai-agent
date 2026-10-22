"""Decode exact original frames for independent, prediction-free role review."""
from pathlib import Path
import hashlib
import json

import cv2

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'artifacts/demos/sources/dimitrov_tiafoe_2024_rally.mkv'
output = ROOT / 'artifacts/validation/product_execution/active_person_original_review'
output.mkdir(parents=True, exist_ok=True)
indices = {0, 120, 240, 270, 300, 360, 480, 540}
expected_hash = '79b556aa969ca4ee28eb1ddcfe31d4f2a83dd8bd31443f9fb6e4898323c51a21'
with source.open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
if digest != expected_hash:
    raise ValueError('Original input checksum differs from the authorized diagnostic source')
cap = cv2.VideoCapture(str(source))
manifest = {'source': source.relative_to(ROOT).as_posix(), 'sourceSha256': digest,
            'frames': 600, 'fps': 60, 'scientificSplit': 'DIAGNOSTIC_PRODUCT_DEMO',
            'qualificationEvidence': False, 'extraction': 'SEQUENTIAL_ORIGINAL_DECODE_ZERO_BASED_FRAME_INDEX',
            'predictionOverlays': False, 'images': []}
try:
    for index in range(600):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Original decode failed at frame {index}')
        if index in indices:
            target = output / f'original_frame_{index:04d}.png'
            if not cv2.imwrite(str(target), frame):
                raise OSError('Unable to save original review image')
            manifest['images'].append({'frame_index': index, 'timestamp_seconds': index / 60,
                                      'path': target.relative_to(ROOT).as_posix(),
                                      'width': frame.shape[1], 'height': frame.shape[0],
                                      'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    if cap.read()[0]:
        raise ValueError('Original input contains more than the pinned 600 frames')
finally:
    cap.release()
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print(json.dumps(manifest, indent=2))
