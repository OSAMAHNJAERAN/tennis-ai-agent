"""Freeze a systematic development review sample without reading predictions.

This bounded label-protocol pilot is not a full-clip or final accuracy test.
All reviewed sources already belong to the reused public diagnostic split.
"""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

import cv2

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / 'artifacts/validation/product_execution/public_ball_blind_review01'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'data/demos/sources.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol_path = TARGET / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        for source in protocol['sources']:
            if digest(ROOT / source['inputPath']) != source['inputSha256']:
                raise ValueError('Original review source changed.')
            for item in source['frames']:
                if digest(ROOT / item['path']) != item['sha256']:
                    raise ValueError('Previously frozen review PNG changed.')
        print('Existing frozen sample verified; no re-sampling or prediction access.')
        return
    rng = random.Random(20261001)
    protocol = {'purpose': 'PREDICTION_BLIND_LABEL_PROTOCOL_PILOT; REUSED_DEVELOPMENT; NOT_QUALIFICATION',
        'seed': 20261001, 'sampling': '12 equal decoded-frame strata per source; one seeded uniform center per stratum; context at center±3 frames.',
        'samplePerSource': 12, 'sourceCount': 3,
        'labels': {'visibility': ['VISIBLE', 'CONFIRMED_ABSENT', 'UNCERTAIN'],
                   'center_px': 'Native image ball center/streak midpoint only when VISIBLE; null otherwise.',
                   'uncertainty_radius_px': 'Rough annotation estimate; not a statistical confidence interval.',
                   'absence': 'Only positive visual evidence of outside view/fully absent; failure to find a tiny ball is UNCERTAIN.',
                   'context': 'Read both adjacent originals; do not turn uncertainty into absence.',
                   'non_ball_features': 'Optional bounded regions/centers of clearly visible distracting objects; descriptive only, not a ball-exclusion mask.'},
        'comparison': {'centerTolerancePx': 8,
                       'visibleLabelEligibility': 'Only visible centers with estimated uncertainty <=4 native pixels; others reviewed but excluded from numeric metrics.',
                       'wrongVisibleDetection': 'Both FP and FN; abstention on visible label is FN; confirmed absent detection FP.',
                       'unresolved': 'Excluded, counted and individually retained; no extrapolation to full-clip precision.',
                       'candidate': 'No training, threshold selection or production mutation. Existing frozen detour replay only after labels contain clear negative and true-ball controls.'},
        'manifestSha256': digest(manifest_path), 'preparationScriptSha256': digest(Path(__file__)),
        'predictionsReadDuringPreparation': False, 'sources': []}
    for entry in manifest['sources']:
        if entry['qualificationEvidence'] is not False:
            raise ValueError('Only reused diagnostic sources may enter this pilot.')
        source = ROOT / entry['inputPath']
        if digest(source) != entry['inputSha256']:
            raise ValueError('Original input changed.')
        count = entry['inputVideo']['frames']
        rate = Fraction(entry['inputVideo']['stream']['avg_frame_rate'])
        centers = [rng.randrange(index * count // 12, (index + 1) * count // 12) for index in range(12)]
        selected = sorted({max(0, min(count - 1, index + offset)) for index in centers for offset in (-3, 0, 3)})
        folder = TARGET / entry['id']
        folder.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError('Original input cannot be decoded.')
        records = []
        try:
            for index in range(count):
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError('Original decoded frame count changed.')
                if index in selected:
                    destination = folder / f'original_{index:04d}.png'
                    if destination.exists():
                        raise ValueError('Unrecorded review frame exists; inspect before retrying.')
                    if not cv2.imwrite(str(destination), frame):
                        raise RuntimeError('Could not preserve exact original pixels.')
                    records.append({'frame_index': index, 'relative_timestamp_seconds': float(index / rate),
                                    'path': destination.relative_to(ROOT).as_posix(), 'sha256': digest(destination),
                                    'width': frame.shape[1], 'height': frame.shape[0]})
            if capture.read()[0]:
                raise ValueError('Original contains additional decoded frames.')
        finally:
            capture.release()
        protocol['sources'].append({'id': entry['id'], 'inputPath': entry['inputPath'],
            'inputSha256': entry['inputSha256'], 'decodedFrames': count, 'fpsRational': str(rate),
            'centers': centers, 'frames': records})
    protocol_path.write_text(json.dumps(protocol, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'frozenSources': len(protocol['sources']), 'centerFrames': 36,
        'originalContextImages': sum(len(source['frames']) for source in protocol['sources']),
        'protocolSha256': digest(protocol_path), 'predictionsRead': False}))


if __name__ == '__main__':
    main()
