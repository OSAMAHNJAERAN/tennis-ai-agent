"""Verify the frozen, never-executed response diagnosis without model inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/training/vision_upgrade/wasb_spaced_head_pilot05/manifest.json'
REPORT = ROOT / 'artifacts/validation/product_execution/wasb_proposal_response_preflight_2026-10-02.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    if REPORT.exists():
        raise FileExistsError('Preserve the existing preflight receipt.')
    source = read(SOURCE)
    mismatches = []
    hashes = {}
    for name, expected in source['provenance']['code_hashes'].items():
        actual = digest(ROOT / name)
        hashes[name] = {'expected': expected, 'actual': actual, 'matches': actual == expected}
        if actual != expected:
            mismatches.append(name)
    checkpoint = ROOT / 'artifacts/models/ball/wasb_tennis_best.pth.tar'
    checkpoint_hash = digest(checkpoint)
    if checkpoint_hash != source['provenance']['initial_checkpoint_sha256']:
        mismatches.append('checkpoint')
    original = next(item for item in source['comparators'] if item['name'] == 'original')
    counts = Counter()
    inventory = []
    for clip in original['clips']:
        dataset = ROOT / clip['dataset']
        video = dataset / 'tennis/videos' / (clip['clip'] + '.mp4')
        video_hash = digest(video)
        labels = [dataset / item['path'] for item in read(dataset / 'manifest.json')['files']
                  if item['sha256'] == clip['label_sha256']]
        label_hash = digest(labels[0]) if len(labels) == 1 else None
        if video_hash != clip['video_sha256'] or label_hash != clip['label_sha256']:
            mismatches.append('source:' + clip['clip'])
        correct_control = absent_control = False
        local = Counter()
        for row in sorted(clip['rows'], key=lambda item: item['frame']):
            target = row['target_xy']
            def error(point):
                return math.hypot((point[0] - target[0]) * 512 / clip['width'],
                                  (point[1] - target[1]) * 288 / clip['height'])
            covered = target is not None and any(error((p['x'], p['y'])) <= 4 for p in row['candidates'])
            role = None
            if target is not None and not covered:
                role = 'PROPOSAL_MISS'
            elif target is not None and not correct_control and row['prediction_xy'] is not None and error(row['prediction_xy']) <= 4:
                role = 'CORRECT_CONTROL'
                correct_control = True
            elif target is None and not absent_control:
                role = 'ABSENT_CONTROL'
                absent_control = True
            if role:
                local[role] += 1
        counts.update(local)
        inventory.append({'clip': clip['clip'], 'videoSha256': video_hash,
                          'labelSha256': label_hash, 'selectionCounts': dict(local)})
    output = ROOT / 'outputs/vision_upgrade_audit/wasb_proposal_response01'
    if output.exists():
        mismatches.append('output already exists; inspect before any replay')
    if not source['complete'] or source['advance']:
        mismatches.append('source selection state')
    if dict(counts) != {'PROPOSAL_MISS': 23, 'CORRECT_CONTROL': 12, 'ABSENT_CONTROL': 9}:
        mismatches.append('deterministic selected subset differs from frozen status')
    report = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'inferenceRun': False,
              'sourceSha256': digest(SOURCE), 'checkpointSha256': checkpoint_hash,
              'codeHashes': hashes, 'sources': inventory, 'counts': dict(counts),
              'outputAlreadyExists': output.exists(), 'freeDiskBytes': shutil.disk_usage(ROOT).free,
              'mismatches': mismatches, 'readyForFrozenReplay': not mismatches,
              'qualificationEvidence': False}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'readyForFrozenReplay': not mismatches, 'counts': dict(counts),
                      'mismatches': mismatches, 'freeDiskBytes': report['freeDiskBytes'],
                      'receipt': REPORT.relative_to(ROOT).as_posix()}))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
