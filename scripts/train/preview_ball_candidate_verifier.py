"""Review real training positives or true validation candidates rejected by the verifier."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.train.train_ball_candidate_verifier import load_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, patches, features, _ = load_cache(args.cache)
    evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None
    if evaluation and evaluation['cache_manifest_sha256'] != digest(args.cache / 'manifest.json'):
        raise ValueError('Evaluation and crop cache differ')
    selected, per_clip = [], Counter()
    for row in manifest['rows']:
        correct = [candidate for candidate in row['candidates'] if candidate['training_target'] == 1]
        if not correct or per_clip[row['clip']] >= 2:
            continue
        if evaluation:
            candidate = max(correct, key=lambda value: evaluation['candidate_scores'][value['sample_index']])
            score = evaluation['candidate_scores'][candidate['sample_index']]
            if score >= evaluation['threshold']:
                continue
        else:
            candidate, score = correct[0], None
        selected.append({'clip': row['clip'], 'frame': row['frame'], **candidate, 'verifier_score': score})
        per_clip[row['clip']] += 1
        if len(selected) == 24:
            break
    if not selected:
        raise ValueError('No requested examples')
    canvas = np.full((((len(selected) + 3) // 4) * 200, 4 * 280, 3), 24, dtype=np.uint8)
    for number, item in enumerate(selected):
        patch = patches[item['sample_index']]
        y, x = number // 4 * 200, number % 4 * 280
        for offset, start in ((0, 0), (136, 3)):
            image = cv2.cvtColor(patch[start:start + 3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            canvas[y:y + 128, x + offset:x + offset + 128] = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
        score = f'{item["verifier_score"]:.3f}' if item['verifier_score'] is not None else 'training'
        for offset, text in ((146, f'{item["clip"]} f{item["frame"]}'),
                             (169, f'WASB {item["confidence"]:.3f}, verifier {score}'),
                             (192, 'Current RGB | Past RGB')):
            cv2.putText(canvas, text, (x + 3, y + offset), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 235, 235), 1)
    args.output.mkdir(parents=True)
    if not cv2.imwrite(str(args.output / 'candidate_crops.jpg'), canvas):
        raise RuntimeError('Crop review write failed')
    (args.output / 'review.json').write_text(json.dumps({'cache_manifest_sha256': digest(args.cache / 'manifest.json'),
                                                       'evaluation_sha256': digest(args.evaluation) if args.evaluation else None,
                                                       'selection': 'FIRST_TWO_QUALIFYING_CANDIDATES_PER_CLIP; MAX_24',
                                                       'examples': selected}, indent=2), encoding='utf-8')
    print(f'Reviewed {len(selected)} candidate crops')


if __name__ == '__main__':
    main()
