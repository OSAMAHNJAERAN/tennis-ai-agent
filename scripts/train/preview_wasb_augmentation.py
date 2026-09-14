"""Render synthetic training graphics for visual review, using training data only."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from scripts.train.finetune_wasb import add_stationary_graphic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    dataset = Path(manifest['arguments']['dataset'])
    if json.loads((dataset / 'manifest.json').read_text())['split'] != 'TRAINING_ONLY':
        raise ValueError('Preview requires verified training samples')
    samples = json.loads((args.run / 'training_samples.json').read_text())
    selected = [samples[0], samples[50], next(sample for sample in samples if sample['target'] is None)]
    canvas = np.full((3 * 324, 1024, 3), 24, np.uint8)
    yy, xx = np.mgrid[:288, :512]
    for index, sample in enumerate(selected):
        source = cv2.imread(sample['context_paths'][2])
        if source is None:
            raise ValueError('Cached training frame failed to decode')
        target = np.zeros((288, 512), np.float32)
        if sample['target'] is not None:
            x, y = sample['target']
            target[(xx - x) ** 2 + (yy - y) ** 2 <= 2.5 ** 2] = 1
        modified = add_stationary_graphic([source] * 3, target, random.Random(index + 7))[0]
        original = source.copy()
        for image in (original, modified):
            if sample['target'] is not None:
                cv2.circle(image, tuple(map(round, sample['target'])), 8, (30, 255, 30), 1, cv2.LINE_AA)
        y = index * 324
        canvas[y:y + 288, :512] = original
        canvas[y:y + 288, 512:] = modified
        text = f"TRAIN {sample['clip']} f{sample['frame']} / " + ('VISIBLE' if sample['target'] else 'ABSENT')
        cv2.putText(canvas, text, (8, y + 311), cv2.FONT_HERSHEY_SIMPLEX, .42, (230, 230, 230), 1)
        cv2.putText(canvas, 'SYNTHETIC GRAPHIC / TARGET UNCHANGED', (525, y + 311), cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError('Preview write failed')


if __name__ == '__main__':
    main()
