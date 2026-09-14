"""Evaluate a frozen training-selected verifier on separately prepared validation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.summarize_ball_validation import summarize
from scripts.train.train_ball_candidate_verifier import load_cache, predict_scores, score_rows
from src.detection.ball_candidate_verifier import BallCandidateVerifier
from src.evaluation.point_metrics import evaluate_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, patches, features, _ = load_cache(args.cache)
    if manifest['split'] != 'VALIDATION_ONLY':
        raise ValueError('External evaluation requires a validation cache')
    training = json.loads((args.training_run / 'manifest.json').read_text())
    if training.get('appearance_only', False):
        features = np.zeros_like(features)
    trained_clips = set(training['train_clips'] + training['internal_heldout_training_clips'])
    if any(row['clip'] in trained_clips for row in manifest['rows']):
        raise ValueError('Validation clip overlaps verifier training/selection')
    if (manifest['configuration'] != training['preparation_configuration']
            or manifest['checkpoint_sha256'] != training['base_checkpoint_sha256']):
        raise ValueError('Candidate preparation differs from training protocol')
    selected = training['selected']
    checkpoint = Path(selected['checkpoint'])
    if digest(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum changed')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallCandidateVerifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model_state_dict'], strict=True)
    scores = predict_scores(model, patches, features, np.arange(len(patches)), device)
    metrics, rows = score_rows(manifest['rows'], scores, selected['threshold'])
    baseline_rows = [{key: row[key] for key in ('clip', 'frame', 'width', 'height', 'target_xy')}
                     | {'prediction_xy': [row['candidates'][0]['x'], row['candidates'][0]['y']] if row['candidates'] else None}
                     for row in manifest['rows']]
    result = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'FROZEN_TRAINING_SELECTED_VERIFIER; STRICT_SPARSE_VALIDATION; NO_NEW_POSITIONS',
              'cache_manifest_sha256': digest(args.cache / 'manifest.json'),
              'training_manifest_sha256': digest(args.training_run / 'manifest.json'),
              'selected_checkpoint_sha256': selected['checkpoint_sha256'], 'threshold': selected['threshold'],
              'appearance_only': training.get('appearance_only', False),
              'evaluator_sha256': digest(__file__),
              'verifier_module_sha256': digest(ROOT / 'src/detection/ball_candidate_verifier.py'),
              'prediction_helper_sha256': digest(ROOT / 'scripts/train/train_ball_candidate_verifier.py'),
              'baseline_raw_top1_metrics': evaluate_points(baseline_rows), 'metrics': metrics,
              'strata': summarize(rows), 'labeled_rows': rows, 'candidate_scores': scores.tolist(),
              'source_broadcast_independence': manifest['source_broadcast_independence']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: value for key, value in metrics.items() if key != 'localization_errors_reference_px'}))


if __name__ == '__main__':
    main()
