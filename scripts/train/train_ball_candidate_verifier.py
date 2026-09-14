"""Train a bounded candidate verifier, selecting only on held-out training clips."""

import argparse
import json
from pathlib import Path
import random
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.detection.ball_candidate_verifier import BallCandidateVerifier
from src.evaluation.point_metrics import evaluate_points
from scripts.train.verifier_negative_review import reviewed_negative_indices


def load_cache(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'candidates.npz') != manifest['cache_sha256']:
        raise ValueError('Candidate cache checksum mismatch')
    with np.load(directory / 'candidates.npz', allow_pickle=False) as values:
        patches, features, targets = values['patches'], values['features'], values['targets']
    count = len(targets)
    if (patches.shape != (count, 6, 32, 32) or patches.dtype != np.uint8
            or features.shape != (count, 2) or not np.isfinite(features).all()
            or not np.isin(targets, [-1, 0, 1]).all()):
        raise ValueError('Invalid candidate cache arrays')
    seen, indices = set(), set()
    for row in manifest['rows']:
        key = row['clip'], row['frame']
        if key in seen:
            raise ValueError('Duplicate labeled frame')
        seen.add(key)
        for candidate in row['candidates']:
            index = candidate['sample_index']
            if index in indices or not 0 <= index < count or candidate['training_target'] != targets[index]:
                raise ValueError('Candidate mapping mismatch')
            indices.add(index)
    if indices != set(range(count)):
        raise ValueError('Candidate mapping is incomplete')
    return manifest, patches, features, targets


def predict_scores(model, patches, features, indices, device):
    scores = np.full(len(patches), np.nan, dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            batch = indices[start:start + 256]
            images = torch.from_numpy(patches[batch]).to(device).float() / 255
            metadata = torch.from_numpy(features[batch]).to(device)
            values = model(images, metadata).sigmoid().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite verifier scores')
            scores[batch] = values
    return scores


def score_rows(rows, scores, threshold):
    if not 0 < threshold < 1:
        raise ValueError('Verifier threshold must lie in (0,1)')
    predictions = []
    for row in rows:
        candidates = row['candidates']
        if any(not np.isfinite(scores[candidate['sample_index']]) for candidate in candidates):
            raise ValueError('Missing score for an evaluated candidate')
        chosen = max(candidates, key=lambda candidate: float(scores[candidate['sample_index']]), default=None)
        prediction = ([chosen['x'], chosen['y']] if chosen is not None
                      and scores[chosen['sample_index']] >= threshold else None)
        predictions.append({key: row[key] for key in ('clip', 'frame', 'width', 'height', 'target_xy')}
                           | {'prediction_xy': prediction})
    return evaluate_points(predictions), predictions


class TrainingCandidates(Dataset):
    def __init__(self, patches, features, targets, indices, motion_blur_probability=0.):
        self.patches, self.features, self.targets, self.indices = patches, features, targets, indices
        self.motion_blur_probability = motion_blur_probability
        self.blur_random = random.Random(17)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image = torch.from_numpy(self.patches[index].copy()).float() / 255
        if random.random() < .5:
            image = torch.flip(image, (-1,))
        if self.motion_blur_probability and self.blur_random.random() < self.motion_blur_probability:
            size = self.blur_random.choice([3, 5])
            radius = size // 2
            horizontal = self.blur_random.random() < .5
            padding, kernel = ((radius, radius, 0, 0), (1, size)) if horizontal else ((0, 0, radius, radius), (size, 1))
            image = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(image, padding, mode='reflect'), kernel, stride=1)
        contrast, brightness = random.uniform(.8, 1.2), random.uniform(.8, 1.2)
        image = (((image - .5) * contrast + .5) * brightness).clamp(0, 1)
        return image, torch.from_numpy(self.features[index]), float(self.targets[index])


def training_partition(manifest):
    if manifest['split'] != 'TRAINING_ONLY':
        raise ValueError('Verifier training refuses validation/test candidate caches')
    clips = [f'{match}_{rally}' for match, rally in manifest['selected_clips']]
    if len(clips) != 20 or len(set(clips)) != len(clips):
        raise ValueError('This predeclared pilot requires twenty distinct training clips')
    train_clips, heldout_clips = clips[:-4], clips[-4:]
    if any(row['clip'] not in clips for row in manifest['rows']):
        raise ValueError('Candidate rows do not belong to the declared clips')
    return train_clips, heldout_clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--motion-blur-probability', type=float, default=0.)
    parser.add_argument('--appearance-only', action='store_true')
    parser.add_argument('--negative-quality-review', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.motion_blur_probability <= 1:
        raise ValueError('Motion blur probability must lie in [0,1]')
    manifest, patches, features, targets = load_cache(args.cache)
    if args.appearance_only:
        features = np.zeros_like(features)
    train_clips, heldout_clips = training_partition(manifest)
    train_rows = [row for row in manifest['rows'] if row['clip'] in train_clips]
    heldout_rows = [row for row in manifest['rows'] if row['clip'] in heldout_clips]
    train_indices = np.array([candidate['sample_index'] for row in train_rows for candidate in row['candidates']
                              if candidate['training_target'] >= 0], dtype=np.int64)
    heldout_indices = np.array([candidate['sample_index'] for row in heldout_rows for candidate in row['candidates']], dtype=np.int64)
    excluded, quality_provenance = set(), None
    if args.negative_quality_review:
        review = json.loads(args.negative_quality_review.read_text())
        if review['cache_manifest_sha256'] != digest(args.cache / 'manifest.json'):
            raise ValueError('Negative review belongs to a different candidate cache')
        for field in ('selection', 'render'):
            if digest(review[field + '_path']) != review[field + '_sha256']:
                raise ValueError('Negative review evidence changed')
        selection = json.loads(Path(review['selection_path']).read_text())
        expected = [(r['clip'], r['frame'], r['candidate']) for r in selection['selected_candidates']]
        actual = [(r['clip'], r['frame'], r['candidate']) for r in review['reviews']]
        if expected != actual:
            raise ValueError('Review does not cover the frozen selection exactly')
        excluded = reviewed_negative_indices(manifest, review)
        quality_provenance = {'review_path': str(args.negative_quality_review.resolve()),
                              'review_sha256': digest(args.negative_quality_review),
                              'excluded_sample_indices': sorted(excluded),
                              'policy': review['policy']}
    included = np.array([index not in excluded for index in train_indices], dtype=bool)
    counts = np.bincount(targets[train_indices[included]], minlength=2)
    if not counts.all() or not len(heldout_rows):
        raise ValueError('Training needs both classes and held-out labeled frames')
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BallCandidateVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    weights = np.where(included, 1. / counts[targets[train_indices]], 0.)
    sampler = WeightedRandomSampler(weights.tolist(), len(train_indices), replacement=True)
    loader = DataLoader(TrainingCandidates(patches, features, targets, train_indices, args.motion_blur_probability), batch_size=64,
                        sampler=sampler, num_workers=0)
    run = {'schema_version': '1.0', 'qualification_evidence': False, 'purpose': 'BOUNDED_CANDIDATE_VERIFIER_PILOT',
           'training_cache': str(args.cache), 'cache_manifest_sha256': digest(args.cache / 'manifest.json'),
           'cache_sha256': manifest['cache_sha256'], 'preparation_configuration': manifest['configuration'],
           'base_checkpoint_sha256': manifest['checkpoint_sha256'],
           'trainer_sha256': digest(__file__), 'verifier_module_sha256': digest(ROOT / 'src/detection/ball_candidate_verifier.py'),
           'seed': 7, 'device': device, 'epochs_requested': 8, 'batch_size': 64, 'learning_rate': .001,
           'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
           'negative_quality_review': quality_provenance,
           'negative_review_validator_sha256': digest(ROOT / 'scripts/train/verifier_negative_review.py'),
           'original_training_class_counts_negative_positive': np.bincount(targets[train_indices], minlength=2).tolist(),
           'eligible_training_candidates': int(included.sum()),
           'zero_weight_training_candidates': int((weights == 0).sum()),
           'parameters': sum(parameter.numel() for parameter in model.parameters()),
           'appearance_only': args.appearance_only,
           'train_clips': train_clips, 'internal_heldout_training_clips': heldout_clips,
           'training_class_counts_negative_positive': counts.tolist(), 'training_draws_per_epoch': len(train_indices),
           'training_sampling': 'BALANCED_REPLACEMENT; LABELS_UNCHANGED; AMBIGUOUS_CANDIDATES_OMITTED_FROM_LOSS',
           'augmentation': {'horizontal_flip_probability': .5, 'brightness_factor': [.8, 1.2],
                            'contrast_factor': [.8, 1.2], 'motion_blur_probability': args.motion_blur_probability,
                            'motion_blur_kernel_sizes': [3, 5], 'motion_blur_directions': ['horizontal', 'vertical'],
                            'shared_across_current_past': True, 'independent_blur_random_seed': 17},
           'selection': 'HIGHEST_STRICT_FRAME_F1_THEN_PRECISION_THEN_RECALL_ON_INTERNAL_TRAINING_CLIPS',
           'threshold_grid': [.3, .5, .7, .9], 'epochs': []}
    args.output.mkdir(parents=True)
    best_key, best = (-1., -1., -1.), None
    for epoch in range(1, 9):
        started, total_loss, total, positive_draws = time.perf_counter(), 0., 0, 0
        model.train()
        for images, metadata, labels in loader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images, metadata), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite verifier training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            total += len(labels)
            positive_draws += int(labels.sum().item())
        checkpoint = args.output / f'epoch_{epoch:02}.pt'
        torch.save({'model_state_dict': model.state_dict(), 'architecture': 'BallCandidateVerifier_v1',
                    'epoch': epoch, 'base_checkpoint_sha256': manifest['checkpoint_sha256']}, checkpoint)
        checkpoint_hash = digest(checkpoint)
        scores = predict_scores(model, patches, features, heldout_indices, device)
        evaluations = []
        for threshold in run['threshold_grid']:
            metrics, predictions = score_rows(heldout_rows, scores, threshold)
            evaluations.append({'threshold': threshold, 'metrics': metrics})
            key = tuple(metrics[name] if metrics[name] is not None else -1. for name in ('f1', 'precision', 'recall'))
            if key > best_key:
                best_key = key
                best = {'epoch': epoch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
                        'threshold': threshold, 'internal_heldout_metrics': metrics, 'labeled_rows': predictions}
        result = {'epoch': epoch, 'mean_training_loss': total_loss / total, 'positive_draws': positive_draws,
                  'seconds_including_selection': time.perf_counter() - started,
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash, 'evaluations': evaluations}
        run['epochs'].append(result)
        run['selected'] = best
        (args.output / 'manifest.json').write_text(json.dumps(run, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'epoch': epoch, 'loss': result['mean_training_loss'], 'seconds': result['seconds_including_selection'],
                          'selected_epoch': best['epoch'], 'selected_threshold': best['threshold'],
                          'internal_f1': best['internal_heldout_metrics']['f1']}), flush=True)


if __name__ == '__main__':
    main()
