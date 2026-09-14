"""Bounded WASB adaptation on publisher training labels, with sparse supervision.

Only explicitly labeled output frames contribute to loss. Adjacent frames supply
visual context, never invented labels. Validation/test data are not loaded here.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.detection.wasb_ball_detector import WASBBallDetector
from src.utils.video_frame_sequence import VideoFrameSequence


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(dataset, output, detector):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest['split'] != 'TRAINING_ONLY':
        raise ValueError('Fine-tuning accepts explicitly designated training data only')
    for item in manifest['files']:
        if sha256(dataset / item['path']) != item['sha256']:
            raise ValueError(f"Training source changed: {item['path']}")
    samples = []
    for match, rally in manifest['selected_clips']:
        with (dataset / f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as stream:
            labels = list(csv.DictReader(stream))
        cache = output / 'frame_cache' / f'{match}_{rally}'
        cache.mkdir(parents=True)
        with VideoFrameSequence(str(dataset / f'tennis/videos/{match}_{rally}.mp4')) as frames:
            height, width = frames.metadata.height, frames.metadata.width
            center = np.array([width / 2, height / 2], dtype=np.float32)
            transform = detector.geometry.get_affine_transform(center, max(height, width), 0, (512, 288))
            needed = set()
            seen = set()
            for label in labels:
                index, visible = int(label['Frame']), int(label['Visibility'])
                if index in seen or not 0 <= index < len(frames) or visible not in (0, 1):
                    raise ValueError(f'Invalid training label: {match} frame {index}')
                seen.add(index)
                indices = [min(len(frames) - 1, max(0, index + offset)) for offset in range(-2, 3)]
                point = np.array([float(label['X']) * width / 1920, float(label['Y']) * height / 1080])
                target = detector.geometry.affine_transform(point, transform).tolist() if visible else None
                if target is not None and not (0 <= target[0] < 512 and 0 <= target[1] < 288):
                    raise ValueError('Visible training target is outside transformed image')
                samples.append({'clip': f'{match}_{rally}', 'frame': index, 'target': target,
                                'context_paths': [str(cache / f'{number:06}.jpg') for number in indices]})
                needed.update(indices)
            for index in sorted(needed):
                image = cv2.warpAffine(frames[index], transform, (512, 288), flags=cv2.INTER_LINEAR)
                if not cv2.imwrite(str(cache / f'{index:06}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError('Training cache write failed')
        print(f'Prepared {match}_{rally}: {len(labels)} explicit labels, {len(needed)} context frames', flush=True)
    return samples


def add_stationary_graphic(images, target, rng=random):
    """Add one shared synthetic broadcast graphic without covering the GT ball.

    The small yellow service marker is a negative visual distractor, never a
    generated ball label. All temporal inputs receive the identical graphic.
    """
    height, width = target.shape
    panel_width, panel_height = rng.randint(76, 110), rng.randint(28, 42)
    margin_x, margin_y = rng.randint(8, 24), rng.randint(8, 24)
    positions = [(margin_x, margin_y), (width - margin_x - panel_width, margin_y),
                 (margin_x, height - margin_y - panel_height),
                 (width - margin_x - panel_width, height - margin_y - panel_height)]
    rng.shuffle(positions)
    for left, top in positions:
        right, bottom = left + panel_width, top + panel_height
        if target[max(0, top - 8):min(height, bottom + 8), max(0, left - 8):min(width, right + 8)].any():
            continue
        graphic = np.full((panel_height, panel_width, 3), (30, 22, 16), dtype=np.uint8)
        text_size = .25 + rng.random() * .1
        cv2.putText(graphic, 'PLAYER A  40', (11, 12), cv2.FONT_HERSHEY_SIMPLEX, text_size, (225, 225, 225), 1, cv2.LINE_AA)
        cv2.putText(graphic, 'PLAYER B  30', (11, panel_height - 5), cv2.FONT_HERSHEY_SIMPLEX, text_size, (225, 225, 225), 1, cv2.LINE_AA)
        marker_y = rng.choice([9, panel_height - 8])
        cv2.circle(graphic, (5, marker_y), rng.choice([1, 2]), (40, 235, 225), -1, cv2.LINE_AA)
        augmented = [image.copy() for image in images]
        for image in augmented:
            image[top:bottom, left:right] = graphic
        return augmented
    return images


class SparseBallDataset(Dataset):
    def __init__(self, samples, graphic_probability=0.):
        if not 0 <= graphic_probability <= 1:
            raise ValueError('Graphic probability must lie in [0,1]')
        self.samples = samples
        self.graphic_probability = graphic_probability
        self.yy, self.xx = np.mgrid[:288, :512]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        slot = random.randrange(3)
        paths = sample['context_paths'][2 - slot:5 - slot]
        images = [cv2.imread(path) for path in paths]
        if any(image is None for image in images):
            raise ValueError('Training context failed to decode')
        target = np.zeros((288, 512), dtype=np.float32)
        if sample['target'] is not None:
            x, y = sample['target']
            target[((self.xx - x) ** 2 + (self.yy - y) ** 2) <= 2.5 ** 2] = 1
        if self.graphic_probability and random.random() < self.graphic_probability:
            images = add_stationary_graphic(images, target)
        if random.random() < .5:
            images = [cv2.flip(image, 1) for image in images]
            target = np.fliplr(target).copy()
        brightness = random.uniform(.8, 1.2)
        contrast = random.uniform(.9, 1.1)
        kernel = None
        if random.random() < .3:
            size = random.choice([3, 5])
            kernel = np.zeros((size, size), dtype=np.float32)
            if random.random() < .5:
                kernel[size // 2, :] = 1 / size
            else:
                kernel[:, size // 2] = 1 / size
        channels = []
        for image in images:
            if kernel is not None:
                image = cv2.filter2D(image, -1, kernel)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            image = np.clip((image - .5) * contrast + .5, 0, 1) * brightness
            image = np.clip(image, 0, 1)
            image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            channels.append(image.transpose(2, 0, 1))
        return torch.from_numpy(np.concatenate(channels)), torch.from_numpy(target), slot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('data/external/racketvision_training20'))
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/ball/wasb_tennis_best.pth.tar'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--scope', choices=['head', 'last_stage'], default='last_stage')
    parser.add_argument('--graphic-probability', type=float, default=0.)
    parser.add_argument('--absent-sampling-factor', type=float, default=1.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.epochs <= 10 or not 1 <= args.batch_size <= 4:
        raise ValueError('Pilot budget: 1–10 epochs, batch size 1–4')
    if not 0 <= args.graphic_probability <= 1 or not 1 <= args.absent_sampling_factor <= 10:
        raise ValueError('Require graphic probability 0–1 and absent sampling factor 1–10')
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = WASBBallDetector(args.checkpoint, device=device)
    args.output.mkdir(parents=True)
    samples = prepare(args.dataset, args.output, detector)
    (args.output / 'training_samples.json').write_text(json.dumps(samples, indent=2))
    model = detector.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith('final_layers.') or (args.scope == 'last_stage' and name.startswith('stage4.'))
    # Preserve pretrained batch-normalization statistics for a small-batch pilot.
    model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    sampler = (WeightedRandomSampler([args.absent_sampling_factor if sample['target'] is None else 1.
                                      for sample in samples], num_samples=len(samples), replacement=True)
               if args.absent_sampling_factor != 1 else None)
    loader = DataLoader(SparseBallDataset(samples, args.graphic_probability), batch_size=args.batch_size,
                        shuffle=sampler is None, sampler=sampler, num_workers=0)
    run = {'schema_version': '1.0', 'qualification_evidence': False, 'purpose': 'BOUNDED_TRAINING_PILOT',
           'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
           'initial_checkpoint_sha256': sha256(args.checkpoint), 'dataset_manifest_sha256': sha256(args.dataset / 'manifest.json'),
           'script_sha256': sha256(__file__), 'seed': 7, 'training_samples': len(samples),
           'explicit_absent_samples': sum(sample['target'] is None for sample in samples),
           'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
           'supervision': 'ONE_EXPLICIT_FRAME_PER_TRIPLET; RANDOM_OUTPUT_SLOT; NO_PSEUDO_LABELS',
           'augmentation': {'horizontal_flip_probability': .5, 'brightness_factor': [.8, 1.2], 'contrast_factor': [.9, 1.1],
                            'motion_blur_probability': .3, 'motion_blur_kernel_sizes': [3, 5], 'cache_jpeg_quality': 95,
                            'stationary_graphic_probability': args.graphic_probability,
                            'stationary_graphic_semantics': 'SYNTHETIC_NEGATIVE_OVERLAY_SHARED_ACROSS_TRIPLET; GT_BALL_REGION_PRESERVED'},
           'sampling': {'absent_factor': args.absent_sampling_factor, 'replacement': sampler is not None,
                        'draws_per_epoch': len(samples), 'labels_reassigned': False},
           'batchnorm': 'FROZEN_PRETRAINED_STATISTICS', 'epochs': []}
    (args.output / 'manifest.json').write_text(json.dumps(run, indent=2))
    for epoch in range(1, args.epochs + 1):
        start, loss_sum, count, absent_draws = time.perf_counter(), 0., 0, 0
        for iteration, (frames, target, slot) in enumerate(loader, 1):
            absent_draws += int((target.flatten(1).sum(1) == 0).sum())
            frames, target, slot = frames.to(device), target.to(device), slot.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = model(frames)[0]
                selected = logits[torch.arange(len(slot), device=device), slot]
                # Stable gamma-two focal BCE corresponding to upstream WBCE.
                bce = F.binary_cross_entropy_with_logits(selected.float(), target, reduction='none')
                probability = selected.float().sigmoid()
                pt = target * probability + (1 - target) * (1 - probability)
                loss = ((1 - pt).square() * bce).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(slot)
            count += len(slot)
            if iteration % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': iteration, 'mean_training_loss': loss_sum / count}), flush=True)
        checkpoint = args.output / f'epoch_{epoch:02}.pth.tar'
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        report = {'epoch': epoch, 'mean_training_loss': loss_sum / count, 'seconds': time.perf_counter() - start,
                  'observed_absent_draws': absent_draws, 'observed_visible_draws': count - absent_draws,
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint)}
        run['epochs'].append(report)
        (args.output / 'manifest.json').write_text(json.dumps(run, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
