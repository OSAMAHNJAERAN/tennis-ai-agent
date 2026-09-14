"""Train a bounded spatial-view WASB candidate using publisher training data only.

Each original labeled frame has five real-image views. Training samples one view
uniformly; internal selection evaluates all views without image augmentation.
"""

import argparse
import csv
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
from torch.utils.data import DataLoader, Dataset

from scripts.train.finetune_wasb import SparseBallDataset, sha256
from src.detection.tiled_wasb_candidates import overlapping_tiles
from src.detection.wasb_ball_detector import WASBBallDetector
from src.utils.video_frame_sequence import VideoFrameSequence


def split_training_clips(manifests):
    """Every fifth publisher entry is internal selection; never a validation set."""
    clips, seen, revision = [], set(), None
    for manifest in manifests:
        if manifest['split'] != 'TRAINING_ONLY':
            raise ValueError('Spatial adaptation accepts training manifests only')
        if revision is None:
            revision = manifest['revision']
        if manifest['revision'] != revision:
            raise ValueError('Training manifests must share a pinned revision')
        for match, rally in manifest['selected_clips']:
            key = (match, rally)
            if key in seen:
                raise ValueError('Duplicate training clip')
            seen.add(key)
            clips.append(key)
    # Split by publisher match ID, keeping all rallies of a match together.
    matches = list(dict.fromkeys(match for match, _ in clips))
    if len(matches) < 10:
        raise ValueError('Require at least ten match IDs for internal selection')
    held = set(matches[4::5])
    return {f'{match}_{rally}': 'selection' if match in held else 'train'
            for match, rally in clips}


def crop_target(point, box, geometry):
    """Transform the explicitly labeled active ball; outside-view means absent."""
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    transform = geometry.get_affine_transform(
        np.array([width / 2, height / 2], np.float32), max(width, height), 0, (512, 288))
    if point is None or not (left <= point[0] < right and top <= point[1] < bottom):
        return None, transform
    target = geometry.affine_transform(np.array([point[0] - left, point[1] - top]), transform)
    if not (0 <= target[0] < 512 and 0 <= target[1] < 288):
        raise ValueError('Visible transformed target falls outside view')
    return target.tolist(), transform


def prepare_views(datasets, output, detector):
    manifests = [json.loads((dataset / 'manifest.json').read_text()) for dataset in datasets]
    split = split_training_clips(manifests)
    samples = []
    for dataset, manifest in zip(datasets, manifests, strict=True):
        for item in manifest['files']:
            if sha256(dataset / item['path']) != item['sha256']:
                raise ValueError(f"Training source changed: {item['path']}")
        publisher_train = {tuple(item) for item in json.loads((dataset / 'tennis/info/train.json').read_text())}
        publisher_val = {tuple(item) for item in json.loads((dataset / 'tennis/info/val.json').read_text())}
        selected = {tuple(item) for item in manifest['selected_clips']}
        if not selected <= publisher_train or {m for m, _ in selected} & {m for m, _ in publisher_val}:
            raise ValueError('Manifest selection is not isolated publisher training data')
        for match, rally in manifest['selected_clips']:
            clip = f'{match}_{rally}'
            with (dataset / f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as stream:
                labels = list(csv.DictReader(stream))
            with VideoFrameSequence(str(dataset / f'tennis/videos/{clip}.mp4')) as frames:
                width, height = frames.metadata.width, frames.metadata.height
                boxes = [(0, 0, width, height), *overlapping_tiles(width, height, .6)]
                caches = [output / 'frame_cache' / clip / f'view_{i}' for i in range(5)]
                for cache in caches:
                    cache.mkdir(parents=True)
                seen, needed = set(), set()
                transforms = [crop_target(None, box, detector.geometry)[1] for box in boxes]
                for label in labels:
                    index, visible = int(label['Frame']), int(label['Visibility'])
                    if index in seen or not 0 <= index < len(frames) or visible not in (0, 1):
                        raise ValueError('Invalid explicit training label')
                    seen.add(index)
                    indices = [min(len(frames) - 1, max(0, index + delta)) for delta in range(-2, 3)]
                    needed.update(indices)
                    point = [float(label['X']) * width / 1920, float(label['Y']) * height / 1080] if visible else None
                    if point is not None and (not np.isfinite(point).all() or not
                                             (0 <= point[0] < width and 0 <= point[1] < height)):
                        raise ValueError('Invalid source ball coordinate')
                    views = []
                    for box, cache in zip(boxes, caches, strict=True):
                        target, _ = crop_target(point, box, detector.geometry)
                        views.append({'target': target,
                                      'context_paths': [str(cache / f'{number:06}.jpg') for number in indices]})
                    samples.append({'clip': clip, 'frame': index, 'split': split[clip],
                                    'source_visible': bool(visible), 'views': views})
                for index in sorted(needed):
                    frame = frames[index]
                    for box, transform, cache in zip(boxes, transforms, caches, strict=True):
                        left, top, right, bottom = box
                        image = cv2.warpAffine(frame[top:bottom, left:right], transform, (512, 288))
                        if not cv2.imwrite(str(cache / f'{index:06}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                            raise RuntimeError('Spatial training cache write failed')
            print(json.dumps({'prepared': clip, 'split': split[clip], 'labels': len(labels)}), flush=True)
    return samples


class RandomSpatialView(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.dataset = SparseBallDataset([view for sample in samples for view in sample['views']])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.dataset[index * 5 + random.randrange(5)]


class SelectionViews(Dataset):
    """Deterministic central output and real +/-1 context, no augmentation."""
    def __init__(self, samples):
        self.views = [view for sample in samples for view in sample['views']]
        self.yy, self.xx = np.mgrid[:288, :512]

    def __len__(self):
        return len(self.views)

    def __getitem__(self, index):
        view = self.views[index]
        channels = []
        for path in view['context_paths'][1:4]:
            image = cv2.imread(path)
            if image is None:
                raise ValueError('Selection context failed to decode')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
            channels.append(image.transpose(2, 0, 1))
        target = np.zeros((288, 512), np.float32)
        if view['target'] is not None:
            x, y = view['target']
            target[(self.xx - x) ** 2 + (self.yy - y) ** 2 <= 2.5 ** 2] = 1
        return torch.from_numpy(np.concatenate(channels)), torch.from_numpy(target), 1


def focal_loss(logits, target):
    probability = logits.float().sigmoid()
    pt = target * probability + (1 - target) * (1 - probability)
    return ((1 - pt).square() * F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')).mean()


@torch.inference_mode()
def selection_loss(model, loader, device):
    total, count = 0., 0
    for frames, target, slot in loader:
        frames, target = frames.to(device), target.to(device)
        with torch.autocast(device_type=device, enabled=device == 'cuda'):
            logits = model(frames)[0][:, 1]
            loss = focal_loss(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite selection loss')
        total += float(loss) * len(target)
        count += len(target)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs='+', required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('artifacts/models/ball/wasb_tennis_best.pth.tar'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = WASBBallDetector(args.checkpoint, device=device)
    args.output.mkdir(parents=True)
    samples = prepare_views(args.datasets, args.output, detector)
    (args.output / 'training_samples.json').write_text(json.dumps(samples, indent=2))
    train = [sample for sample in samples if sample['split'] == 'train']
    held = [sample for sample in samples if sample['split'] == 'selection']
    training_loader = DataLoader(RandomSpatialView(train), batch_size=1, shuffle=True, num_workers=0)
    selection_loader = DataLoader(SelectionViews(held), batch_size=2, num_workers=0)
    model = detector.model
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(('final_layers.', 'stage4.'))
    model.eval()  # Preserve pretrained batch-normalization statistics.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    run = {'complete': False, 'qualification_evidence': False, 'seed': 7,
           'purpose': 'TRAINING_ONLY_SPATIAL_VIEW_ADAPTATION',
           'initial_checkpoint': str(args.checkpoint), 'initial_checkpoint_sha256': sha256(args.checkpoint),
           'dataset_manifests': {str(d): sha256(d / 'manifest.json') for d in args.datasets},
           'code_hashes': {str(p): sha256(p) for p in [Path(__file__), ROOT / 'scripts/train/finetune_wasb.py',
                                                    ROOT / 'src/detection/tiled_wasb_candidates.py']},
           'train_labels': len(train), 'selection_labels': len(held),
           'train_clips': sorted({s['clip'] for s in train}), 'selection_clips': sorted({s['clip'] for s in held}),
           'training': {'epochs': 3, 'learning_rate': 1e-5, 'batch_size': 1, 'scope': 'last_stage_and_head',
                        'batchnorm': 'FROZEN', 'view_sampling': 'UNIFORM_FULL_PLUS_FOUR_60_PERCENT_CROPS',
                        'augmentation': 'EXISTING_FLIP_BRIGHTNESS_CONTRAST_MOTION_BLUR; NO_SYNTHETIC_GRAPHICS',
                        'absent_oversampling': False},
           'selection': 'EVERY_FIFTH_PUBLISHER_TRAIN_MATCH; ALL_FIVE_VIEWS; CENTRAL_OUTPUT_FOCAL_LOSS',
           'selection_limitation': 'Loss is not precision/recall; source broadcast independence remains unverified',
           'target_semantics': 'EXPLICIT_ACTIVE_BALL_ONLY; OUTSIDE_CROP_IS_ABSENT; UNLABELED_FRAMES_NEVER_TARGETS',
           'epochs': []}
    manifest = args.output / 'manifest.json'
    run['initial_selection_loss'] = selection_loss(model, selection_loader, device)
    run['selected_checkpoint'] = str(args.checkpoint)
    best = run['initial_selection_loss']
    manifest.write_text(json.dumps(run, indent=2))
    print(json.dumps({'initial_selection_loss': best}), flush=True)
    for epoch in range(1, 4):
        start, total, count, absent = time.perf_counter(), 0., 0, 0
        for frames, target, slot in training_loader:
            frames, target, slot = frames.to(device), target.to(device), slot.to(device)
            absent += int((target.flatten(1).sum(1) == 0).sum())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == 'cuda'):
                logits = model(frames)[0][torch.arange(len(slot), device=device), slot]
                loss = focal_loss(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(target)
            count += len(target)
            if count % 200 == 0:
                print(json.dumps({'epoch': epoch, 'labels': count, 'loss': total / count}), flush=True)
        checkpoint = args.output / f'epoch_{epoch:02}.pth.tar'
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        measured = selection_loss(model, selection_loader, device)
        if measured < best:
            best = measured
            run['selected_checkpoint'] = str(checkpoint)
        row = {'epoch': epoch, 'training_loss': total / count, 'selection_loss': measured,
               'absent_draws': absent, 'visible_draws': count - absent, 'seconds_including_selection': time.perf_counter() - start,
               'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint)}
        run['epochs'].append(row)
        manifest.write_text(json.dumps(run, indent=2))
        print(json.dumps(row), flush=True)
    run['complete'] = True
    manifest.write_text(json.dumps(run, indent=2))


if __name__ == '__main__':
    main()
