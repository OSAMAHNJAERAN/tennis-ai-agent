"""Small source-grouped court heatmap fine-tune with paired image/point warps."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def landmarks_with_center(points):
    points = np.asarray(points, dtype=float)
    if points.shape != (14, 2) or not np.isfinite(points).all():
        raise ValueError('Require14 finite landmarks')
    homogeneous = np.column_stack((points, np.ones(14)))
    first = np.cross(homogeneous[0], homogeneous[3])
    second = np.cross(homogeneous[1], homogeneous[2])
    center = np.cross(first, second)
    if abs(center[2]) < 1e-8:
        raise ValueError('Degenerate court diagonals')
    return np.vstack((points, center[:2] / center[2]))


def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(points).all():
        raise ValueError('Invalid finite projective input')
    transformed = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(transformed[:, 2] <= 1e-6):
        raise ValueError('Invalid projective depth')
    return transformed[:, :2] / transformed[:, 2:]


def make_targets(points, radius=55):
    """Author Gaussian radius55/sigma111/6 at640x360; outside centers absent."""
    points = np.asarray(points, dtype=float)
    if points.shape != (15, 2) or not np.isfinite(points).all():
        raise ValueError('Require15 finite points')
    result = np.zeros((15, 360, 640), np.float32)
    coords = np.arange(-radius, radius + 1)
    gaussian = np.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * ((2 * radius + 1) / 6) ** 2))
    for channel, (x, y) in enumerate(points):
        if not 0 <= x < 640 or not 0 <= y < 360:
            continue
        x, y = int(x), int(y)
        left, right = min(x, radius), min(640 - x, radius + 1)
        top, bottom = min(y, radius), min(360 - y, radius + 1)
        result[channel, y-top:y+bottom, x-left:x+right] = gaussian[radius-top:radius+bottom, radius-left:radius+right]
    return result


def warp_matrix(rng, fixed_angle=None):
    angle = float(rng.uniform(-60, 60)) if fixed_angle is None else fixed_angle
    scale = float(rng.uniform(.7, 1.2)) if fixed_angle is None else .9
    matrix = np.vstack((cv2.getRotationMatrix2D((319.5, 179.5), angle, scale), [0, 0, 1]))
    if fixed_angle is None:
        matrix[0, 2] += rng.uniform(-96, 96)
        matrix[1, 2] += rng.uniform(-54, 54)
        projective = np.eye(3)
        projective[2, :2] = rng.uniform(-.00025, .00025, 2)
        matrix = projective @ matrix
    return matrix


def tensor_example(image, points, matrix, brightness=1.0):
    warped = cv2.warpPerspective(image, matrix, (640, 360), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    transformed = transform_points(points, matrix)
    tensor = torch.from_numpy(np.ascontiguousarray(np.clip(warped.astype(np.float32) / 255 * brightness, 0, 1).transpose(2, 0, 1)))
    return tensor.unsqueeze(0), torch.from_numpy(make_targets(transformed)).unsqueeze(0), transformed, warped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data/external/court_heatmap_pilot')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest['complete']:
        raise ValueError('Complete verified pilot required')
    for row in manifest['files']:
        if digest(args.dataset / row['path']) != row['sha256']:
            raise ValueError('Pilot image changed')
    groups = {partition: {r['source_video_id'] for r in manifest['samples'] if r['partition'] == partition}
              for partition in ['train', 'selection', 'external_source_check']}
    if any(groups[a] & groups[b] for a, b in [('train', 'selection'), ('train', 'external_source_check'), ('selection', 'external_source_check')]):
        raise ValueError('Source groups overlap')
    examples = {}
    for partition in ['train', 'selection']:
        examples[partition] = []
        for row in manifest['samples']:
            if row['partition'] != partition:
                continue
            image = cv2.imread(str(args.dataset / 'images' / (row['id'] + '.png')))
            if image is None or image.shape != (720, 1280, 3):
                raise ValueError('Invalid image')
            points = landmarks_with_center(row['kps']) / 2
            examples[partition].append((row['id'], cv2.resize(image, (640, 360)), points))
    source = ROOT / 'artifacts/research/TennisCourtDetector/tracknet.py'
    weights = ROOT / 'artifacts/models/court/heatmap_research/model_tennis_court_det.pt'
    if digest(weights) != '09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04':
        raise ValueError('Starting checkpoint changed')
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    spec = importlib.util.spec_from_file_location('_court_heatmap_pilot', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = module.BallTrackerNet(out_channels=15).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    args.output.mkdir(parents=True)
    report = {'complete': False, 'qualification_evidence': False, 'seed': 17, 'epochs_planned': 3,
              'dataset_manifest_sha256': digest(manifest_path), 'script_sha256': digest(__file__), 'architecture_sha256': digest(source),
              'starting_checkpoint_sha256': digest(weights), 'pretraining_caveat': manifest['pretraining_caveat'],
              'training': 'Full weights Adam1e-5 batch1 three epochs; BatchNorm running statistics frozen; clip gradient norm1; sigmoid MSE author Gaussian targets radius55',
              'augmentation': 'Per image probability0.5 identity, else rotation +/-60deg scale0.7-1.2 translation +/-96x54px projective terms +/-0.00025; independent brightness0.6-1.4; black borders; label centers outside view absent',
              'selection': 'Minimum mean MSE on separate source groups, all identity and fixed +/-45deg scale0.9 views; include original checkpoint; no external images used for selection',
              'counts': {k: len(v) for k, v in examples.items()}, 'epochs': []}
    selection_matrices = [np.eye(3), warp_matrix(rng, -45), warp_matrix(rng, 45)]

    def evaluate():
        model.eval()
        losses = []
        with torch.inference_mode():
            for _, image, points in examples['selection']:
                for matrix in selection_matrices:
                    tensor, target, _, _ = tensor_example(image, points, matrix)
                    prediction = model(tensor.to(device)).sigmoid()
                    losses.append(float(torch.mean((prediction - target.to(device)) ** 2).cpu()))
        return float(np.mean(losses))

    best = evaluate()
    report['initial_selection_mse'] = best
    report['selected_checkpoint'] = str(weights)
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'initial_selection_mse': best}), flush=True)
    for epoch in range(1, 4):
        started = time.perf_counter()
        model.train()
        for layer in model.modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        losses, transformed_count, outside_count = [], 0, 0
        for step, index in enumerate(rng.permutation(len(examples['train']))):
            name, image, points = examples['train'][index]
            augmented = rng.random() < .5
            matrix = warp_matrix(rng) if augmented else np.eye(3)
            tensor, target, transformed, warped = tensor_example(image, points, matrix, brightness=float(rng.uniform(.6, 1.4)))
            transformed_count += int(augmented)
            outside_count += int(((transformed[:, 0] < 0) | (transformed[:, 0] >= 640) | (transformed[:, 1] < 0) | (transformed[:, 1] >= 360)).sum())
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tensor.to(device)).sigmoid()
            loss = torch.mean((prediction - target.to(device)) ** 2)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if epoch == 1 and step < 8:
                for x, y in transformed:
                    if 0 <= x < 640 and 0 <= y < 360:
                        cv2.circle(warped, (round(x), round(y)), 3, (0, 255, 255), -1)
                cv2.imwrite(str(args.output / f'target_preview_{step}_{name}.jpg'), warped)
        selection_loss = evaluate()
        checkpoint = args.output / f'epoch_{epoch:02}.pt'
        torch.save(model.state_dict(), checkpoint)
        row = {'epoch': epoch, 'training_mse': float(np.mean(losses)), 'selection_mse': selection_loss,
               'transformed_examples': transformed_count, 'outside_landmark_centers': outside_count,
               'seconds_including_selection': time.perf_counter() - started, 'checkpoint_sha256': digest(checkpoint)}
        report['epochs'].append(row)
        if selection_loss < best:
            best = selection_loss
            report['selected_checkpoint'] = str(checkpoint)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
    report['complete'] = True
    report['selected_checkpoint_sha256'] = digest(report['selected_checkpoint'])
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
