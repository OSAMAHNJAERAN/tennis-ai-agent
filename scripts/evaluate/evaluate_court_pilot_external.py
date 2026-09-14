"""Paired fixed-threshold landmark localization on eight source-separated images."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, default=ROOT / 'artifacts/training/vision_upgrade/court_heatmap_pilot01/report.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    training = json.loads(args.training.read_text(encoding='utf-8'))
    dataset = ROOT / 'data/external/court_heatmap_pilot'
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not training['complete'] or not manifest['complete'] or digest(manifest_path) != training['dataset_manifest_sha256']:
        raise ValueError('Completed matching training/data required')
    for item in manifest['files']:
        if digest(dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    source = ROOT / 'artifacts/research/TennisCourtDetector/tracknet.py'
    if digest(source) != training['architecture_sha256']:
        raise ValueError('Architecture changed')
    weights = {'original': ROOT / 'artifacts/models/court/heatmap_research/model_tennis_court_det.pt',
               'selected': Path(training['selected_checkpoint'])}
    if digest(weights['selected']) != training['selected_checkpoint_sha256'] or digest(weights['original']) != training['starting_checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    rows = [row for row in manifest['samples'] if row['partition'] == 'external_source_check']
    other_sources = {row['source_video_id'] for row in manifest['samples'] if row['partition'] != 'external_source_check'}
    if len(rows) != 8 or any(row['source_video_id'] in other_sources for row in rows):
        raise ValueError('External selection changed')
    spec = importlib.util.spec_from_file_location('_court_pilot_external', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for name, weight in weights.items():
        model = module.BallTrackerNet(out_channels=15).to(device)
        model.load_state_dict(torch.load(weight, map_location=device, weights_only=True), strict=True)
        models[name] = model.eval()
    report = {'complete': False, 'qualification_evidence': False, 'training_report_sha256': digest(args.training),
              'script_sha256': digest(__file__), 'checkpoint_hashes': {name: digest(weight) for name, weight in weights.items()},
              'scope': 'Eight publisher validation images whose source IDs are absent from publisher training; no guarantee of separate tournaments/broadcast reuse',
              'metric': 'Same-channel landmark match distance<=7 pixels in original1280x720; wrong location counts FP+FN; out-of-image GT excluded, not declared absent',
              'inference': 'BGR/255 resize640x360 sigmoid; author170 Hough dp1 minDist20 param1=50 param2=2 radius10..25 first circle; scale2 to native1280x720',
              'counts': {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in models}, 'images': []}
    args.output.mkdir(parents=True)
    for row in rows:
        image = cv2.imread(str(dataset / 'images' / f"{row['id']}.png"))
        if image is None or image.shape != (720, 1280, 3):
            raise ValueError('Image geometry changed')
        tensor = torch.from_numpy(np.ascontiguousarray((cv2.resize(image, (640, 360)).astype(np.float32) / 255).transpose(2, 0, 1))).unsqueeze(0).to(device)
        entry, panels = {'id': row['id'], 'models': {}}, []
        for name, model in models.items():
            with torch.inference_mode():
                heatmaps = model(tensor)[0].sigmoid().cpu().numpy()
            predictions = []
            panel = cv2.resize(image, (640, 360))
            for channel, gt in zip(heatmaps[:14], row['kps'], strict=True):
                binary = cv2.threshold((channel * 255).astype(np.uint8), 170, 255, cv2.THRESH_BINARY)[1]
                circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25)
                prediction = None if circles is None else (circles[0][0][:2] * 2).astype(float).tolist()
                eligible = 0 <= gt[0] < 1280 and 0 <= gt[1] < 720
                distance = None if prediction is None else float(np.linalg.norm(np.asarray(prediction) - gt))
                if eligible:
                    if distance is not None and distance <= 7:
                        report['counts'][name]['tp'] += 1
                    else:
                        report['counts'][name]['fn'] += 1
                        report['counts'][name]['fp'] += int(prediction is not None)
                    cv2.circle(panel, tuple(np.rint(np.asarray(gt) / 2).astype(int)), 4, (0, 255, 0), 1)
                if prediction is not None:
                    cv2.circle(panel, tuple(np.rint(np.asarray(prediction) / 2).astype(int)), 2, (0, 0, 255), -1)
                predictions.append({'gt': gt, 'eligible': eligible, 'prediction': prediction, 'distance_pixels': distance})
            entry['models'][name] = predictions
            panel = cv2.copyMakeBorder(panel, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cv2.putText(panel, f"{row['id']} {name}: green GT / red prediction", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            panels.append(panel)
        cv2.imwrite(str(args.output / f"{row['id']}.jpg"), np.hstack(panels))
        report['images'].append(entry)
    for counts in report['counts'].values():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        counts.update({'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None})
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['counts']), flush=True)


if __name__ == '__main__':
    main()
