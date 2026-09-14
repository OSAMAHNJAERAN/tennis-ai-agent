"""Author heatmap checkpoint on fixed challenging views, with native coordinates."""

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

from scripts.evaluate.audit_caltennis_projection import LINES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/models/court/heatmap_research/model_tennis_court_det.pt')
    parser.add_argument('--datasets', nargs='+', choices=['uvy_tennis_videos', 'caltennis_diagnostic', 'racketvision_validation'], default=['uvy_tennis_videos', 'caltennis_diagnostic'])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = ROOT / 'artifacts/research/TennisCourtDetector'
    weights = args.checkpoint
    if weights == ROOT / 'artifacts/models/court/heatmap_research/model_tennis_court_det.pt' and digest(weights) != '09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04':
        raise ValueError('Reviewed checkpoint changed')
    spec = importlib.util.spec_from_file_location('_reviewed_court_heatmap', source / 'tracknet.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = module.BallTrackerNet(out_channels=15).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    model.eval()
    report = {'complete': False, 'qualification_evidence': False,
              'protocol': 'All acquired clips in named datasets; CalTennis first/middle/last, other datasets first frame; no model-based sample selection',
              'selected_datasets': args.datasets,
              'preprocessing': 'Author BGR float32/255, resize640x360 linear; sigmoid on15channels; first14 landmarks used',
              'postprocessing': 'Author image-inference threshold170 uint8, HoughCircles dp1 minDist20 param1=50 param2=2 minRadius10 maxRadius25; first returned circle; no refinement/homography filling',
              'coordinate_adaptation': 'Author scale2 assumes1280x720; instead scale each heatmap x by native_width/640 and y by native_height/360',
              'limitations': ['No independent GT accuracy on these views', 'Broadcast-trained source; ground-level/side-camera scope unqualified',
                              'No explicit upstream source license found; no production redistribution grant inferred'],
              'checkpoint_sha256': digest(weights), 'script_sha256': digest(__file__),
              'source_hashes': {p: digest(source / p) for p in ['tracknet.py', 'infer_in_image.py', 'postprocess.py', 'court_reference.py']},
              'datasets': {}, 'frames': []}
    args.output.mkdir(parents=True)
    for dataset_name in args.datasets:
        dataset = ROOT / 'data/external' / dataset_name
        manifest_path = dataset / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if dataset_name == 'racketvision_validation':
            expected = {f'tennis/videos/{match}_{clip}.mp4' for match, clip in manifest['selected_clips']}
            listed = {row['path'] for row in manifest['files']}
            if manifest['split'] != 'VALIDATION_ONLY' or len(expected) != 6 or not expected.issubset(listed):
                raise ValueError('Require complete six-clip legacy validation manifest; file hashes checked below')
        elif not manifest['complete']:
            raise ValueError('Complete acquisition required')
        for row in manifest['files']:
            if digest(dataset / row['path']) != row['sha256']:
                raise ValueError('Dataset changed')
        report['datasets'][dataset_name] = digest(manifest_path)
        if dataset_name == 'racketvision_validation':
            sequences = [{'video': f'tennis/videos/{match}_{clip}.mp4'} for match, clip in manifest['selected_clips']]
        else:
            sequences = list(manifest['sequences'].values()) if isinstance(manifest['sequences'], dict) else manifest['sequences']
        for clip, sequence in enumerate(sequences, 1):
            capture = cv2.VideoCapture(str(dataset / sequence['video']))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = sorted({0, count // 2, count - 1}) if dataset_name == 'caltennis_diagnostic' else [0]
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok or round(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                    raise ValueError('Frame seek failed')
                height, width = image.shape[:2]
                resized = cv2.resize(image, (640, 360))
                tensor = torch.from_numpy(np.ascontiguousarray((resized.astype(np.float32) / 255).transpose(2, 0, 1))).unsqueeze(0).to(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    heatmaps = model(tensor)[0].sigmoid().cpu().numpy()
                if heatmaps.shape != (15, 360, 640) or not np.isfinite(heatmaps).all():
                    raise ValueError('Invalid model output')
                points, peaks, candidates = [], [], []
                for channel in heatmaps[:14]:
                    heatmap = (channel * 255).astype(np.uint8)
                    binary = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)[1]
                    circles = cv2.HoughCircles(binary, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25)
                    point = None if circles is None else [float(circles[0][0][0] * width / 640), float(circles[0][0][1] * height / 360)]
                    points.append(point)
                    peaks.append(float(channel.max()))
                    candidates.append([] if circles is None else circles[0].tolist())
                elapsed = time.perf_counter() - started
                preview = cv2.resize(image, (960, round(height * 960 / width)))
                for a, b in LINES:
                    if points[a] is None or points[b] is None:
                        continue
                    p, q = [tuple(np.rint(np.array(points[k]) * 960 / width).astype(int)) for k in (a, b)]
                    ok_line, p, q = cv2.clipLine((0, 0, preview.shape[1], preview.shape[0]), p, q)
                    if ok_line:
                        cv2.line(preview, p, q, (0, 0, 255), 1, cv2.LINE_AA)
                for k, point in enumerate(points):
                    if point is not None:
                        p = tuple(np.rint(np.array(point) * 960 / width).astype(int))
                        cv2.circle(preview, p, 3, (0, 255, 255), -1)
                        cv2.putText(preview, str(k), p, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                name = f'{dataset_name}_clip{clip}_frame{index+1}'
                preview = cv2.copyMakeBorder(preview, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                detected = sum(p is not None for p in points)
                cv2.putText(preview, f'{name}: {detected}/14 heatmap points; diagnostic only', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                if not cv2.imwrite(str(args.output / f'{name}.jpg'), preview):
                    raise IOError('Preview failed')
                np.savez_compressed(args.output / f'{name}_heatmaps.npz', heatmaps=heatmaps)
                report['frames'].append({'dataset': dataset_name, 'video': sequence['video'], 'frame_index': index,
                                         'size': [width, height], 'points': points, 'peaks': peaks, 'hough_candidates_heatmap_pixels': candidates,
                                         'detected_landmarks': detected, 'inference_and_postprocess_seconds': elapsed, 'artifact_prefix': name})
                print(json.dumps({'frame': name, 'detected': detected}), flush=True)
            capture.release()
    report['complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
