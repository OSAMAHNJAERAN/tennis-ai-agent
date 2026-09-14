"""Replay a frozen photometric persistence gate on continuous selected points."""

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

from scripts.data.acquire_coco_tennis_validation import digest
from scripts.evaluate.summarize_ball_validation import summarize
from scripts.evaluate.compare_ball_resolution import compare
from src.tracking.candidate_patch_similarity import shifted_patch_similarity, aligned_local_residual
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-local-residual', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = json.loads(args.report.read_text())
    if not source['complete'] or source['dataset_manifest_sha256'] != digest(manifest_path):
        raise ValueError('Require completed source report on this exact dataset')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset hash mismatch')
    thresholds = [.98, .95]
    result = {'qualification_evidence': False, 'complete': False,
              'scope': 'SELECTED_OBSERVATION_REJECTION_ONLY; NO_RERANKING_OR_INTERPOLATION; REUSED_VALIDATION',
              'source_report': str(args.report.resolve()), 'source_sha256': digest(args.report),
              'dataset_manifest_sha256': digest(manifest_path),
              'code_hashes': {name: digest(ROOT / name) for name in (
                  'scripts/evaluate/benchmark_ball_patch_similarity.py',
                  'src/tracking/candidate_patch_similarity.py', 'src/evaluation/point_metrics.py')},
              'configuration': {'reference_size': [960,540], 'lag_seconds': .1, 'patch_radius':5,
                                'translation_radius':2, 'reject_similarity_at_least': thresholds,
                                'local_residual_maximum': 12. if args.require_local_residual else None,
                                'primary_frozen_threshold': .98, 'secondary_development_threshold': .95},
              'clips': []}
    rows = {threshold: [] for threshold in thresholds}
    baseline = []
    started = time.perf_counter()
    for clip in source['clips']:
        predictions = clip['predictions']['pixel_motion']
        history, scores, residuals = deque(), [], []
        with VideoFrameSequence(str(args.dataset / f'tennis/videos/{clip["clip"]}.mp4')) as frames:
            if len(frames) != len(predictions) or frames.metadata.fps != clip['fps']:
                raise ValueError('Source video and predictions misaligned')
            width, height = frames.metadata.width, frames.metadata.height
            for index, frame in enumerate(frames):
                timestamp = index / frames.metadata.fps
                gray = cv2.cvtColor(cv2.resize(frame, (960,540), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                while len(history) > 1 and history[1][0] <= timestamp - .1 + 1e-9:
                    history.popleft()
                previous = history[0][1] if history and history[0][0] <= timestamp - .1 + 1e-9 else None
                point = predictions[index]
                value = (shifted_patch_similarity(gray, previous, (point[0]*960/width,point[1]*540/height))
                         if previous is not None and point is not None else None)
                residual = (aligned_local_residual(gray, previous, (point[0]*960/width,point[1]*540/height))
                            if args.require_local_residual and previous is not None and point is not None else None)
                residuals.append(residual)
                scores.append(value)
                history.append((timestamp, gray))
        for row in clip['labeled_rows']['pixel_motion']:
            baseline.append(row)
            for threshold in thresholds:
                score = scores[row['frame']]
                reject = score is not None and score >= threshold
                if args.require_local_residual:
                    residual = residuals[row['frame']]
                    reject = reject and residual is not None and residual <= 12.
                rows[threshold].append({**row, 'prediction_xy': None if reject else row['prediction_xy']})
        result['clips'].append({'clip': clip['clip'], 'scores': scores, 'local_residuals': residuals})
        print(json.dumps({'clip':clip['clip'], 'frames':len(scores)}), flush=True)
    result.update({'complete': True, 'seconds':time.perf_counter()-started,
                   'results': [{'threshold':threshold, **summarize(values),
                                'paired':compare(baseline,values), 'labeled_rows':values}
                               for threshold,values in rows.items()]})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    for item in result['results']:
        print(json.dumps({'threshold':item['threshold'], **{key:value for key,value in item['pooled'].items()
                                                          if key != 'localization_errors_reference_px'}}),flush=True)


if __name__ == '__main__':
    main()
