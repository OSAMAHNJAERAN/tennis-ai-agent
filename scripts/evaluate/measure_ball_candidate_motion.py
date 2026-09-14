"""Attach local pixel-change evidence to saved model candidates on original videos."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

from scripts.evaluate.audit_ball_candidate_coverage import digest
from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.temporal_ball_tracker import BallObservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patch-radius', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or len({path.stem for path in args.reports}) != len(args.reports):
        raise ValueError('Output must be fresh and report stems distinct')
    sources = [json.loads(path.read_text()) for path in args.reports]
    manifest_path = args.dataset / 'manifest.json'
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if any(source['dataset_manifest_sha256'] != manifest_hash for source in sources):
        raise ValueError('Dataset manifest does not match source predictions')
    for item in manifest['files']:
        if digest(args.dataset / item['path']) != item['sha256']:
            raise ValueError('Dataset source checksum mismatch')
    for source in sources[1:]:
        if len(source['clips']) != len(sources[0]['clips']):
            raise ValueError('Reports contain different clip sequences')
    metadata = {'schema_version': '1.0', 'qualification_evidence': False,
                'feature_scope': 'CAUSAL_PIXEL_DIFFERENCE; NOT_OBJECT_RECOGNITION_OR_CAMERA_COMPENSATION',
                'configuration': {'lag_seconds': .1, 'reference_size': [960, 540], 'patch_radius': args.patch_radius,
                                  'score': 'MEAN_OF_THREE_LARGEST_ABSOLUTE_GRAYSCALE_DIFFERENCES'},
                'dataset_manifest_sha256': manifest_hash, 'evaluator_sha256': digest(__file__),
                'feature_code_sha256': digest(ROOT / 'src/tracking/candidate_pixel_motion.py')}
    results = [{**metadata, 'source_report': str(path), 'source_sha256': digest(path),
                'source_threshold': source['wasb_heatmap_threshold'], 'clips': []}
               for path, source in zip(args.reports, sources)]
    started = time.perf_counter()
    for number, initial_clip in enumerate(sources[0]['clips']):
        clips = [source['clips'][number] for source in sources]
        for clip in clips:
            if any(clip[key] != initial_clip[key] for key in ('clip', 'fps', 'size', 'decoded_frames')):
                raise ValueError('Report frame geometry or clip order mismatch')
        video_path = args.dataset / ('tennis/videos/' + initial_clip['clip'] + '.mp4')
        cap = cv2.VideoCapture(str(video_path))
        size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if not cap.isOpened() or size != initial_clip['size'] or cap.get(cv2.CAP_PROP_FPS) != initial_clip['fps']:
            raise ValueError('Video geometry/timing differs from source report')
        extractor = CandidatePixelMotion(patch_radius=args.patch_radius)
        enriched = [[] for _ in sources]
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if any(index >= len(clip['predictions']) for clip in clips):
                raise ValueError('More decoded frames than predictions')
            batches = [clip['predictions'][index] for clip in clips]
            observations = [BallObservation(item['x'], item['y'], item['confidence']) for batch in batches for item in batch]
            values = extractor.measure(frame, observations, index / initial_clip['fps'])
            offset = 0
            for output, batch in zip(enriched, batches):
                output.append([{**item, 'pixel_motion_score': score} for item, score in zip(batch, values[offset:offset + len(batch)])])
                offset += len(batch)
            index += 1
        cap.release()
        if any(index != len(clip['predictions']) or index != clip['decoded_frames'] for clip in clips):
            raise ValueError('Decoded frame count mismatch')
        for result, clip, predictions in zip(results, clips, enriched):
            result['clips'].append({key: clip[key] for key in ('clip', 'fps', 'size', 'decoded_frames', 'raw_labeled_rows')}
                                  | {'predictions': predictions})
        print(json.dumps({'clip': initial_clip['clip'], 'decoded_frames': index}), flush=True)
    args.output.mkdir(parents=True)
    for path, result in zip(args.reports, results):
        result['joint_extraction_seconds_including_decode'] = time.perf_counter() - started
        (args.output / (path.stem + '_motion.json')).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
