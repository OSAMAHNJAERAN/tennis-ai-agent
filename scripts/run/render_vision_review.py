"""Render a chronological review video from Phase 6 exports without model inference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2

from src.utils.video_frame_sequence import VideoFrameSequence
from src.visualization.vision_review_renderer import VisionReviewRenderer


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def aligned_motion_samples(motion, frame_count):
    """Accept legacy per-frame arrays and the newer summary-plus-samples export."""
    players = {}
    for identity in ('1', '2'):
        value = motion[f'player_{identity}']
        samples = value['samples'] if isinstance(value, dict) else value
        if not isinstance(samples, list) or len(samples) != frame_count:
            raise ValueError('Motion export length differs from source video')
        players[identity] = samples
    return players


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = {name: json.loads((args.run / f'{name}.json').read_text())
              for name in ('detections', 'player_motion', 'racket_tracking', 'court_geometry', 'run_validation')}
    detections, motion = inputs['detections'], inputs['player_motion']
    video = Path(detections['metadata']['video'])
    if digest(video) != inputs['run_validation']['input_sha256']:
        raise ValueError('Source video differs from the validated run')
    args.output.mkdir(parents=True)
    destination = args.output / 'tracking_review.mp4'
    with VideoFrameSequence(str(video)) as frames:
        if len(frames) != len(detections['frames']) or frames.metadata.fps != detections['metadata']['fps']:
            raise ValueError('Export does not align with source video')
        motion_samples = aligned_motion_samples(motion, len(frames))
        renderer = VisionReviewRenderer(frames.metadata.fps, inputs['court_geometry']['is_valid'])
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'), frames.metadata.fps,
                                 (frames.metadata.width, frames.metadata.height))
        if not writer.isOpened():
            raise RuntimeError('Could not open review writer')
        try:
            for index, frame in enumerate(frames):
                poses = motion['pose_frames'][index] if motion.get('pose_frames') else {}
                rackets = inputs['racket_tracking']['frames'][index] if inputs['racket_tracking']['frames'] else {}
                output = renderer.render(frame, index, detections['frames'][index], poses, rackets,
                                         {identity: samples[index] for identity, samples in motion_samples.items()})
                writer.write(output)
        finally:
            writer.release()
        expected_count, expected_fps = len(frames), frames.metadata.fps
    # Verify the encoded artifact, not only the pre-encode canvas.
    cap = cv2.VideoCapture(str(destination))
    if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) != expected_fps:
        raise ValueError('Review video cannot be decoded at native FPS')
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in (0, 50, 87, 100, 150):
            if not cv2.imwrite(str(args.output / f'frame_{count:05}.jpg'), frame):
                raise RuntimeError('Review screenshot write failed')
        count += 1
    cap.release()
    if count != expected_count:
        raise ValueError('Encoded review frame count differs from source')
    report = {'schema_version': '1.0', 'source_run': str(args.run), 'output': str(destination),
              'decoded_frames': count, 'fps': expected_fps, 'source_video_sha256': digest(video),
              'input_hashes': {name: digest(args.run / f'{name}.json') for name in inputs},
              'renderer_sha256': digest(ROOT / 'src/visualization/vision_review_renderer.py'),
              'accuracy_validated': False, 'events_or_bounces_fabricated': False}
    (args.output / 'review_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
