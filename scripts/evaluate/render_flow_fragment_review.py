"""Render a paired full-sequence research review with actual observed boxes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.audit_uvy_videos import digest
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', choices=('tennis_V01', 'tennis_V02', 'tennis_V03'), default='tennis_V03')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT/'outputs/vision_upgrade_audit'
    folders = [base/'uvy_racket_supported_players_pilot02_nearest', base/'uvy_flow_fragment_linking_pilot01']
    reports = [json.loads((p/'report.json').read_text()) for p in folders]
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if any(not r['complete'] or r['dataset_manifest_sha256'] != digest(dataset/'manifest.json') for r in reports):
        raise ValueError('Incomplete or changed input')
    video = manifest['sequences'][args.sequence]['video']
    entry = next(r for r in manifest['files'] if r['path'] == video)
    if digest(dataset/video) != entry['sha256']:
        raise ValueError('Source video changed')
    tracks = []
    for folder, report in zip(folders, reports, strict=True):
        info = report['sequences'][args.sequence]
        path = folder/info['selected_file']
        if digest(path) != info['selected_sha256']:
            raise ValueError('Selected tracks changed')
        tracks.append(json.loads(path.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with VideoFrameSequence(str(dataset/video)) as frames:
        count, fps = len(frames), frames.metadata.fps
        if any(len(t) != count for t in tracks):
            raise ValueError('Frame count changed')
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 420))
        if not writer.isOpened():
            raise OSError('Video writer failed')
        try:
            for index, source in enumerate(frames):
                canvas = np.full((420, 1280, 3), (25, 22, 18), dtype=np.uint8)
                for column, (title, rows) in enumerate(zip(('BEFORE FRAGMENT LINKING', 'AFTER IMAGE-FLOW LINKING'), tracks, strict=True)):
                    frame = source.copy()
                    for person in rows[index]:
                        x1, y1, x2, y2 = map(round, person['box'])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 230, 30), 2)
                        cv2.putText(frame, f'candidate {person["id"]}', (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 230, 30), 1)
                    canvas[35:395, column*640:(column+1)*640] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    cv2.putText(canvas, title, (column*640+14, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (230, 230, 230), 1)
                cv2.putText(canvas, f'RESEARCH REPLAY  |  {args.sequence}  |  frame {index+1}/{count}  |  {index/fps:.2f}s  |  Missing boxes remain missing',
                            (14, 412), cv2.FONT_HERSHEY_SIMPLEX, .45, (180, 180, 180), 1)
                writer.write(canvas)
        finally:
            writer.release()
    capture = cv2.VideoCapture(str(args.output))
    measured_fps = capture.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame.shape != (420, 1280, 3):
            raise ValueError('Unexpected export dimensions')
        decoded += 1
    capture.release()
    if decoded != count or abs(measured_fps-fps) > .001:
        raise ValueError('Export timing or decode count differs')
    metadata = {'complete': True, 'qualification_evidence': False, 'frames': count, 'decoded_frames': decoded,
                'source_fps': fps, 'export_fps': measured_fps, 'source_video_sha256': entry['sha256'],
                'video_sha256': digest(args.output), 'script_sha256': digest(__file__),
                'report_hashes': [digest(p/'report.json') for p in folders],
                'scope': 'FULL_SEQUENCE_FLOW_LINKING_RESEARCH_REVIEW; NO_COURT_ROLES_OR_PHYSICAL_ANALYTICS'}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
