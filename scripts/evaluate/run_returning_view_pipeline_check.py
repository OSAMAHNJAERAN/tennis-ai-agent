"""Bounded real-model integration on the already frozen controlled splice."""
import contextlib
import hashlib
import json
from pathlib import Path
import time

import cv2
import torch

from src.pipeline.phase6_pipeline import Phase6Pipeline
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path('outputs/vision_upgrade_audit/returning_view_pipeline01')
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source_report = Path('outputs/vision_upgrade_audit/returning_view_registration01/report.json')
    source = json.loads(source_report.read_text())
    mapping = next(c['entries'] for c in source['cases'] if c['name'] == 'unrelated_cut_return')
    videos = {Path(p).stem.split('_')[0]: Path(p) for p in source['source_hashes'] if p.endswith('.mp4')}
    for path in videos.values():
        assert digest(path) == source['source_hashes'][str(path)]
    video = out/'controlled_splice.mp4'
    with contextlib.ExitStack() as stack:
        frames = {}
        for key, path in videos.items():
            frames[key] = VideoFrameSequence(str(path))
            stack.callback(frames[key].close)
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 60, (1920, 1080))
        assert writer.isOpened()
        try:
            for entry in mapping:
                writer.write(frames[entry['source']][entry['source_frame']])
        finally:
            writer.release()
    config = Path('configs/phase6_analytics/wasb_returning_view_validation.yaml')
    record = dict(complete=False, qualification_evidence=False,
                  source_report_sha256=digest(source_report), input_sha256=digest(video),
                  config_sha256=digest(config), cuda_available=torch.cuda.is_available(),
                  code_hashes={str(p): digest(p) for p in Path('src').rglob('*.py')},
                  model_hashes={str(p): digest(p) for p in [Path('yolo11m.pt'), Path('yolo11s-pose.pt'),
                                  Path('models/keypoints_model.pth'), Path('artifacts/models/ball/wasb_tennis_best.pth.tar')]},
                  script_sha256=digest(__file__), scope='CONTROLLED_60_FPS_SPLICE; REAL_MODELS; NO_BALL_OR_PHYSICAL_ACCURACY_LABELS')
    manifest = out/'integration_review.json'
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    assert record['cuda_available']
    started = time.perf_counter()
    with (out/'pipeline.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = Phase6Pipeline(str(config)).run(str(video), str(out/'pipeline'))
    record['wall_seconds_including_model_setup'] = time.perf_counter()-started
    assert result['status'] == 'success'
    read = lambda name: json.loads((out/'pipeline'/name).read_text())
    camera = read('camera_registration.json')['frames']
    detections = read('detections.json')['frames']
    motion = read('player_motion.json')
    assert len(camera) == len(detections) == 150
    assert all(not e['is_valid'] for e in camera[60:90])
    restored = [e['frame_index'] for e in camera if e['recovered_this_frame']]
    assert restored and 95 <= restored[0] < 120
    for i, state in enumerate(camera):
        for player in ('player_1', 'player_2'):
            sample = motion[player]['samples'][i]
            if not state['is_valid']:
                assert detections[i][player] is None
                assert sample['ground_position_m'] is None and sample['speed_kmh'] is None
            if i in restored:
                assert sample['speed_kmh'] is None and sample['acceleration_mps2'] is None
    assert read('ball_metrics.json')['summary']['available'] is False
    assert read('match_state.json')['authoritative_events_enabled'] is False
    cap = cv2.VideoCapture(str(out/'pipeline/annotated.mp4'))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 60
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (1080, 1920)
        if count in (59, 75, 119):
            assert cv2.imwrite(str(out/f'pipeline_frame_{count:03}.jpg'), frame)
        count += 1
    cap.release()
    assert count == 150
    record.update(complete=True, recovered_frames=restored, valid_frames=sum(e['is_valid'] for e in camera),
                  decoded_frames=count, gap_coordinates_withheld=True, velocity_not_bridged_across_gap=True,
                  exports_sha256={p.name:digest(p) for p in (out/'pipeline').iterdir() if p.is_file()})
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if not isinstance(v, dict)}))


if __name__ == '__main__':
    main()
