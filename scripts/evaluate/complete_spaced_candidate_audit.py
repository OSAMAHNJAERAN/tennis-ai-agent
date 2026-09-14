"""Acquire missing stride-one candidate evidence at unchanged detector settings."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluate.benchmark_ball_temporal_spacing import infer_candidates
from scripts.evaluate.audit_spaced_ball_errors import distance
from src.detection.wasb_ball_detector import WASBBallDetector
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01'
    target = folder/'stride_one_candidates.json'
    if target.exists():
        raise FileExistsError(target)
    audit = json.loads((folder/'report.json').read_text())
    for name,checksum in audit['sources'].items():
        assert digest(ROOT/name) == checksum
    checkpoint = ROOT/'artifacts/models/ball/wasb_tennis_best.pth.tar'
    assert digest(checkpoint) == '9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb'
    for group in ('expansion12','additional6'):
        prior = json.loads((ROOT/f'artifacts/validation/vision_upgrade/{group}_temporal_spacing_pilot01.json').read_text())
        assert digest(ROOT/'scripts/evaluate/benchmark_ball_temporal_spacing.py') == prior['script_sha256']
    selected = [r for r in audit['rows'] if r['candidates'] is None]
    assert len(selected) == 250 and all(r['stride'] == 1 for r in selected)
    report = dict(complete=False,qualification_evidence=False,audit_sha256=digest(folder/'report.json'),
                  protocol_sha256=digest(ROOT/'docs/experiments/BALL_SPACED_ERROR_AUDIT_PROTOCOL.md'),
                  checkpoint_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),
                  configuration=dict(threshold=.2,crop_fraction=.6,stride=1,merge_radius_reference_px=4),
                  code_hashes={p:digest(ROOT/p) for p in [
                      'src/detection/wasb_ball_detector.py','src/detection/tiled_wasb_candidates.py',
                      'scripts/evaluate/benchmark_ball_temporal_spacing.py']},clips=[])
    def save():
        target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    detector = WASBBallDetector(checkpoint,threshold=.2,device='cuda')
    for clip_name in sorted({r['clip'] for r in selected}):
        rows = [r for r in selected if r['clip'] == clip_name]
        video = ROOT/'data/external'/rows[0]['dataset']/f'tennis/videos/{clip_name}.mp4'
        manifest = json.loads((video.parents[2]/'manifest.json').read_text())
        expected = next(f['sha256'] for f in manifest['files'] if f['path'].replace('\\','/') == f'tennis/videos/{clip_name}.mp4')
        assert digest(video) == expected
        entry = dict(clip=clip_name,video_sha256=expected,rows=[],maximum_raw_difference_native_px=0.)
        started = time.perf_counter()
        with VideoFrameSequence(str(video),cache_size=12) as frames:
            for row in rows:
                merged,candidates,windows = infer_candidates(detector,frames,row['frame'],1)
                point = [merged[0].x_px,merged[0].y_px] if merged else None
                reference = row['stages']['raw']
                if point is None or reference is None:
                    assert point == reference
                else:
                    error = max(abs(a-b) for a,b in zip(point,reference,strict=True))
                    entry['maximum_raw_difference_native_px'] = max(entry['maximum_raw_difference_native_px'],error)
                    assert error <= .0001
                correct = [c for c in candidates if row['target_xy'] is not None and distance([c['x'],c['y']],row['target_xy'],row['width'],row['height']) <= 4]
                entry['rows'].append(dict(frame=row['frame'],raw_prediction_xy=point,windows=windows,candidates=candidates,
                                          has_correct_candidate=bool(correct) if row['target_xy'] is not None else None))
        entry['seconds'] = time.perf_counter()-started
        report['clips'].append(entry)
        save()
        print(json.dumps(dict(clip=clip_name,labels=len(rows),seconds=entry['seconds'],maximum_raw_difference_native_px=entry['maximum_raw_difference_native_px'])),flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
