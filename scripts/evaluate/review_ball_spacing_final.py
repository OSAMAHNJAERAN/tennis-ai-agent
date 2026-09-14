"""Verify final frozen-filter spacing reports and render a paired native-FPS clip."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.compare_ball_resolution import compare
from scripts.evaluate.summarize_ball_validation import summarize
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_final(path):
    final = read(path)
    assert final['complete']
    for name, checksum in final['source_reports'].items():
        assert digest(name) == checksum
    sources = [(Path(name), read(name)) for name in final['source_reports']]
    residual_path, residual = next((p, r) for p, r in sources if 'source_report' in r)
    stream_path = Path(residual['source_report'])
    assert digest(stream_path) == residual['source_sha256']
    stream = read(stream_path)
    assert stream['complete'] and residual['complete']
    for report in (final, residual):
        for name, checksum in report['code_hashes'].items():
            assert digest(ROOT / name) == checksum
    replay = summarize(final['labeled_rows'])
    for name, value in replay.items():
        assert final[name] == value
    patches = {c['clip']: c for c in residual['clips']}
    detours = {c['clip']: c for c in final['clips']}
    clips = {}
    for clip in stream['clips']:
        patch = patches[clip['clip']]
        assert len(patch['scores']) == len(patch['local_residuals']) == clip['frames']
        rejected = set(detours[clip['clip']]['rejected_frames'])
        points = [None if i in rejected or (score is not None and score >= .98 and value is not None and value <= 12)
                  else point for i, (point, score, value) in enumerate(zip(
                      clip['predictions']['pixel_motion'], patch['scores'], patch['local_residuals'], strict=True))]
        for row in final['labeled_rows']:
            if row['clip'] == clip['clip']:
                assert row['prediction_xy'] == points[row['frame']]
        clips[clip['clip']] = {**clip, 'final_points': points}
    return final, stream, clips


def render_changes(old, new, changes, dataset, output):
    first = {(r['clip'], r['frame']): r for r in old['labeled_rows']}
    second = {(r['clip'], r['frame']): r for r in new['labeled_rows']}
    panels, entries, images = [], [], {}
    for clip in sorted({c['clip'] for c in changes}):
        with VideoFrameSequence(str(dataset / f'tennis/videos/{clip}.mp4')) as frames:
            for change in (c for c in changes if c['clip'] == clip):
                index = change['frame']
                a, b = first[clip,index], second[clip,index]
                frame = frames[index].copy()
                h, w = frame.shape[:2]
                x,y = map(round, b['target_xy'] or b['prediction_xy'] or a['prediction_xy'])
                for point,color in ((b['target_xy'],(40,255,60)),(a['prediction_xy'],(50,50,255)),(b['prediction_xy'],(255,210,30))):
                    if point: cv2.circle(frame,tuple(map(round,point)),15,color,2)
                panel=np.full((325,1120,3),24,np.uint8)
                panel[40:321,:500]=cv2.resize(frame,(500,281))
                contexts=[max(0,index-1),index,min(len(frames)-1,index+1)]
                for column,fi in enumerate(contexts):
                    patch=frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                    left=510+column*202
                    panel[60:255,left:left+195]=cv2.resize(patch,(195,195),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel,f'Frame {fi}',(left,285),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,235,235),1)
                title=f"{clip} frame {index}: {change['before']} -> {change['after']}"
                cv2.putText(panel,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.47,(245,245,245),1)
                cv2.putText(panel,'Final filters / green GT / red adjacent / cyan spaced; unmarked native context',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1)
                entries.append({**change,'board':f'board_{len(panels)//3:02}.jpg','row':len(panels)%3,'context_frames':contexts})
                panels.append(panel)
    for start in range(0,len(panels),3):
        path=output/f'board_{start//3:02}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,82])
        images[path.name]=digest(path)
    (output/'changed_outcome_review.json').write_text(json.dumps({'complete':True,'entries':entries,'image_hashes':images,'visually_inspected':False},indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    for name in ('baseline', 'report', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--clip', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old, old_stream, old_clips = load_final(args.baseline)
    new, new_stream, new_clips = load_final(args.report)
    for name, checksum in new_stream['code_hashes'].items():
        assert digest(ROOT / name) == checksum
    for name, checksum in old_stream['code_hashes'].items():
        if name.startswith('src/'):
            assert digest(ROOT / name) == checksum
    manifest = read(args.dataset / 'manifest.json')
    assert digest(args.dataset / 'manifest.json') == new_stream['dataset_manifest_sha256'] == old_stream['dataset_manifest_sha256']
    for item in manifest['files']:
        assert digest(args.dataset / item['path']) == item['sha256']
    assert new_stream['checkpoint_sha256'] == old_stream['checkpoint_sha256']
    for a, b in zip(old['labeled_rows'], new['labeled_rows'], strict=True):
        for key in ('clip', 'frame', 'width', 'height', 'target_xy'):
            assert a[key] == b[key]
    paired = compare(old['labeled_rows'], new['labeled_rows'])
    args.output.mkdir(parents=True)
    audit = {'complete': True, 'qualification_evidence': False,
             'source_reports': {str(p.resolve()): digest(p) for p in (args.baseline, args.report)},
             'metric_replay_exact': True, 'final_stream_reconstruction_exact': True,
             'paired_labels_exact': True, 'paired_final': paired,
             'before': summarize(old['labeled_rows']), 'after': summarize(new['labeled_rows']),
             'frames': sum(c['frames'] for c in new_stream['clips']),
             'new_inference_frames': sum(c['frames'] for c in new_stream['clips'] if c['stride'] > 1),
             'seconds_recorded': new_stream['seconds'],
             'timing_qualification': False,
             'current_inference_code_verified': True,
             'all_sparse_raw_points_exact': all(c['sparse_raw_max_difference_source_px'] == 0 for c in new_stream['clips'])}
    (args.output / 'final_comparison.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    render_changes(old,new,paired['changes'],args.dataset,args.output)
    a, b = old_clips[args.clip], new_clips[args.clip]
    assert (a['frames'], a['fps'], a['size']) == (b['frames'], b['fps'], b['size'])
    labels = {r['frame']: r['target_xy'] for r in new['labeled_rows'] if r['clip'] == args.clip}
    targets = {0, a['frames'] // 2, a['frames'] - 1}
    targets.update(c['frame'] for c in paired['changes'] if c['clip'] == args.clip)
    video = args.output / f'paired_{args.clip}.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), a['fps'], (1280, 400))
    assert writer.isOpened()
    snapshots = {}
    with VideoFrameSequence(str(args.dataset / f'tennis/videos/{args.clip}.mp4')) as frames:
        assert len(frames) == a['frames']
        for i, frame in enumerate(frames):
            canvas = np.full((400, 1280, 3), 22, np.uint8)
            for column, (clip, title) in enumerate(((a, 'Adjacent context'), (b, 'Spaced context'))):
                tile = cv2.resize(frame, (640, 360))
                sx, sy = 640 / a['size'][0], 360 / a['size'][1]
                points = clip['final_points']
                def xy(point): return round(point[0] * sx), round(point[1] * sy)
                for j in range(max(1, i - 8), i + 1):
                    if points[j-1] is not None and points[j] is not None:
                        cv2.line(tile, xy(points[j-1]), xy(points[j]), (0, 215, 255), 1)
                if points[i] is not None: cv2.circle(tile, xy(points[i]), 5, (0, 215, 255), 2)
                if labels.get(i) is not None: cv2.circle(tile, xy(labels[i]), 8, (60, 255, 60), 1)
                left = column * 640
                canvas[40:, left:left+640] = tile
                cv2.putText(canvas, title + ' / identical final filters', (left+8, 17), cv2.FONT_HERSHEY_SIMPLEX, .46, (245,245,245), 1)
                cv2.putText(canvas, f'{args.clip} frame {i} | yellow observation / green sparse label', (left+8,34), cv2.FONT_HERSHEY_SIMPLEX, .4, (245,245,245), 1)
            writer.write(canvas)
            if i in targets:
                image = args.output / f'paired_frame_{i}.jpg'
                assert cv2.imwrite(str(image), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
                snapshots[image.name] = digest(image)
    writer.release()
    capture = cv2.VideoCapture(str(video))
    fps, count = capture.get(cv2.CAP_PROP_FPS), 0
    while True:
        ok, frame = capture.read()
        if not ok: break
        assert frame.shape[:2] == (400, 1280)
        count += 1
    capture.release()
    assert count == a['frames'] and abs(fps - a['fps']) < .01
    review = {'complete': True, 'video_sha256': digest(video), 'decoded_frames': count, 'fps': fps,
              'snapshot_hashes': snapshots, 'visually_inspected': False,
              'scope': 'Research observations only; no interpolated coordinates, event or speed authority'}
    (args.output / 'video_review.json').write_text(json.dumps(review, indent=2), encoding='utf-8')
    print(json.dumps({'paired': paired, 'before': {k:v for k,v in old['pooled'].items() if k != 'localization_errors_reference_px'},
                      'after': {k:v for k,v in new['pooled'].items() if k != 'localization_errors_reference_px'}, 'video': review}))


if __name__ == '__main__':
    main()
