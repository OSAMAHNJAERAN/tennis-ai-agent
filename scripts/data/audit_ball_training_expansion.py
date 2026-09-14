"""Verify sixty training clips and render a fixed source-label review sample."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_labels(rows, frame_count, width, height):
    """Convert only explicit visible labels; reject malformed sparse supervision."""
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid source geometry')
    labels, seen = [], set()
    for row in rows:
        index, visibility = int(row['Frame']), int(row['Visibility'])
        if index in seen or not 0 <= index < frame_count or visibility not in (0, 1):
            raise ValueError('Duplicate, out-of-range or invalid visibility label')
        seen.add(index)
        point = [float(row['X']) * width / 1920, float(row['Y']) * height / 1080] if visibility else None
        if point is not None and (not all(math.isfinite(x) for x in point) or not
                                  (0 <= point[0] < width and 0 <= point[1] < height)):
            raise ValueError('Invalid visible coordinate')
        labels.append(dict(frame=index, target_xy=point))
    return sorted(labels, key=lambda r: r['frame'])


def validate_split(manifests, publisher_train, publisher_val):
    if {m for m, _ in publisher_train} & {m for m, _ in publisher_val}:
        raise ValueError('Publisher match IDs overlap')
    revision, clips, seen = manifests[0]['revision'], [], set()
    for manifest in manifests:
        if manifest['split'] != 'TRAINING_ONLY' or manifest['revision'] != revision:
            raise ValueError('Require training-only manifests at one revision')
        for raw in manifest['selected_clips']:
            clip = tuple(raw)
            if clip in seen or clip not in publisher_train:
                raise ValueError('Duplicate or non-training clip')
            seen.add(clip)
            clips.append(clip)
    matches = list(dict.fromkeys(m for m, _ in clips))
    held = set(matches[4::5])
    return {f'{m}_{r}': 'selection' if m in held else 'train' for m, r in clips}


def review_sample(labels):
    """Choose before inference or visual review, independent of model errors."""
    return [row for visible in (True, False)
            if (row := next((r for r in labels if (r['target_xy'] is not None) == visible), None)) is not None]


def render_review(video, labels, path):
    import cv2
    import numpy as np
    from src.utils.video_frame_sequence import VideoFrameSequence
    panels, entries = [], []
    with VideoFrameSequence(str(video)) as frames:
        for label in labels:
            index, target = label['frame'], label['target_xy']
            context = [max(0,index-1), index, min(len(frames)-1,index+1)]
            if target is not None:
                panel = np.full((320,960,3),22,np.uint8)
                frame = frames[index].copy()
                cv2.circle(frame,tuple(map(round,target)),14,(30,255,60),2)
                panel[45:315,:480] = cv2.resize(frame,(480,270))
                x,y = map(round,target)
                h,w = frame.shape[:2]
                for col,fi in enumerate(context):
                    left=486+col*158
                    crop=frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                    panel[85:235,left:left+150]=cv2.resize(crop,(150,150),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel,f'Frame {fi}',(left,260),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            else:
                panel=np.full((230,960,3),22,np.uint8)
                for col,fi in enumerate(context):
                    panel[45:225,col*320:(col+1)*320]=cv2.resize(frames[fi],(320,180))
                    cv2.putText(panel,f'Frame {fi}',(col*320+5,39),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,f"TRAINING ONLY {video.stem} frame {index}: {'visible label' if target is not None else 'publisher absence'}",(7,17),cv2.FONT_HERSHEY_SIMPLEX,.46,(245,245,245),1)
            panels.append(panel)
            entries.append(dict(**label,context_frames=context,visually_inspected=False))
    assert cv2.imwrite(str(path),np.concatenate(panels),[cv2.IMWRITE_JPEG_QUALITY,85])
    return entries


def main():
    import cv2
    from src.utils.video_frame_sequence import VideoFrameSequence
    datasets=[ROOT/'data/external'/name for name in ('racketvision_training20','racketvision_training_additional20','racketvision_training_third20')]
    out=ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60'
    if out.exists():
        raise FileExistsError(out)
    manifests=[read(d/'manifest.json') for d in datasets]
    train_list=read(datasets[0]/'tennis/info/train.json')
    val_list=read(datasets[0]/'tennis/info/val.json')
    split=validate_split(manifests,{tuple(x) for x in train_list},{tuple(x) for x in val_list})
    assert manifests[-1]['selected_clips']==train_list[40:60]
    assert sum(len(m['selected_clips']) for m in manifests)==60
    previous=read(ROOT/'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json')
    assert all(split[c]=='train' for c in previous['train_clips'])
    assert all(split[c]=='selection' for c in previous['selection_clips'])
    for d,m in zip(datasets,manifests,strict=True):
        assert read(d/'tennis/info/train.json')==train_list and read(d/'tennis/info/val.json')==val_list
        for item in m['files']:
            assert digest(d/item['path'])==item['sha256']
    out.mkdir(parents=True)
    report=dict(complete=False,qualification_evidence=False,
                protocol_sha256=digest(ROOT/'docs/experiments/BALL_TRAINING_EXPANSION60_PROTOCOL.md'),
                script_sha256=digest(Path(__file__)),dataset_manifest_hashes={str(d.relative_to(ROOT)):digest(d/'manifest.json') for d in datasets},
                previous_training_manifest_sha256=digest(ROOT/'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json'),
                previous_split_preserved=True, publisher_train_val_match_overlap=[],
                source_broadcast_independence='UNVERIFIED', clips=[],image_hashes={},visual_review_complete=False)
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for d,m in zip(datasets,manifests,strict=True):
        fresh=d==datasets[-1]
        for match,rally in m['selected_clips']:
            clip=f'{match}_{rally}'
            video=d/f'tennis/videos/{clip}.mp4'
            labels_path=d/f'tennis/all/{match}/csv/{rally}_ball.csv'
            with labels_path.open(newline='') as handle:
                raw=list(csv.DictReader(handle))
            with VideoFrameSequence(str(video)) as frames:
                width,height,fps=frames.metadata.width,frames.metadata.height,frames.metadata.fps
                assert math.isfinite(fps) and fps>0
                labels=validate_labels(raw,len(frames),width,height)
                stride=max(1,int(math.floor(fps/30+.5)))
                context_available=sum(any(r['frame']-slot*stride>=0 and r['frame']+(2-slot)*stride<len(frames) for slot in (0,1,2)) for r in labels)
                count=len(frames)
            if fresh:
                cap=cv2.VideoCapture(str(video))
                decoded=0
                try:
                    while True:
                        ok,frame=cap.read()
                        if not ok:
                            break
                        assert frame.shape[:2]==(height,width)
                        decoded+=1
                finally:
                    cap.release()
                assert decoded==count
            entry=dict(clip=clip,dataset=str(d.relative_to(ROOT)),split=split[clip],new_acquisition=fresh,
                       video_sha256=digest(video),labels_sha256=digest(labels_path),frames=count,width=width,height=height,
                       fps=fps,duration_seconds=count/fps,label_count=len(labels),visible_labels=sum(r['target_xy'] is not None for r in labels),
                       absent_labels=sum(r['target_xy'] is None for r in labels),label_density=len(labels)/count,
                       temporal_stride=stride,labels_with_real_spaced_context=context_available,
                       fully_decoded_this_audit=fresh,review=[])
            if fresh:
                name=f'{clip}.jpg'
                entry['review']=render_review(video,review_sample(labels),out/name)
                entry['review_image']=name
                report['image_hashes'][name]=digest(out/name)
            report['clips'].append(entry)
            save()
            if fresh:
                print(json.dumps({k:entry[k] for k in ('clip','frames','label_count','visible_labels','absent_labels','fps')}),flush=True)
    hashes=Counter(c['video_sha256'] for c in report['clips'])
    report['exact_duplicate_video_hashes']=[k for k,v in hashes.items() if v>1]
    assert not report['exact_duplicate_video_hashes']
    report['summary']={name:dict(clips=sum(c['split']==name for c in report['clips']),
                               labels=sum(c['label_count'] for c in report['clips'] if c['split']==name),
                               visible=sum(c['visible_labels'] for c in report['clips'] if c['split']==name),
                               absent=sum(c['absent_labels'] for c in report['clips'] if c['split']==name)) for name in ('train','selection')}
    report.update(complete=True,new_clips_fully_decoded=20,selected_visual_examples=sum(len(c['review']) for c in report['clips']),
                  acquisition_bytes=sum(f['bytes'] for f in manifests[-1]['files']),
                  next_gate='INSPECT_ALL_PRESELECTED_EXAMPLES_BEFORE_TRAINING')
    save()
    print(json.dumps({k:report[k] for k in ('complete','summary','new_clips_fully_decoded','selected_visual_examples','acquisition_bytes')}),flush=True)


if __name__=='__main__':
    main()
