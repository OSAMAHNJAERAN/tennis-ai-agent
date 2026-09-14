"""Render at most two examples of each paired internal-selection outcome change."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(point,target,width,height):
    if target is None:
        return point is None
    return point is not None and math.hypot((point[0]-target[0])*512/width,(point[1]-target[1])*288/height)<=4


def main():
    out=ROOT/'outputs/vision_upgrade_audit/wasb_training_threshold01'
    target=out/'visual_review.json'
    if target.exists():
        raise FileExistsError(target)
    report=json.loads((out/'report.json').read_text())
    review=json.loads((out/'review.json').read_text())
    assert report['complete'] and review['report_sha256']==digest(out/'report.json')
    original,adapted=report['models']
    old={(c['clip'],r['frame']):r['thresholds'][original['selected_threshold']]['prediction_xy'] for c in original['clips'] for r in c['rows']}
    changes=[]
    for clip in adapted['clips']:
        for row in clip['rows']:
            before=old[clip['clip'],row['frame']]
            after=row['thresholds'][adapted['selected_threshold']]['prediction_xy']
            a=correct(before,row['target_xy'],clip['width'],clip['height'])
            b=correct(after,row['target_xy'],clip['width'],clip['height'])
            if a==b:
                continue
            kind=('gained_' if b else 'lost_')+('absence' if row['target_xy'] is None else 'visible')
            changes.append(dict(clip=clip['clip'],frame=row['frame'],dataset=clip['dataset'],target_xy=row['target_xy'],
                                before_xy=before,after_xy=after,kind=kind,video_sha256=clip['video_sha256']))
    changes.sort(key=lambda r:(r['clip'],r['frame']))
    selected=[]
    used=Counter()
    for row in changes:
        if used[row['kind']]<2:
            selected.append(row)
            used[row['kind']]+=1
    panels=[]
    for row in selected:
        video=ROOT/row['dataset']/f"tennis/videos/{row['clip']}.mp4"
        assert digest(video)==row['video_sha256']
        with VideoFrameSequence(str(video)) as frames:
            index=row['frame']
            frame=frames[index].copy()
            h,w=frame.shape[:2]
            for point,color in [(row['target_xy'],(40,255,60)),(row['before_xy'],(50,50,255)),(row['after_xy'],(255,210,30))]:
                if point is not None:
                    cv2.circle(frame,tuple(map(round,point)),14,color,2)
            panel=np.full((320,960,3),22,np.uint8)
            panel[45:315,:480]=cv2.resize(frame,(480,270))
            x,y=map(round,row['target_xy'] or row['after_xy'] or row['before_xy'])
            context=[max(0,index-1),index,min(len(frames)-1,index+1)]
            for col,fi in enumerate(context):
                left=486+col*158
                patch=frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                panel[85:235,left:left+150]=cv2.resize(patch,(150,150),interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel,f'Frame {fi}',(left,260),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,f"INTERNAL SELECTION {row['clip']} frame {index}: {row['kind']}",(7,16),cv2.FONT_HERSHEY_SIMPLEX,.43,(245,245,245),1)
            cv2.putText(panel,'Green label / red original / cyan adapted; each training-selected threshold',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
            row.update(context_frames=context,board=f'comparison_{len(panels)//3:02}.jpg',board_row=len(panels)%3,visually_inspected=False)
            panels.append(panel)
    hashes={}
    for start in range(0,len(panels),3):
        path=out/f'comparison_{start//3:02}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,74])
        hashes[path.name]=digest(path)
    result=dict(complete=True,qualification_evidence=False,report_sha256=digest(out/'report.json'),script_sha256=digest(Path(__file__)),
                selection_rule='First two per gained/lost visible/absence category in clip/frame order',
                paired_counts=dict(Counter(r['kind'] for r in changes)),changes=changes,selected=selected,
                image_hashes=hashes,visually_inspected=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(paired_counts=result['paired_counts'],selected=len(selected),images=list(hashes))))


if __name__=='__main__':
    main()
