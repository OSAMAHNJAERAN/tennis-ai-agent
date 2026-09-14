"""Render fixed internal outcome transitions for aligned-context adaptation."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    out=ROOT/'outputs/vision_upgrade_audit/wasb_spaced_head_pilot05'
    target=out/'visual_review.json'
    if target.exists():
        raise FileExistsError(target)
    review=json.loads((out/'review.json').read_text())
    assert review['complete']
    panels,entries,hashes=[],[],{}
    for row in review['selected_review']:
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
                crop=frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                panel[85:235,left:left+150]=cv2.resize(crop,(150,150),interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel,f'Frame {fi}',(left,260),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,f"INTERNAL SELECTION {row['clip']} frame {index}: {row['comparator']} {row['kind']}",(7,16),cv2.FONT_HERSHEY_SIMPLEX,.30,(245,245,245),1)
            cv2.putText(panel,'Green label / red comparator / cyan spaced-head; fixed threshold 0.20',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
            entries.append(dict(**row,context_frames=context,board=f'comparison_{len(panels)//3:02}.jpg',board_row=len(panels)%3,visually_inspected=False))
            panels.append(panel)
    for start in range(0,len(panels),3):
        path=out/f'comparison_{start//3:02}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,85])
        hashes[path.name]=digest(path)
    result=dict(complete=True,qualification_evidence=False,review_sha256=digest(out/'review.json'),script_sha256=digest(Path(__file__)),
                selection='First two per comparator/outcome transition, clip/frame order',entries=entries,image_hashes=hashes,visually_inspected=False)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(examples=len(entries),images=list(hashes))))


if __name__=='__main__':
    main()
