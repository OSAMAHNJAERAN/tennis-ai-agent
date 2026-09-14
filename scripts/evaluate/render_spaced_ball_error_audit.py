"""Render the predetermined diagnostic sample with native temporal context."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01'
    target = out/'visual_review.json'
    if target.exists():
        raise FileExistsError(target)
    report = json.loads((out/'report.json').read_text())
    lookup = {(r['clip'],r['frame']):r for r in report['rows']}
    panels, entries = [], []
    for selected in report['selected_review']:
        row = lookup[selected['clip'],selected['frame']]
        video = ROOT/'data/external'/row['dataset']/f"tennis/videos/{row['clip']}.mp4"
        with VideoFrameSequence(str(video)) as frames:
            index = row['frame']
            full = frames[index].copy()
            h,w = full.shape[:2]
            for point,color in [(row['target_xy'],(40,255,60)),(row['prediction_xy'],(255,210,30)),(row['stages']['raw'],(50,50,255))]:
                if point is not None:
                    cv2.circle(full,tuple(map(round,point)),14,color,2)
            panel = np.full((320,960,3),22,np.uint8)
            panel[45:315,:480] = cv2.resize(full,(480,270))
            center = row['target_xy'] or row['prediction_xy']
            x,y = map(round,center)
            indices = [max(0,index-1),index,min(len(frames)-1,index+1)]
            for col,fi in enumerate(indices):
                left = 486+col*158
                crop = frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                panel[85:235,left:left+150] = cv2.resize(crop,(150,150),interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel,f'Frame {fi}',(left,260),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,f"{row['clip']} frame {index} / {row['category']}",(7,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
            cv2.putText(panel,'Green label / cyan final / red raw; unmarked target or prediction crops',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.41,(245,245,245),1)
            entries.append(dict(**selected,board=f'board_{len(panels)//3:02}.jpg',row=len(panels)%3,
                                source_sha256=digest(video),context_frames=indices,visually_inspected=False))
            panels.append(panel)
    hashes = {}
    for start in range(0,len(panels),3):
        path = out/f'board_{start//3:02}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,72])
        hashes[path.name] = digest(path)
    review = dict(complete=True,report_sha256=digest(out/'report.json'),renderer_sha256=digest(Path(__file__)),
                  entries=entries,image_hashes=hashes,visually_inspected=False)
    target.write_text(json.dumps(review,indent=2),encoding='utf-8')
    print(json.dumps(dict(entries=len(entries),images=list(hashes))))


if __name__ == '__main__':
    main()
