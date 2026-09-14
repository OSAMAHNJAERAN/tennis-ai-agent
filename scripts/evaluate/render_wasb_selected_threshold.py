"""Render the frozen review sample from the completed operating-point test."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    out = ROOT / 'outputs/vision_upgrade_audit/wasb_selected_threshold01'
    destination = out / 'visual_review.json'
    if destination.exists():
        raise FileExistsError(destination)
    review = json.loads((out / 'review.json').read_text())
    assert review['complete'] and digest(out/'report.json') == review['report_sha256']
    panels, entries, image_hashes = [], [], {}
    for row in review['selected_review']:
        video = ROOT / 'data/external' / row['dataset'] / f"tennis/videos/{row['clip']}.mp4"
        assert digest(video) == row['video_sha256']
        with VideoFrameSequence(str(video)) as frames:
            index = row['frame']
            frame = frames[index].copy()
            height, width = frame.shape[:2]
            for point, color in [(row['target_xy'], (40,255,60)), (row['control'], (50,50,255)), (row['selected'], (255,210,30))]:
                if point is not None:
                    cv2.circle(frame, tuple(map(round,point)), 14, color, 2)
            panel = np.full((320,960,3),22,np.uint8)
            panel[45:315,:480] = cv2.resize(frame,(480,270))
            x,y = map(round,row['target_xy'] or row['selected'] or row['control'])
            context = [max(0,index-1),index,min(len(frames)-1,index+1)]
            for col,fi in enumerate(context):
                left = 486 + col*158
                patch = frames[fi][max(0,y-35):min(height,y+36),max(0,x-35):min(width,x+36)]
                panel[85:235,left:left+150] = cv2.resize(patch,(150,150),interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel,f'Frame {fi}',(left,260),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,f"DEVELOPMENT {row['clip']} frame {index}: {row['kind']}",(7,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1)
            cv2.putText(panel,'Green label / red 0.20 / cyan 0.35; same original checkpoint',(7,35),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,245),1)
            entries.append(dict(**row,context_frames=context,board=f'comparison_{len(panels)//3:02}.jpg',
                                board_row=len(panels)%3,visually_inspected=False))
            panels.append(panel)
    for start in range(0,len(panels),3):
        path = out / f'comparison_{start//3:02}.jpg'
        assert cv2.imwrite(str(path),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,80])
        image_hashes[path.name] = digest(path)
    result = dict(complete=True,qualification_evidence=False,visually_inspected=False,
                  report_sha256=review['report_sha256'],review_sha256=digest(out/'review.json'),
                  script_sha256=digest(Path(__file__)),entries=entries,image_hashes=image_hashes)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(examples=len(entries),images=list(image_hashes))))


if __name__ == '__main__':
    main()
