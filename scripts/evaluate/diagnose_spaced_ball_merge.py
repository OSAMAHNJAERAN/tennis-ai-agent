"""Inspect the observed cross-view suppression failure without changing its rule."""
import hashlib
import json
import math
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
    target = out/'merge_diagnosis.json'
    if target.exists():
        raise FileExistsError(target)
    report = json.loads((out/'report.json').read_text())
    sparse_path = ROOT/'artifacts/validation/vision_upgrade/additional6_temporal_spacing_pilot01.json'
    assert digest(sparse_path) == report['sources'][str(sparse_path.relative_to(ROOT))]
    sparse = json.loads(sparse_path.read_text())
    row = next(r for r in report['rows'] if (r['clip'],r['frame']) == ('match156_000',32))
    clip = next(c for c in sparse['clips'] if c['clip'] == row['clip'])
    detail = next(d for d in clip['candidate_details'] if d['frame'] == row['frame'])
    def distance(a,b):
        return math.hypot((a[0]-b[0])*512/row['width'],(a[1]-b[1])*288/row['height'])
    kept,diagnostics = [],[]
    for rank,candidate in enumerate(sorted(detail['candidates'],key=lambda c:-c['confidence']),1):
        point = [candidate['x'],candidate['y']]
        suppressor = next((k for k in kept if distance(point,k['point']) <= 4),None)
        entry = dict(rank=rank,point=point,confidence=candidate['confidence'],view=candidate['view'],
                     target_error_reference_px=distance(point,row['target_xy']),
                     suppressed_by_rank=suppressor['rank'] if suppressor else None,
                     suppressor_distance_reference_px=distance(point,suppressor['point']) if suppressor else None)
        diagnostics.append(entry)
        if suppressor is None:
            kept.append(entry)
    video = ROOT/'data/external'/row['dataset']/f"tennis/videos/{row['clip']}.mp4"
    with VideoFrameSequence(str(video)) as frames:
        frame = frames[row['frame']].copy()
        for point,color in [(row['target_xy'],(40,255,60)),(diagnostics[0]['point'],(50,50,255)),(diagnostics[1]['point'],(255,210,30))]:
            cv2.circle(frame,tuple(map(round,point)),12,color,2)
        panel = np.full((350,960,3),22,np.uint8)
        panel[70:340,:480] = cv2.resize(frame,(480,270))
        x,y = map(round,row['target_xy'])
        for col,index in enumerate([31,32,33]):
            crop = frames[index][max(0,y-35):y+36,max(0,x-35):x+36]
            left = 486+col*158
            panel[100:250,left:left+150] = cv2.resize(crop,(150,150),interpolation=cv2.INTER_NEAREST)
            cv2.putText(panel,f'Frame {index}',(left,280),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
        cv2.putText(panel,'match156_000 frame 32: correct candidate removed by duplicate suppression',(6,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1)
        cv2.putText(panel,'Green label / red retained rank 1 / cyan suppressed rank 2; unmarked native crops',(6,43),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
        image = out/'merge_diagnosis.jpg'
        assert cv2.imwrite(str(image),panel,[cv2.IMWRITE_JPEG_QUALITY,80])
    result = dict(complete=True,qualification_evidence=False,audit_sha256=digest(out/'report.json'),
                  candidate_source_sha256=digest(sparse_path),video_sha256=digest(video),
                  script_sha256=digest(Path(__file__)),row=row,candidates=diagnostics,
                  image=image.name,image_sha256=digest(image),visually_inspected=False)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(diagnostics,indent=2))


if __name__ == '__main__':
    main()
