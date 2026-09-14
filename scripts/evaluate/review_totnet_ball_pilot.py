"""Render deterministic diagnostic examples from a completed TOTNet pilot."""
import hashlib
import json
import math
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'outputs/vision_upgrade_audit/totnet_ball_pilot01'


def main():
    path=FOLDER/'report.json'; report=json.loads(path.read_text(encoding='utf-8'))
    assert report['complete']
    board=Image.new('RGB',(1280,3*400),'#111827'); draw=ImageDraw.Draw(board)
    selected=[]
    for i,clip in enumerate(report['clips']):
        def error(p):
            if p['target_xy'] is None or p['prediction_xy'] is None: return -1
            return math.hypot((p['target_xy'][0]-p['prediction_xy'][0])*512/p['width'],
                              (p['target_xy'][1]-p['prediction_xy'][1])*288/p['height'])
        # Prioritize genuine in-image localization failures, not sentinel corners.
        candidates=[p for p in clip['predictions'] if p['grid_xy'] not in ([0,0],[511,0]) and error(p)>4]
        p=max(candidates or clip['predictions'],key=error)
        video=ROOT/f"data/external/racketvision_validation/tennis/videos/{clip['match']}_{clip['rally']}.mp4"
        cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES,p['frame_index'])
        ok,frame=cap.read();cap.release();assert ok
        tile=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,360))
        td=ImageDraw.Draw(tile)
        for key,color in [('target_xy','#63ff87'),('prediction_xy','#ff5c73')]:
            if p[key] is not None:
                x=p[key][0]*640/p['width']; y=p[key][1]*360/p['height']
                td.ellipse((x-7,y-7,x+7,y+7),outline=color,width=2)
        left=(i%2)*640;top=(i//2)*400;board.paste(tile,(left,top+40))
        draw.text((left+8,top+5),f"{clip['match']} frame {p['frame_index']} | error {error(p):.1f}px | peak {p['peak_probability']:.3f}",fill='white')
        draw.text((left+8,top+21),'Green: publisher label   Red: TOTNet argmax',fill='white')
        selected.append({'clip':clip['match'],**p,'error_reference_px':error(p)})
    output=FOLDER/'localization_failures.jpg';board.resize((960,900)).save(output,quality=78)
    review={'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected':selected,
            'selection':'largest visible non-corner localization error per clip; corner fallback',
            'image':str(output),'visually_inspected':False}
    (FOLDER/'visual_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':main()
