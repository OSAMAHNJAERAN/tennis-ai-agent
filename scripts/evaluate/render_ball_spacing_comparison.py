"""Render verified full-frame paired ball predictions at native frame rate."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=ROOT/'artifacts/validation/vision_upgrade'
    old_path=root/'additional6_tiled_ball_stream_resumed.json'
    new_path=root/'additional6_temporal_spacing_stream01.json'
    old=json.loads(old_path.read_text(encoding='utf-8'));new=json.loads(new_path.read_text(encoding='utf-8'))
    assert old['complete'] and new['complete'] and new['baseline_sha256']==digest(old_path)
    dataset=ROOT/'data/external/racketvision_validation_additional6'
    manifest_path=dataset/'manifest.json'
    assert digest(manifest_path)==new['dataset_manifest_sha256']
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for file in manifest['files']:assert digest(dataset/file['path'])==file['sha256']
    clip_id='match156_000'
    a=next(c for c in old['clips'] if c['clip']==clip_id)
    b=next(c for c in new['clips'] if c['clip']==clip_id)
    assert (a['frames'],a['fps'],a['size'])==(b['frames'],b['fps'],b['size'])
    output=ROOT/'outputs/vision_upgrade_audit/ball_temporal_spacing_pilot01'
    output.mkdir(parents=True,exist_ok=True)
    video=output/'paired_match156.mp4'
    if video.exists():raise FileExistsError(video)
    labels={r['frame']:r['target_xy'] for r in a['labeled_rows']['pixel_motion']}
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),a['fps'],(1280,400))
    assert writer.isOpened()
    snapshots=[]
    with VideoFrameSequence(str(dataset/f'tennis/videos/{clip_id}.mp4')) as frames:
        for i,frame in enumerate(frames):
            canvas=np.full((400,1280,3),22,np.uint8)
            for column,(clip,title) in enumerate([(a,'Adjacent frames'),(b,'Stride-two temporal context')]):
                tile=cv2.resize(frame,(640,360));sx=640/a['size'][0];sy=360/a['size'][1]
                points=clip['predictions']['pixel_motion']
                # Connect only consecutive actually observed positions.
                for j in range(max(1,i-8),i+1):
                    if points[j-1] is not None and points[j] is not None:
                        cv2.line(tile,(round(points[j-1][0]*sx),round(points[j-1][1]*sy)),
                                 (round(points[j][0]*sx),round(points[j][1]*sy)),(0,215,255),1)
                if points[i] is not None:
                    cv2.circle(tile,(round(points[i][0]*sx),round(points[i][1]*sy)),5,(0,215,255),2)
                if labels.get(i) is not None:
                    x,y=labels[i];cv2.circle(tile,(round(x*sx),round(y*sy)),8,(60,255,60),1)
                left=column*640;canvas[40:400,left:left+640]=tile
                cv2.putText(canvas,title+' / stationary + pixel motion',(left+8,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(245,245,245),1)
                cv2.putText(canvas,f'{clip_id} frame {i} | yellow observation / green sparse label',(left+8,34),cv2.FONT_HERSHEY_SIMPLEX,.4,(245,245,245),1)
            writer.write(canvas)
            if i in (0,a['frames']//2,a['frames']-1,246,495):
                p=output/f'paired_frame_{i}.jpg';cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,85]);snapshots.append(p.name)
    writer.release()
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(400,1280);decoded+=1
    cap.release();assert decoded==a['frames'] and abs(fps-a['fps'])<.01
    report={'complete':True,'scope':'PAIRED_RESEARCH_BALL_ONLY_NO_EVENT_OR_SPEED_AUTHORITY','clip':clip_id,
            'source_reports':{str(old_path):digest(old_path),str(new_path):digest(new_path)},
            'video_sha256':digest(video),'decoded_frames':decoded,'fps':fps,'snapshots':snapshots,
            'snapshot_hashes':{name:digest(output/name) for name in snapshots},
            'visually_inspected':False}
    (output/'video_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
