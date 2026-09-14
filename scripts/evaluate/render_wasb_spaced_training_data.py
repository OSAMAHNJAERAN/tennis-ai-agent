"""Render fixed adjacent/spaced training-context examples from original videos."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data.prepare_wasb_spaced_training import read, digest


def main():
    import cv2
    import numpy as np
    from src.utils.video_frame_sequence import VideoFrameSequence

    out=ROOT/'outputs/vision_upgrade_audit/wasb_spaced_training_data60'
    target=out/'visual_review.json'
    if target.exists():
        raise FileExistsError(target)
    review=read(out/'review.json')
    assert review['complete']
    audit=read(ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60/report.json')
    clips={c['clip']:c for c in audit['clips']}
    entries,hashes=[],{}
    for row in review['selected_visual_examples']:
        c=clips[row['clip']]
        folder=ROOT/c['dataset']
        video=folder/f"tennis/videos/{row['clip']}.mp4"
        assert digest(video)==c['video_sha256']
        match,rally=row['clip'].rsplit('_',1)
        with (folder/f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as handle:
            labels={int(r['Frame']):r for r in csv.DictReader(handle)}
        label=labels[row['frame']]
        assert int(label['Visibility'])==1
        with VideoFrameSequence(str(video)) as frames:
            index,stride=row['frame'],row['stride']
            w,h=frames.metadata.width,frames.metadata.height
            x,y=round(float(label['X'])*w/1920),round(float(label['Y'])*h/1080)
            panel=np.full((510,1020,3),22,np.uint8)
            frame=frames[index].copy()
            cv2.circle(frame,(x,y),14,(40,255,60),2)
            panel[50:320,:480]=cv2.resize(frame,(480,270))
            cv2.putText(panel,f"{row['split']} {row['clip']} frame {index}; {c['fps']:.2f} FPS, stride {stride}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.52,(240,240,240),1)
            cv2.putText(panel,'Green: unchanged source label; crops have no marks',(8,344),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
            contexts={}
            for label_name,step,top in [('Adjacent',1,62),('Spaced',stride,292)]:
                indices=[index-step,index,index+step]
                contexts[label_name.lower()]=indices
                cv2.putText(panel,label_name,(490,top-12),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
                for col,fi in enumerate(indices):
                    assert 0<=fi<len(frames)
                    patch=frames[fi][max(0,y-50):min(h,y+51),max(0,x-50):min(w,x+51)]
                    left=490+col*175
                    panel[top:top+160,left:left+160]=cv2.resize(patch,(160,160),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel,f'Frame {fi}',(left,top+184),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1)
            path=out/f'context_{len(entries):02}.jpg'
            assert cv2.imwrite(str(path),panel,[cv2.IMWRITE_JPEG_QUALITY,90])
            hashes[path.name]=digest(path)
            entries.append(dict(**row,video_sha256=c['video_sha256'],target_xy=[x,y],contexts=contexts,
                                image=path.name,visually_inspected=False))
    result=dict(complete=True,qualification_evidence=False,review_sha256=digest(out/'review.json'),
                script_sha256=digest(Path(__file__)),entries=entries,image_hashes=hashes,visually_inspected=False,
                source_labels_modified=False,model_trained=False)
    target.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(examples=len(entries),images=list(hashes))))


if __name__=='__main__':
    main()
