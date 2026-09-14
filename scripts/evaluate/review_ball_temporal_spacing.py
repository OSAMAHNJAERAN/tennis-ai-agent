"""Verify paired spacing outcomes and render every changed scored outcome."""
import json
import argparse
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.evaluation.point_metrics import evaluate_points
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outcome(row):
    m=evaluate_points([row])
    return next(k for k in ('true_positives','true_negatives','false_positives','false_negatives') if m[k])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=ROOT/'artifacts/validation/vision_upgrade/additional6_temporal_spacing_pilot01.json')
    parser.add_argument('--baseline',type=Path,default=ROOT/'artifacts/validation/vision_upgrade/additional6_tiled_ball_stream_resumed.json')
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/external/racketvision_validation_additional6')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/vision_upgrade_audit/ball_temporal_spacing_pilot01')
    args=parser.parse_args()
    path=args.report;baseline_path=args.baseline
    report=json.loads(path.read_text(encoding='utf-8'));baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    assert report['complete'] and digest(baseline_path)==report['baseline_sha256']
    assert digest(ROOT/'scripts/evaluate/benchmark_ball_temporal_spacing.py')==report['script_sha256']
    dataset=args.dataset
    assert digest(dataset/'manifest.json')==report['dataset_manifest_sha256']
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:assert digest(dataset/item['path'])==item['sha256']
    assert evaluate_points([r for c in report['clips'] for r in c['labeled_rows']])==report['metrics']
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    reviews=[];panels=[]
    for new,old in zip(report['clips'],baseline['clips'],strict=True):
        assert new['clip']==old['clip']
        lookup={r['frame']:r for r in old['labeled_rows']['raw']}
        with VideoFrameSequence(str(dataset/f"tennis/videos/{new['clip']}.mp4")) as frames:
            for row in new['labeled_rows']:
                before=lookup[row['frame']]
                for key in ('clip','frame','width','height','target_xy'):assert row[key]==before[key]
                if outcome(row)==outcome(before):continue
                index=row['frame'];frame=frames[index].copy();h,w=frame.shape[:2]
                center=row['target_xy'] or row['prediction_xy'] or before['prediction_xy']
                x,y=map(round,center)
                for point,color in [(row['target_xy'],(40,255,60)),(before['prediction_xy'],(50,50,255)),(row['prediction_xy'],(255,210,30))]:
                    if point:cv2.circle(frame,tuple(map(round,point)),15,color,2)
                panel=np.full((325,1120,3),24,np.uint8)
                panel[40:321,:500]=cv2.resize(frame,(500,281))
                contexts=[max(0,index-1),index,min(len(frames)-1,index+1)]
                for column,fi in enumerate(contexts):
                    patch=frames[fi][max(0,y-35):min(h,y+36),max(0,x-35):min(w,x+36)]
                    left=510+column*202
                    panel[60:255,left:left+195]=cv2.resize(patch,(195,195),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(panel,f'Frame {fi}',(left,285),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,235,235),1)
                text=f"{row['clip']} frame {index}: {outcome(before)} -> {outcome(row)}"
                cv2.putText(panel,text,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1)
                cv2.putText(panel,'Green GT / Red prior / Cyan spaced; unmarked native target context',(8,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(245,245,245),1)
                reviews.append({**row,'before_xy':before['prediction_xy'],'before':outcome(before),'after':outcome(row),
                                'context_frames':contexts,'board':f'board_{len(panels)//3:02}.jpg','row':len(panels)%3})
                panels.append(panel)
    images={}
    for start in range(0,len(panels),3):
        image=output/f'board_{start//3:02}.jpg'
        cv2.imwrite(str(image),np.concatenate(panels[start:start+3]),[cv2.IMWRITE_JPEG_QUALITY,82]);images[image.name]=digest(image)
    audit={'complete':True,'source_sha256':digest(path),'baseline_sha256':digest(baseline_path),
           'metric_replay_exact':True,'paired_labels_exact':True,'reviews':reviews,'image_hashes':images,'visually_inspected':False}
    (output/'review.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'changed_outcomes':len(reviews),'images':list(images),'metrics':{k:v for k,v in report['metrics'].items() if k!='localization_errors_reference_px'}}))


if __name__=='__main__':main()
