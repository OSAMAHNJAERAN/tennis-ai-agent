"""Evaluate court-independent player selection on the fixed UVY recordings."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.nn import tasks
from ultralytics.nn.modules import block,conv,head
from ultralytics.utils import IterableSimpleNamespace
from scripts.evaluate.replay_ball_temporal_detours import digest
from scripts.evaluate.audit_uvy_videos import read_gt
from src.evaluation.tracking_metrics import evaluate_sequence
from src.tracking.global_racket_tracking import GlobalRacketTracking
from src.tracking.racket_supported_players import select_racket_supported_people
from src.utils.bbox_utils import BBox
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/external/uvy_tennis_videos')
    parser.add_argument('--person-report',type=Path,default=ROOT/'outputs/vision_upgrade_audit/uvy_person_tracked640/report.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    manifest_path=args.dataset/'manifest.json';manifest=json.loads(manifest_path.read_text())
    person=json.loads(args.person_report.read_text())
    if not manifest['complete'] or not person['complete'] or person['mode']!='tracked640' or digest(manifest_path)!=person['dataset_manifest_sha256']:
        raise ValueError('Verified aligned tracked-person input required')
    for item in manifest['files']:
        if digest(args.dataset/item['path'])!=item['sha256']:raise ValueError('Dataset changed')
    weights=ROOT/'yolo11m.pt'
    if digest(weights)!=person['checkpoint_sha256'] or digest(weights)!='d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95':
        raise ValueError('Original detector changed')
    allowed=[tasks.DetectionModel,IterableSimpleNamespace,torch.nn.SiLU,torch.nn.BatchNorm2d,
             torch.nn.ModuleList,torch.nn.Sequential,torch.nn.Conv2d,torch.nn.Identity,torch.nn.MaxPool2d,
             torch.nn.Upsample,block.Attention,block.Bottleneck,block.C2PSA,block.C3k,block.C3k2,
             block.DFL,block.PSABlock,block.SPPF,conv.Concat,conv.Conv,conv.DWConv,head.Detect]
    with torch.serialization.safe_globals(allowed):model=ultralytics.YOLO(str(weights))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    report={'complete':False,'qualification_evidence':False,'scope':'OFFLINE_RACKET_SUPPORTED_PLAYER_SELECTION_WITHOUT_COURT',
            'dataset_manifest_sha256':digest(manifest_path),'person_report_sha256':digest(args.person_report),
            'checkpoint_sha256':digest(weights),'weights_only_forced':True,
            'configuration':{'sample_seconds':.2,'racket_confidence':.25,'racket_imgsz':640,'minimum_person_observation_seconds':.5,
                             'minimum_racket_support_frames':2,'opportunity_prior':5,'maximum_selected_people':2},
            'limitations':['Publisher player labels contain known omissions and misclassification; raw metrics are diagnostic.',
                           'Racket association is not independent ownership truth; source IDs may switch.',
                           'Whole-sequence evidence is offline; near/far roles and physical metrics are unavailable.'],
            'code_hashes':{name:digest(ROOT/name) for name in ('scripts/evaluate/benchmark_uvy_racket_supported_players.py',
                 'src/tracking/racket_supported_players.py','src/tracking/global_racket_tracking.py','src/tracking/racket_tracking.py',
                 'src/evaluation/tracking_metrics.py','scripts/evaluate/audit_uvy_videos.py')},
            'runtime':{'torch':str(torch.__version__),'ultralytics':ultralytics.__version__,'device':device},'sequences':{}}
    args.output.mkdir(parents=True)
    for sequence,info in manifest['sequences'].items():
        started=time.perf_counter();cache=person['sequences'][sequence]
        prediction_path=args.person_report.parent/cache['predictions_file']
        if digest(prediction_path)!=cache['predictions_sha256']:raise ValueError('Person predictions changed')
        detections=json.loads(prediction_path.read_text());count=len(detections);fps=cache['fps']
        sampled=sorted({round(k*.2*fps) for k in range(math.ceil(count/fps/.2)) if round(k*.2*fps)<count})
        tracker=GlobalRacketTracking(model=model,device=device,confidence=.25,imgsz=640)
        evidence=[];observations=[]
        with VideoFrameSequence(str(args.dataset/info['video'])) as frames:
            if len(frames)!=count or frames.metadata.fps!=fps:raise ValueError('Frame alignment changed')
            for number,index in enumerate(sampled):
                players={r['id']:BBox(*r['box'],confidence=r['confidence'],track_id=r['id']) for r in detections[index]}
                outputs=tracker.observe(frames[index],players,index/fps)
                for identity,value in outputs.items():
                    if value['state']=='DETECTED':evidence.append({'frame':index,'source_id':identity,'confidence':value['confidence'],'racket_box':value['bbox_xyxy']})
                observations.append({'frame':index,'assigned':outputs,
                                     'raw_candidates':[{**r,'sources':sorted(r['sources'])} for r in tracker.last_candidates]})
                if number%25==0:print(json.dumps({'sequence':sequence,'sampled_frames_done':number+1,'sampled_frames_total':len(sampled),'supports':len(evidence)}),flush=True)
            selected,statistics=select_racket_supported_people(detections,fps,sampled,evidence)
            gt=[[r for r in row if r['class']==1] for row in read_gt(args.dataset/'UVY'/sequence/'gt/gt.txt',count)]
            metrics=evaluate_sequence(gt,selected,ROOT/'artifacts/research/TrackEval')
            for index in sorted({0,count//2,count-1}):
                image=frames[index].copy()
                for row in selected[index]:
                    x1,y1,x2,y2=map(round,row['box']);cv2.rectangle(image,(x1,y1),(x2,y2),(255,230,30),2)
                    cv2.putText(image,f'candidate player {row["id"]}',(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,230,30),1)
                cv2.imwrite(str(args.output/f'{sequence}_{index+1:06}.jpg'),image)
        detail_path=args.output/f'{sequence}_observations.json';detail_path.write_text(json.dumps(observations,allow_nan=False),encoding='utf-8')
        selected_path=args.output/f'{sequence}_selected.json';selected_path.write_text(json.dumps(selected,allow_nan=False),encoding='utf-8')
        report['sequences'][sequence]={'frames':count,'fps':fps,'sampled_frames':sampled,'racket_supports':evidence,'track_statistics':statistics,
             'selected_observations':sum(map(len,selected)),'metrics':metrics,'seconds':time.perf_counter()-started,
             'observations_file':detail_path.name,'observations_sha256':digest(detail_path),
             'selected_file':selected_path.name,'selected_sha256':digest(selected_path),'person_predictions_sha256':digest(prediction_path)}
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'sequence':sequence,'selected_observations':sum(map(len,selected)),'metrics':metrics['summary']}),flush=True)
    report['complete']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
