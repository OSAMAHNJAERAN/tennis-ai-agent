"""Frozen temporal-spacing pilot using real, target-aligned source windows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.detection.tiled_wasb_candidates import overlapping_tiles, merge_by_confidence
from src.evaluation.point_metrics import evaluate_points
from src.tracking.temporal_ball_tracker import BallObservation


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_windows(length, target, stride):
    if type(length) is not int or type(target) is not int or type(stride) is not int or stride<1 or not 0<=target<length:
        raise ValueError('Invalid temporal source geometry')
    result=[]
    for slot in (2,1,0):
        start=target-slot*stride
        if start>=0 and start+2*stride<length:
            result.append(([start,start+stride,start+2*stride],slot))
    if not result: raise ValueError('No complete real triplet for target')
    return result


def infer_candidates(detector,frames,target,stride):
    windows=aligned_windows(len(frames),target,stride)
    context={i:frames[i] for i in sorted({i for indices,_ in windows for i in indices})}
    height,width=context[target].shape[:2]
    boxes=[(0,0,width,height),*overlapping_tiles(width,height,.6)]
    candidates=[]; provenance=[]
    for view, (left,top,right,bottom) in enumerate(boxes):
        aligned=[]
        for indices,slot in windows:
            maps,shape=detector.predict_heatmaps([context[i][top:bottom,left:right] for i in indices])
            if len(maps)!=3: raise ValueError('Model output slots differ')
            aligned.append(maps[slot])
        decoded=detector.decode_heatmaps([np.mean(aligned,axis=0)],shape)[0]
        for rank,p in enumerate(decoded):
            value=BallObservation(p.x_px+left,p.y_px+top,p.confidence)
            candidates.append(value)
            provenance.append({'view':view,'box':[left,top,right,bottom],'mass_rank':rank,
                               'x':value.x_px,'y':value.y_px,'confidence':value.confidence})
    return merge_by_confidence(candidates,(width,height)),provenance,windows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    manifest_path=args.dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert baseline['complete'] and manifest['split']=='VALIDATION_ONLY'
    assert [c['clip'] for c in baseline['clips']]==[f'{m}_{r}' for m,r in manifest['selected_clips']]
    assert all(baseline['configuration'][k]==v for k,v in {'crop_fraction':.6,'heatmap_threshold':.2,'temporal_step':1,'merge_radius_reference_px':4.}.items())
    assert digest(manifest_path)==baseline['dataset_manifest_sha256']
    for file in manifest['files']: assert digest(args.dataset/file['path'])==file['sha256']
    checkpoint=ROOT/'artifacts/models/ball/wasb_tennis_best.pth.tar'
    assert digest(checkpoint)==baseline['checkpoint_sha256']=='9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb'
    for path,checksum in baseline['code_hashes'].items():
        # Frozen inference and metric modules must still be identical.
        if path.startswith('src/detection/') or path=='src/evaluation/point_metrics.py': assert digest(ROOT/path)==checksum
    from src.detection.wasb_ball_detector import WASBBallDetector
    from src.utils.video_frame_sequence import VideoFrameSequence
    detector=WASBBallDetector(checkpoint,threshold=.2,device='cuda')
    report={'complete':False,'qualification_evidence':False,'scope':'SPARSE_TEMPORAL_SPACING_DEVELOPMENT',
            'baseline_sha256':digest(args.baseline),'dataset_manifest_sha256':digest(manifest_path),
            'checkpoint_sha256':digest(checkpoint),'script_sha256':digest(Path(__file__)),
            'configuration':{'target_temporal_fps':30,'heatmap_threshold':.2,'crop_fraction':.6,'merge_radius':4},
            'clips':[]}
    save=lambda:args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save(); all_rows=[];baseline_rows=[]
    for clip in baseline['clips']:
        stride=max(1,int(math.floor(clip['fps']/30+.5)))
        rows=[];details=[];started=time.perf_counter()
        with VideoFrameSequence(str(args.dataset/f"tennis/videos/{clip['clip']}.mp4"),cache_size=12) as frames:
            assert len(frames)==clip['frames'] and frames.metadata.fps==clip['fps']
            assert [frames.metadata.width,frames.metadata.height]==clip['size']
            for row in clip['labeled_rows']['raw']:
                if stride==1:
                    rows.append(dict(row));continue
                selected,candidates,windows=infer_candidates(detector,frames,row['frame'],stride)
                point=[selected[0].x_px,selected[0].y_px] if selected else None
                rows.append({**row,'prediction_xy':point})
                oracle=None
                if row['target_xy'] is not None:
                    oracle=any(math.hypot((p['x']-row['target_xy'][0])*512/row['width'],
                                         (p['y']-row['target_xy'][1])*288/row['height'])<=4 for p in candidates)
                details.append({'frame':row['frame'],'windows':windows,'candidates':candidates,'has_correct_candidate':oracle})
        all_rows.extend(rows);baseline_rows.extend(clip['labeled_rows']['raw'])
        report['clips'].append({'clip':clip['clip'],'fps':clip['fps'],'stride':stride,
                                'source':'verified cached stride-one raw rows' if stride==1 else 'new stride-two inference',
                                'seconds':time.perf_counter()-started,'labeled_rows':rows,'candidate_details':details,
                                'metrics':evaluate_points(rows)})
        save();m=evaluate_points(rows)
        print(json.dumps({'clip':clip['clip'],'stride':stride,'tp':m['true_positives'],'fp':m['false_positives'],'fn':m['false_negatives']}),flush=True)
    def correct(row): return evaluate_points([row])['true_positives']==1
    report.update(complete=True,metrics=evaluate_points(all_rows),baseline_metrics=evaluate_points(baseline_rows),
                  gained_true=sum(correct(a) and not correct(b) for a,b in zip(all_rows,baseline_rows)),
                  lost_true=sum(not correct(a) and correct(b) for a,b in zip(all_rows,baseline_rows)))
    save();print(json.dumps({k:v for k,v in report.items() if k not in ('clips','metrics','baseline_metrics')}),flush=True)


if __name__=='__main__':main()
