"""Continuous verification of the fixed temporal-spacing ball candidate."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluate.benchmark_ball_temporal_spacing import digest
from scripts.evaluate.temporal_spacing_stream import predict_spaced_tiled_stream
from scripts.evaluate.spacing_resume import resume_prefix
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.tracking.candidate_pixel_motion import CandidatePixelMotion
from src.tracking.stationary_candidate_filter import StationaryCandidateFilter
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--sparse',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--original-driver',type=Path)
    args=parser.parse_args()
    if bool(args.resume_from) != bool(args.original_driver):
        parser.error('--resume-from and --original-driver must be supplied together')
    if args.output.exists():raise FileExistsError(args.output)
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))
    sparse=json.loads(args.sparse.read_text(encoding='utf-8'))
    manifest_path=args.dataset/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert baseline['complete'] and sparse['complete'] and manifest['split']=='VALIDATION_ONLY'
    assert sparse['baseline_sha256']==digest(args.baseline)
    assert digest(manifest_path)==baseline['dataset_manifest_sha256']==sparse['dataset_manifest_sha256']
    for item in manifest['files']:assert digest(args.dataset/item['path'])==item['sha256']
    checkpoint=ROOT/'artifacts/models/ball/wasb_tennis_best.pth.tar'
    assert digest(checkpoint)==sparse['checkpoint_sha256']==baseline['checkpoint_sha256']
    assert [c['clip'] for c in baseline['clips']]==[f'{m}_{r}' for m,r in manifest['selected_clips']]
    for path,checksum in baseline['code_hashes'].items():
        if path.startswith('src/'):assert digest(ROOT/path)==checksum
    assert digest(ROOT/'scripts/evaluate/benchmark_ball_temporal_spacing.py')==sparse['script_sha256']
    report={'complete':False,'qualification_evidence':False,
            'scope':'CONTINUOUS_TEMPORAL_SPACING_BALL_ONLY_REUSED_DEVELOPMENT',
            'baseline_sha256':digest(args.baseline),'sparse_sha256':digest(args.sparse),
            'dataset_manifest_sha256':digest(manifest_path),'checkpoint_sha256':digest(checkpoint),
            'configuration':{**baseline['configuration'],'native_temporal_stride_rule':'max(1,floor(fps/30+.5))'},
            'code_hashes':{p:digest(ROOT/p) for p in [
                'scripts/evaluate/benchmark_ball_spacing_stream.py','scripts/evaluate/temporal_spacing_stream.py',
                'scripts/evaluate/spacing_resume.py',
                'src/detection/wasb_ball_detector.py','src/detection/tiled_wasb_candidates.py',
                'src/tracking/candidate_pixel_motion.py','src/tracking/stationary_candidate_filter.py',
                'src/evaluation/point_metrics.py']},'clips':[]}
    if args.resume_from:
        report['clips'],report['recovery']=resume_prefix(args.resume_from,report,baseline,sparse,ROOT,args.original_driver)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();detector=WASBBallDetector(checkpoint,threshold=.2,device='cuda')
    all_rows={name:[] for name in ('raw','stationary','pixel_motion')}
    inherited=len(report['clips'])
    for clip in report['clips']:
        for name in all_rows:all_rows[name].extend(clip['labeled_rows'][name])
    for old,pilot in zip(baseline['clips'][inherited:],sparse['clips'][inherited:],strict=True):
        assert old['clip']==pilot['clip']
        stride=max(1,int(math.floor(old['fps']/30+.5)));assert stride==pilot['stride']
        started=time.perf_counter()
        if stride==1:
            clip={**old,'source':'verified stride-one cached continuous baseline','stride':1,
                  'inherited_seconds':old['seconds'],'seconds':0.,'sparse_raw_max_difference_source_px':0.}
            assert old['labeled_rows']['raw']==pilot['labeled_rows']
        else:
            with VideoFrameSequence(str(args.dataset/f"tennis/videos/{old['clip']}.mp4")) as frames:
                fps=frames.metadata.fps;size=(frames.metadata.width,frames.metadata.height)
                assert len(frames)==old['frames'] and fps==old['fps'] and list(size)==old['size']
                queue=deque();max_queue=0
                def source():
                    nonlocal max_queue
                    for frame in frames:
                        queue.append(frame);max_queue=max(max_queue,len(queue));yield frame
                stationary=StationaryCandidateFilter(minimum_seconds=.25)
                pixels=CandidatePixelMotion()
                predictions={name:[] for name in all_rows}
                for index,candidates in enumerate(predict_spaced_tiled_stream(detector,source(),size,stride)):
                    current=queue.popleft()
                    filtered,_=stationary.filter(candidates,index/fps,fps,size)
                    moving,_=pixels.filter(current,filtered,index/fps,minimum_score=12.)
                    for name,values in zip(predictions,(candidates,filtered,moving),strict=True):
                        predictions[name].append([values[0].x_px,values[0].y_px] if values else None)
                assert not queue and all(len(x)==len(frames) for x in predictions.values())
                rows={name:[{**row,'prediction_xy':predictions[name][row['frame']]} for row in pilot['labeled_rows']] for name in all_rows}
                maximum=0.
                for a,b in zip(rows['raw'],pilot['labeled_rows'],strict=True):
                    for key in ('clip','frame','width','height','target_xy'):assert a[key]==b[key]
                    x,y=a['prediction_xy'],b['prediction_xy']
                    if x is None or y is None:assert x==y
                    else:maximum=max(maximum,max(abs(i-j) for i,j in zip(x,y)))
                assert maximum<=1e-4, f'Sparse/continuous mismatch: {maximum}'
                clip={'clip':old['clip'],'fps':fps,'frames':len(frames),'size':list(size),'stride':stride,
                      'source':'new complete native-frame inference','seconds':time.perf_counter()-started,
                      'maximum_queued_native_frames':max_queue,'sparse_raw_max_difference_source_px':maximum,
                      'labeled_rows':rows,'predictions':predictions}
        report['clips'].append(clip)
        for name in all_rows:all_rows[name].extend(clip['labeled_rows'][name])
        save();print(json.dumps({'clip':clip['clip'],'frames':clip['frames'],'stride':stride,
                                'seconds':clip['seconds'],'max_sparse_difference':clip['sparse_raw_max_difference_source_px']}),flush=True)
    report.update(complete=True,results={name:summarize(rows) for name,rows in all_rows.items()},
                  timing_scope='SUM_NEW_60FPS_CLIP_BALL_ONLY_SECONDS;25FPS_CLIPS_REUSE_VERIFIED_BASELINE;NOT_END_TO_END_FPS',
                  seconds=sum(c['seconds'] for c in report['clips']))
    save()
    for name,result in report['results'].items():
        print(json.dumps({'variant':name,**{k:v for k,v in result['pooled'].items() if k!='localization_errors_reference_px'}}),flush=True)


if __name__=='__main__':main()
