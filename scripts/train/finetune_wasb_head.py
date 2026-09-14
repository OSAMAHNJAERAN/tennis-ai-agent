"""One bounded head-only epoch with detection-based internal selection."""
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from scripts.train.finetune_wasb_spatial import prepare_views, RandomSpatialView, focal_loss
from scripts.evaluate.select_wasb_training_threshold import infer_thresholds, selection_key, THRESHOLDS
from scripts.evaluate.benchmark_ball_temporal_spacing import infer_candidates
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)


def configure_head(model):
    names=[]
    for name,p in model.named_parameters():
        p.requires_grad=name.startswith('final_layers.')
        if p.requires_grad:
            names.append(name)
    if not names:
        raise ValueError('Model has no final prediction head')
    model.eval()
    return names


def frozen_state_equal(model, original):
    return all(torch.equal(value.detach().cpu(),original[name]) for name,value in model.state_dict().items()
               if not name.startswith('final_layers.'))


def validate_reused_clip(clip, rows, width, height, fps, length):
    if (clip['width'],clip['height'],clip['fps'],clip['frames'])!=(width,height,fps,length):
        raise ValueError('Cached source geometry changed')
    if len(rows)!=len(clip['rows']) or len({r['frame'] for r in clip['rows']})!=len(rows):
        raise ValueError('Cached label count or uniqueness changed')
    saved={r['frame']:r for r in clip['rows']}
    stride=max(1,int(math.floor(fps/30+.5)))
    if clip['stride']!=stride:
        raise ValueError('Cached spacing changed')
    for label in rows:
        index=int(label['Frame'])
        target=[float(label['X'])*width/1920,float(label['Y'])*height/1080] if int(label['Visibility']) else None
        if index not in saved or saved[index]['target_xy']!=target:
            raise ValueError('Cached target changed')
        windows=[]
        for slot in (2,1,0):
            start=index-slot*stride
            if start>=0 and start+2*stride<length:
                windows.append([[start,start+stride,start+2*stride],slot])
        if saved[index]['windows']!=windows:
            raise ValueError('Cached context changed')


def evaluate(detector, samples, datasets, cached, model_name, run, save):
    result=dict(name=model_name,complete=False,clips=[])
    run['evaluations'].append(result)
    for clip in sorted({s['clip'] for s in samples if s['split']=='selection'}):
        match,rally=clip.rsplit('_',1)
        folder=next(d for d in datasets if (d/f'tennis/videos/{clip}.mp4').exists())
        video=folder/f'tennis/videos/{clip}.mp4'
        label_path=folder/f'tennis/all/{match}/csv/{rally}_ball.csv'
        with label_path.open(newline='') as stream:
            labels=list(csv.DictReader(stream))
        start=time.perf_counter()
        with VideoFrameSequence(str(video),cache_size=12) as frames:
            width,height,fps=frames.metadata.width,frames.metadata.height,frames.metadata.fps
            stride=max(1,int(math.floor(fps/30+.5)))
            if clip in cached:
                entry=cached[clip]
                assert digest(video)==entry['video_sha256'] and digest(label_path)==entry['label_sha256']
                validate_reused_clip(entry,labels,width,height,fps,len(frames))
                entry={**entry,'source':'VERIFIED_PRIOR_ORIGINAL_INFERENCE','reuse_verification_seconds':time.perf_counter()-start}
            else:
                entry=dict(clip=clip,dataset=str(folder.relative_to(ROOT)),video_sha256=digest(video),label_sha256=digest(label_path),
                           width=width,height=height,fps=fps,stride=stride,frames=len(frames),rows=[],source='NEW_INFERENCE')
                for label in labels:
                    index=int(label['Frame'])
                    predictions,windows=infer_thresholds(detector,frames,index,stride)
                    if not entry['rows']:
                        merged,candidates,reference_windows=infer_candidates(detector,frames,index,stride)
                        assert predictions['0.2']['prediction_xy']==([merged[0].x_px,merged[0].y_px] if merged else None)
                        assert predictions['0.2']['candidates']==candidates and windows==reference_windows
                        entry['original_point_and_all_candidates_exact']=True
                    target=[float(label['X'])*width/1920,float(label['Y'])*height/1080] if int(label['Visibility']) else None
                    entry['rows'].append(dict(frame=index,target_xy=target,windows=windows,thresholds=predictions))
                entry['seconds']=time.perf_counter()-start
        result['clips'].append(entry)
        save()
        print(json.dumps(dict(model=model_name,clip=clip,completed_clips=len(result['clips']),source=entry['source'])),flush=True)
    assert len(result['clips'])==12 and sum(len(c['rows']) for c in result['clips'])==600
    scores={str(t):summarize([dict(clip=c['clip'],frame=r['frame'],width=c['width'],height=c['height'],target_xy=r['target_xy'],
                                  prediction_xy=r['thresholds'][str(t)]['prediction_xy']) for c in result['clips'] for r in c['rows']]) for t in THRESHOLDS}
    result.update(complete=True,scores=scores,selected_threshold=max(scores.items(),key=selection_key)[0])
    save()
    print(json.dumps(dict(model=model_name,threshold=result['selected_threshold'],metrics={k:v for k,v in scores[result['selected_threshold']]['pooled'].items() if k!='localization_errors_reference_px'})),flush=True)
    return result


def main():
    out=ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04'
    if out.exists():
        raise FileExistsError(out)
    audit_path=ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60/report.json'
    audit,review=read(audit_path),read(audit_path.with_name('review.json'))
    assert audit['complete'] and review['complete'] and review['report_sha256']==digest(audit_path)
    assert review['preselected_examples_inspected']==35
    for r in review['observations']:
        assert r['visually_inspected']
    datasets=[ROOT/p for p in audit['dataset_manifest_hashes']]
    for d in datasets:
        assert digest(d/'manifest.json')==audit['dataset_manifest_hashes'][str(d.relative_to(ROOT))]
    old_path=ROOT/'outputs/vision_upgrade_audit/wasb_training_threshold01/report.json'
    old=read(old_path)
    old_review=read(old_path.with_name('review.json'))
    assert old['complete'] and old_review['report_sha256']==digest(old_path)
    for path,checksum in old['code_hashes'].items():
        assert digest(ROOT/path)==checksum
    cached_model=next(m for m in old['models'] if m['name']=='original')
    checkpoint=ROOT/cached_model['checkpoint']
    assert digest(checkpoint)==cached_model['checkpoint_sha256']
    assert torch.cuda.is_available()
    seed()
    detector=WASBBallDetector(checkpoint,device='cuda',threshold=.2)
    out.mkdir(parents=True)
    run=dict(complete=False,qualification_evidence=False,stage='PREPARING',
             protocol_sha256=digest(ROOT/'docs/experiments/WASB_HEAD_ADAPTATION_PROTOCOL.md'),audit_sha256=digest(audit_path),
             audit_review_sha256=digest(audit_path.with_name('review.json')),prior_original_selection_sha256=digest(old_path),
             dataset_manifests=audit['dataset_manifest_hashes'],initial_checkpoint=str(checkpoint.relative_to(ROOT)),initial_checkpoint_sha256=digest(checkpoint),
             code_hashes={p:digest(ROOT/p) for p in ['scripts/train/finetune_wasb_head.py','scripts/train/finetune_wasb_spatial.py',
                 'scripts/train/finetune_wasb.py','scripts/evaluate/select_wasb_training_threshold.py',
                 'scripts/evaluate/benchmark_ball_temporal_spacing.py','src/detection/wasb_ball_detector.py',
                 'src/detection/tiled_wasb_candidates.py','src/evaluation/point_metrics.py',
                 'artifacts/research/WASB-SBDT/src/models/hrnet.py']},evaluations=[],
             runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda),
             training=dict(seed=7,epochs=1,scope='final_layers_only',lr=1e-5,batch=1,weight_decay=.0001,gradient_clip=1.,batchnorm='FROZEN'))
    def save():
        path=out/'manifest.json'
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(run,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(path)
    save()
    started=time.perf_counter()
    samples=prepare_views(datasets,out,detector)
    samples_path=out/'training_samples.json'
    samples_path.write_text(json.dumps(samples,indent=2),encoding='utf-8')
    train=[s for s in samples if s['split']=='train']
    assert len(train)==2399 and len(samples)==2999
    run.update(samples_sha256=digest(samples_path),preparation_seconds=time.perf_counter()-started,
               train_clips=sorted({s['clip'] for s in train}),selection_clips=sorted({s['clip'] for s in samples if s['split']=='selection'}))
    # All training tensors were just derived from verified local sources.
    model=detector.model
    original={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    run['trainable_parameters']=configure_head(model)
    def optimizers():
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=.0001),torch.amp.GradScaler('cuda')
    optimizer,scaler=optimizers()
    def step(frames,target,slot):
        frames,target,slot=frames.cuda(),target.cuda(),slot.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits=model(frames)[0][torch.arange(len(slot),device='cuda'),slot]
            loss=focal_loss(logits,target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(norm) or float(norm)<=0:
            raise FloatingPointError('Invalid or zero head gradient')
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach()),float(norm)
    loader=DataLoader(RandomSpatialView(train),batch_size=1,shuffle=True,num_workers=0)
    batch=next(iter(loader))
    smoke=[step(*batch) for _ in range(3)]
    assert frozen_state_equal(model,original)
    model.load_state_dict(original,strict=True)
    seed()
    optimizer,scaler=optimizers()
    run.update(stage='ORIGINAL_SELECTION',smoke_losses_and_gradient_norms=smoke,smoke_frozen_state_exact=True,smoke_state_restored=True)
    save()
    original_eval=evaluate(detector,samples,datasets,{c['clip']:c for c in cached_model['clips']},'original',run,save)
    seed()
    run['stage']='TRAINING'
    save()
    started=time.perf_counter()
    losses,absent=[],0
    for count,(frames,target,slot) in enumerate(loader,1):
        loss,norm=step(frames,target,slot)
        losses.append(loss)
        absent+=int((target.flatten(1).sum(1)==0).sum())
        if count%200==0:
            print(json.dumps(dict(stage='TRAINING',samples=count,mean_loss=sum(losses)/count)),flush=True)
    assert count==2399 and frozen_state_equal(model,original)
    path=out/'epoch_01.pth.tar'
    torch.save({'model_state_dict':model.state_dict()},path)
    run.update(stage='ADAPTED_SELECTION',trained_checkpoint=str(path.relative_to(ROOT)),trained_checkpoint_sha256=digest(path),
               frozen_state_exact=True,epoch=dict(samples=count,absent_view_draws=absent,visible_view_draws=count-absent,
                                                mean_loss=sum(losses)/len(losses),seconds=time.perf_counter()-started))
    save()
    adapted_eval=evaluate(detector,samples,datasets,{},'head_epoch01',run,save)
    a=original_eval['scores'][original_eval['selected_threshold']]['pooled']
    b=adapted_eval['scores'][adapted_eval['selected_threshold']]['pooled']
    run.update(complete=True,stage='COMPLETE',adapted_wins_internal_selection=b['f1']>a['f1'],
               next_gate='FROZEN_EXTERNAL_COMPARISON' if b['f1']>a['f1'] else 'REJECT_HEAD_CANDIDATE',
               peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    save()
    print(json.dumps(dict(complete=True,next_gate=run['next_gate'],original_f1=a['f1'],adapted_f1=b['f1'])),flush=True)


if __name__=='__main__':
    main()
