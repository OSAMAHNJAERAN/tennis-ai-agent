"""Controlled head adaptation with only FPS-aligned training context changed."""
import argparse
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
from scripts.train.finetune_wasb_head import digest, read, seed, configure_head, frozen_state_equal, validate_reused_clip
from scripts.train.finetune_wasb_spatial import RandomSpatialView, focal_loss
from scripts.evaluate.benchmark_ball_temporal_spacing import infer_candidates
from scripts.evaluate.select_wasb_training_threshold import infer_thresholds
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.utils.video_frame_sequence import VideoFrameSequence


def coverage(clips):
    return sum(r['target_xy'] is not None and any(math.hypot((p['x']-r['target_xy'][0])*512/c['width'],
               (p['y']-r['target_xy'][1])*288/c['height'])<=4 for p in r['candidates']) for c in clips for r in c['rows'])


def advance_gate(original, adjacent, candidate):
    a,b,c=[m['scores']['pooled'] for m in (original,adjacent,candidate)]
    return (c['f1']>a['f1'] and c['precision']>=a['precision'] and c['recall']>=a['recall'] and
            candidate['coverage']>=original['coverage'] and c['f1']>=b['f1'] and
            candidate['coverage']>=adjacent['coverage'] and
            (c['f1']>b['f1'] or candidate['coverage']>adjacent['coverage']))


def scores(clips):
    return summarize([dict(clip=c['clip'],frame=r['frame'],width=c['width'],height=c['height'],
                           target_xy=r['target_xy'],prediction_xy=r['prediction_xy']) for c in clips for r in c['rows']])


class RecordedViews:
    def __init__(self,samples):
        self.samples=samples
        self.base=RandomSpatialView(samples)
        self.schedule=[]
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        before=random.getstate()
        predicted_view=random.randrange(5)
        random.setstate(before)
        value=self.base[index]
        self.schedule.append(dict(clip=self.samples[index]['clip'],frame=self.samples[index]['frame'],view=predicted_view,slot=value[2]))
        return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-evaluation',action='store_true')
    args=parser.parse_args()
    out=ROOT/'artifacts/training/vision_upgrade/wasb_spaced_head_pilot05'
    if out.exists() and not args.resume_evaluation:
        raise FileExistsError(out)
    cache=ROOT/'artifacts/training/vision_upgrade/wasb_spaced_cache60'
    prior=ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04'
    cache_run,old=read(cache/'manifest.json'),read(prior/'manifest.json')
    cache_review_path=ROOT/'outputs/vision_upgrade_audit/wasb_spaced_training_data60/review.json'
    visual_path=cache_review_path.with_name('visual_review.json')
    cache_review,visual=read(cache_review_path),read(visual_path)
    old_review_path=ROOT/'outputs/vision_upgrade_audit/wasb_head_pilot04/review.json'
    old_review=read(old_review_path)
    assert all(d['complete'] for d in (cache_run,old,cache_review,visual,old_review))
    assert visual['visually_inspected'] and all(e['visually_inspected'] for e in visual['entries'])
    assert digest(cache_review_path)==visual['review_sha256']
    assert digest(cache/'manifest.json')==cache_review['manifest_sha256']
    assert digest(prior/'manifest.json')==old_review['manifest_sha256']==cache_run['provenance']['prior_manifest_sha256']
    assert digest(cache/'training_samples.json')==cache_run['samples_sha256']
    samples=read(cache/'training_samples.json')
    for collection in (old['code_hashes'],cache_run['provenance']['code_hashes']):
        for p,h in collection.items():
            assert digest(ROOT/p)==h
    for c in cache_run['clips']:
        for p,h in c['cache_hashes'].items():
            assert digest(ROOT/p)==h
    for p,h in old['dataset_manifests'].items():
        folder=ROOT/p
        assert digest(folder/'manifest.json')==h
        for item in read(folder/'manifest.json')['files']:
            assert digest(folder/item['path'])==item['sha256']
    checkpoint=ROOT/old['initial_checkpoint']
    assert digest(checkpoint)==old['initial_checkpoint_sha256']
    code=['scripts/train/finetune_wasb_spaced_head.py','scripts/train/finetune_wasb_head.py',
          'scripts/train/finetune_wasb_spatial.py','scripts/train/finetune_wasb.py',
          'scripts/evaluate/benchmark_ball_temporal_spacing.py','scripts/evaluate/select_wasb_training_threshold.py',
          'scripts/evaluate/summarize_ball_validation.py','src/detection/wasb_ball_detector.py',
          'src/detection/tiled_wasb_candidates.py','src/evaluation/point_metrics.py',
          'src/utils/video_frame_sequence.py','artifacts/research/WASB-SBDT/src/models/hrnet.py']
    provenance=dict(cache_manifest_sha256=digest(cache/'manifest.json'),cache_review_sha256=digest(cache_review_path),
                    cache_visual_sha256=digest(visual_path),prior_manifest_sha256=digest(prior/'manifest.json'),
                    prior_review_sha256=digest(old_review_path),initial_checkpoint_sha256=digest(checkpoint),
                    samples_sha256=digest(cache/'training_samples.json'),
                    protocol_sha256=digest(ROOT/'docs/experiments/WASB_SPACED_HEAD_PILOT05_PROTOCOL.md'),
                    code_hashes={p:digest(ROOT/p) for p in code})
    if args.resume_evaluation:
        run=read(out/'manifest.json')
        assert not run['complete'] and run['stage']=='EVALUATING' and run['provenance']==provenance
        assert digest(ROOT/run['trained_checkpoint'])==run['trained_checkpoint_sha256']
    else:
        out.mkdir(parents=True)
        run=dict(complete=False,qualification_evidence=False,stage='TRAINING',provenance=provenance,threshold=.2,
                 training=old['training'],comparators=[],candidate=dict(name='spaced_head_epoch01',clips=[]))
    def save():
        p=out/'manifest.json'
        temp=p.with_suffix('.tmp')
        temp.write_text(json.dumps(run,indent=2,allow_nan=False),encoding='utf-8')
        temp.replace(p)
    save()
    assert torch.cuda.is_available()
    seed()
    detector=WASBBallDetector(ROOT/run['trained_checkpoint'] if args.resume_evaluation else checkpoint,device='cuda',threshold=.2)
    train=[s for s in samples if s['split']=='train']
    assert len(samples)==2999 and len(train)==2399
    if not args.resume_evaluation:
        run['runtime']=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda)
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
                raise FloatingPointError('Nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm) or float(norm)<=0:
                raise FloatingPointError('Nonfinite or zero gradient')
            scaler.step(optimizer);scaler.update()
            return float(loss.detach()),float(norm)
        loader=DataLoader(RandomSpatialView(train),batch_size=1,shuffle=True,num_workers=0)
        batch=next(iter(loader))
        smoke=[step(*batch) for _ in range(3)]
        assert frozen_state_equal(model,original)
        model.load_state_dict(original,strict=True)
        seed();optimizer,scaler=optimizers()
        run.update(smoke_losses_and_gradient_norms=smoke,smoke_frozen_state_exact=True,smoke_state_restored=True)
        save()
        # Dataset calls make the same number of random draws for unchanged targets;
        # record the schedule directly while using the real unchanged loader.
        old_samples=read(prior/'training_samples.json')
        old_train=[s for s in old_samples if s['split']=='train']
        assert [(s['clip'],s['frame']) for s in old_train]==[(s['clip'],s['frame']) for s in train]
        seed()
        reference_data=RecordedViews(old_train)
        reference_absent=0
        for _,target,_ in DataLoader(reference_data,batch_size=1,shuffle=True,num_workers=0):
            reference_absent+=int((target.flatten(1).sum(1)==0).sum())
        assert reference_absent==1021
        reference_schedule=reference_data.schedule
        (out/'reference_training_schedule.json').write_text(json.dumps(reference_schedule,indent=2),encoding='utf-8')
        run['reference_schedule_sha256']=digest(out/'reference_training_schedule.json')
        save()
        seed()
        recorded=RecordedViews(train)
        loader=DataLoader(recorded,batch_size=1,shuffle=True,num_workers=0)
        started=time.perf_counter();losses=[];absent=0
        for count,(frames,target,slot) in enumerate(loader,1):
            loss,norm=step(frames,target,slot)
            losses.append(loss);absent+=int((target.flatten(1).sum(1)==0).sum())
            if count%400==0:
                print(json.dumps(dict(stage='TRAINING',samples=count,mean_loss=sum(losses)/count)),flush=True)
        assert count==2399 and absent==old['epoch']['absent_view_draws']==1021
        schedule=recorded.schedule
        assert schedule==reference_schedule
        assert frozen_state_equal(model,original)
        path=out/'epoch_01.pth.tar'
        torch.save({'model_state_dict':model.state_dict()},path)
        schedule_path=out/'training_schedule.json'
        schedule_path.write_text(json.dumps(schedule,indent=2),encoding='utf-8')
        run.update(stage='EVALUATING',trained_checkpoint=str(path.relative_to(ROOT)),trained_checkpoint_sha256=digest(path),
                   schedule_sha256=digest(schedule_path),schedule_matches_adjacent_reference=True,frozen_state_exact=True,
                   epoch=dict(samples=count,absent_view_draws=absent,visible_view_draws=count-absent,
                              mean_loss=sum(losses)/len(losses),seconds=time.perf_counter()-started))
        save()
    # Prior models were already evaluated with deployment-aligned source windows.
    if not run['comparators']:
        for old_model in old['evaluations']:
            clips=[]
            for c in old_model['clips']:
                folder=ROOT/c['dataset']
                match,rally=c['clip'].rsplit('_',1)
                import csv
                with (folder/f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as handle:
                    labels=list(csv.DictReader(handle))
                validate_reused_clip(c,labels,c['width'],c['height'],c['fps'],c['frames'])
                clips.append({**{k:c[k] for k in ('clip','dataset','width','height','fps','stride','frames','video_sha256','label_sha256')},
                              'rows':[dict(frame=r['frame'],target_xy=r['target_xy'],windows=r['windows'],**r['thresholds']['0.2']) for r in c['rows']]})
            measured=scores(clips)
            assert measured==old_model['scores']['0.2']
            run['comparators'].append(dict(name=old_model['name'],scores=measured,coverage=coverage(clips),clips=clips))
        save()
    completed={c['clip'] for c in run['candidate']['clips']}
    for source in run['comparators'][0]['clips']:
        if source['clip'] in completed:continue
        clip={k:v for k,v in source.items() if k!='rows'}
        clip['rows']=[]
        started=time.perf_counter()
        video=ROOT/source['dataset']/f"tennis/videos/{source['clip']}.mp4"
        assert digest(video)==source['video_sha256']
        with VideoFrameSequence(str(video),cache_size=12) as frames:
            assert (frames.metadata.width,frames.metadata.height,frames.metadata.fps,len(frames))==tuple(source[k] for k in ('width','height','fps','frames'))
            for i,row in enumerate(source['rows']):
                selected,candidates,windows=infer_candidates(detector,frames,row['frame'],source['stride'])
                point=[selected[0].x_px,selected[0].y_px] if selected else None
                assert json.loads(json.dumps(windows))==row['windows']
                if i==0:
                    reference,rwindows=infer_thresholds(detector,frames,row['frame'],source['stride'])
                    assert rwindows==windows and reference['0.2']==dict(candidates=candidates,prediction_xy=point)
                    clip['internal_inference_reference_exact']=True
                clip['rows'].append(dict(frame=row['frame'],target_xy=row['target_xy'],windows=windows,candidates=candidates,prediction_xy=point))
        torch.cuda.synchronize()
        clip['seconds']=time.perf_counter()-started
        run['candidate']['clips'].append(clip);save()
        print(json.dumps(dict(stage='EVALUATING',clip=clip['clip'],completed_clips=len(run['candidate']['clips']))),flush=True)
    candidate=run['candidate']
    assert len(candidate['clips'])==12 and sum(len(c['rows']) for c in candidate['clips'])==600
    candidate.update(scores=scores(candidate['clips']),coverage=coverage(candidate['clips']))
    advance=advance_gate(*run['comparators'],candidate)
    run.update(complete=True,stage='COMPLETE',advance=advance,next_gate='FROZEN_EXTERNAL_020_COMPARISON' if advance else 'REJECT_SPACING_ONLY_HEAD',
               peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    save()
    print(json.dumps(dict(complete=True,next_gate=run['next_gate'],models=[dict(name=m['name'],coverage=m['coverage'],metrics={k:m['scores']['pooled'][k] for k in ('true_positives','false_positives','false_negatives','precision','recall','f1')}) for m in [*run['comparators'],candidate]])),flush=True)


if __name__=='__main__':main()
