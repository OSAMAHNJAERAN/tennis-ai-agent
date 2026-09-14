"""Independent completion review of the spacing-only head experiment."""
import csv
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data.prepare_wasb_spaced_training import digest,read
from scripts.evaluate.review_wasb_selected_threshold import score,outcome,error


def main():
    import torch
    folder=ROOT/'artifacts/training/vision_upgrade/wasb_spaced_head_pilot05'
    out=ROOT/'outputs/vision_upgrade_audit/wasb_spaced_head_pilot05'
    if out.exists():raise FileExistsError(out)
    run=read(folder/'manifest.json')
    assert run['complete'] and run['stage']=='COMPLETE' and run['threshold']==.2
    p=run['provenance']
    sources=dict(cache_manifest_sha256='artifacts/training/vision_upgrade/wasb_spaced_cache60/manifest.json',
                 cache_review_sha256='outputs/vision_upgrade_audit/wasb_spaced_training_data60/review.json',
                 cache_visual_sha256='outputs/vision_upgrade_audit/wasb_spaced_training_data60/visual_review.json',
                 prior_manifest_sha256='artifacts/training/vision_upgrade/wasb_head_pilot04/manifest.json',
                 prior_review_sha256='outputs/vision_upgrade_audit/wasb_head_pilot04/review.json',
                 samples_sha256='artifacts/training/vision_upgrade/wasb_spaced_cache60/training_samples.json',
                 protocol_sha256='docs/experiments/WASB_SPACED_HEAD_PILOT05_PROTOCOL.md')
    for key,path in sources.items():assert digest(ROOT/path)==p[key]
    for path,h in p['code_hashes'].items():assert digest(ROOT/path)==h
    prior=read(ROOT/sources['prior_manifest_sha256'])
    assert run['training']==prior['training']
    assert digest(ROOT/prior['initial_checkpoint'])==p['initial_checkpoint_sha256']
    assert digest(ROOT/run['trained_checkpoint'])==run['trained_checkpoint_sha256']
    a=torch.load(ROOT/prior['initial_checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
    b=torch.load(ROOT/run['trained_checkpoint'],map_location='cpu',weights_only=True)['model_state_dict']
    assert a.keys()==b.keys()
    changed=[k for k in a if not torch.equal(a[k],b[k])]
    assert changed==['final_layers.0.weight','final_layers.0.bias']
    assert run['frozen_state_exact'] and run['smoke_frozen_state_exact'] and run['smoke_state_restored']
    assert all(math.isfinite(x) and x>0 for pair in run['smoke_losses_and_gradient_norms'] for x in pair)
    assert run['epoch']['samples']==2399 and run['epoch']['absent_view_draws']==1021 and run['epoch']['visible_view_draws']==1378
    schedule=read(folder/'training_schedule.json')
    assert digest(folder/'training_schedule.json')==run['schedule_sha256']
    assert digest(folder/'reference_training_schedule.json')==run['reference_schedule_sha256']
    assert schedule==read(folder/'reference_training_schedule.json') and len(schedule)==2399
    assert run['schedule_matches_adjacent_reference']
    samples=read(ROOT/sources['samples_sha256'])
    assert {(s['clip'],s['frame']) for s in samples if s['split']=='train'}=={(s['clip'],s['frame']) for s in schedule}
    assert len({(s['clip'],s['frame']) for s in schedule})==2399
    assert all(s['view'] in range(5) and s['slot'] in range(3) for s in schedule)
    old_models={m['name']:m for m in prior['evaluations']}
    models=[*run['comparators'],run['candidate']]
    all_rows,verified=[],[]
    for model in models:
        paired=[];coverage=0
        assert len(model['clips'])==12
        assert sorted(c['clip'] for c in model['clips'])==prior['selection_clips']
        for c in model['clips']:
            if model is run['candidate']:
                assert c['internal_inference_reference_exact']
            folder_data=ROOT/c['dataset']
            match,rally=c['clip'].rsplit('_',1)
            label_path=folder_data/f'tennis/all/{match}/csv/{rally}_ball.csv'
            assert digest(label_path)==c['label_sha256']
            assert digest(folder_data/f"tennis/videos/{c['clip']}.mp4")==c['video_sha256']
            with label_path.open(newline='') as handle:source=list(csv.DictReader(handle))
            labels={int(r['Frame']):r for r in source}
            assert len(source)==len(labels)==len(c['rows'])
            assert set(labels)=={r['frame'] for r in c['rows']}
            old_clip=next(x for x in old_models['original']['clips'] if x['clip']==c['clip'])
            for key in ('width','height','fps','stride','frames','dataset','video_sha256','label_sha256'):
                assert c[key]==old_clip[key]
            old_rows=None
            if model['name'] in old_models:
                old_rows={r['frame']:r for x in old_models[model['name']]['clips'] if x['clip']==c['clip'] for r in x['rows']}
            stride=max(1,int(math.floor(c['fps']/30+.5)))
            assert stride==c['stride']
            for r in c['rows']:
                label=labels[r['frame']]
                expected=[float(label['X'])*c['width']/1920,float(label['Y'])*c['height']/1080] if int(label['Visibility']) else None
                assert expected==r['target_xy']
                windows=[]
                for slot in (2,1,0):
                    start=r['frame']-slot*stride
                    if 0<=start and start+2*stride<c['frames']:windows.append([[start,start+stride,start+2*stride],slot])
                assert r['windows']==windows and windows
                if old_rows is not None:
                    assert dict(candidates=r['candidates'],prediction_xy=r['prediction_xy'])==old_rows[r['frame']]['thresholds']['0.2']
                best=max(r['candidates'],key=lambda x:x['confidence'],default=None)
                assert r['prediction_xy']==([best['x'],best['y']] if best else None)
                assert all(math.isfinite(x[k]) for x in r['candidates'] for k in ('x','y','confidence'))
                coverage+=expected is not None and any(error([x['x'],x['y']],expected,c['width'],c['height'])<=4 for x in r['candidates'])
                paired.append(dict(clip=c['clip'],frame=r['frame'],width=c['width'],height=c['height'],fps=c['fps'],
                                   target_xy=expected,point=r['prediction_xy'],dataset=c['dataset'],video_sha256=c['video_sha256']))
        assert len(paired)==600
        measured=score(paired,'point')
        assert all(v==model['scores']['pooled'][k] for k,v in measured.items())
        per_clip={clip:score([r for r in paired if r['clip']==clip],'point') for clip in prior['selection_clips']}
        for clip,values in per_clip.items():
            assert all(v==model['scores']['per_clip'][clip][k] for k,v in values.items())
        assert coverage==model['coverage']
        per_fps={str(fps):score([r for r in paired if r['fps']==fps],'point') for fps in sorted({r['fps'] for r in paired})}
        verified.append(dict(name=model['name'],metrics=measured,coverage=coverage,per_clip=per_clip,per_fps=per_fps))
        all_rows.append(paired)
    changes=[]
    for comparator,before_rows in zip(verified[:2],all_rows[:2],strict=True):
        for before,after in zip(before_rows,all_rows[2],strict=True):
            assert all(before[k]==after[k] for k in ('clip','frame','width','height','target_xy','fps'))
            a=outcome(before['point'],before['target_xy'],before['width'],before['height'])
            b=outcome(after['point'],after['target_xy'],after['width'],after['height'])
            if a!=b:
                changes.append(dict(**{k:v for k,v in after.items() if k!='point'},comparator=comparator['name'],
                                    before_xy=before['point'],after_xy=after['point'],kind=a+' -> '+b))
    a,b,c=[m['metrics'] for m in verified]
    ca,cb,cc=[m['coverage'] for m in verified]
    advance=c['f1']>a['f1'] and c['precision']>=a['precision'] and c['recall']>=a['recall'] and cc>=ca and c['f1']>=b['f1'] and cc>=cb and(c['f1']>b['f1'] or cc>cb)
    assert advance==run['advance']
    selected=[];used=Counter()
    for r in sorted(changes,key=lambda r:(r['comparator'],r['clip'],r['frame'])):
        key=r['comparator'],r['kind']
        if used[key]<2:
            selected.append(r);used[key]+=1
    result=dict(complete=True,qualification_evidence=False,manifest_sha256=digest(folder/'manifest.json'),reviewer_sha256=digest(Path(__file__)),
                all_nonhead_tensors_exact=True,changed_tensors=changed,sampling_schedule_exact=True,
                labels_windows_and_cached_comparators_exact=True,independent_scores_and_gate_exact=True,
                models=verified,paired_changes=dict(Counter(r['comparator']+': '+r['kind'] for r in changes)),
                changes=changes,selected_review=selected,advance=advance,next_gate=run['next_gate'])
    out.mkdir(parents=True)
    (out/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('models','changes','selected_review')},indent=2))
    print(json.dumps([dict(name=m['name'],metrics=m['metrics'],coverage=m['coverage']) for m in verified],indent=2))


if __name__=='__main__':main()
