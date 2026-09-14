"""Verify the completed head-only pilot with independent point accounting."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.evaluate.review_wasb_training_threshold import calculate, choose


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def correct(point,target,width,height):
    if target is None:
        return point is None
    return point is not None and math.hypot((point[0]-target[0])*512/width,(point[1]-target[1])*288/height)<=4


def main():
    import torch
    folder=ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04'
    out=ROOT/'outputs/vision_upgrade_audit/wasb_head_pilot04'
    if out.exists():
        raise FileExistsError(out)
    run=read(folder/'manifest.json')
    assert run['complete'] and run['stage']=='COMPLETE' and len(run['evaluations'])==2
    assert digest(ROOT/'docs/experiments/WASB_HEAD_ADAPTATION_PROTOCOL.md')==run['protocol_sha256']
    for name,checksum in run['code_hashes'].items():
        assert digest(ROOT/name)==checksum
    assert digest(folder/'training_samples.json')==run['samples_sha256']
    samples=read(folder/'training_samples.json')
    assert sum(s['split']=='train' for s in samples)==2399
    assert sum(s['split']=='selection' for s in samples)==600
    assert set(run['train_clips']).isdisjoint(run['selection_clips'])
    for name,checksum in run['dataset_manifests'].items():
        dataset=ROOT/name
        assert digest(dataset/'manifest.json')==checksum
        for item in read(dataset/'manifest.json')['files']:
            assert digest(dataset/item['path'])==item['sha256']
    initial,adapted=ROOT/run['initial_checkpoint'],ROOT/run['trained_checkpoint']
    assert digest(initial)==run['initial_checkpoint_sha256']
    assert digest(adapted)==run['trained_checkpoint_sha256']
    original_tensors=torch.load(initial,map_location='cpu',weights_only=True)['model_state_dict']
    adapted_tensors=torch.load(adapted,map_location='cpu',weights_only=True)['model_state_dict']
    assert original_tensors.keys()==adapted_tensors.keys()
    changed=[name for name in original_tensors if not torch.equal(original_tensors[name],adapted_tensors[name])]
    assert changed and all(name.startswith('final_layers.') for name in changed)
    assert run['epoch']['samples']==2399
    assert run['epoch']['absent_view_draws']+run['epoch']['visible_view_draws']==2399
    assert all(math.isfinite(x) and x>0 for pair in run['smoke_losses_and_gradient_norms'] for x in pair)
    assert run['smoke_frozen_state_exact'] and run['smoke_state_restored'] and run['frozen_state_exact']
    prior_path=ROOT/'outputs/vision_upgrade_audit/wasb_training_threshold01/report.json'
    assert digest(prior_path)==run['prior_original_selection_sha256']
    old={c['clip']:c for c in next(m for m in read(prior_path)['models'] if m['name']=='original')['clips']}
    identities=None
    verified=[]
    for model in run['evaluations']:
        assert model['complete'] and len(model['clips'])==12
        assert sorted(c['clip'] for c in model['clips'])==run['selection_clips']
        rows={str(t):[] for t in (.1,.2,.35,.5,.7)}
        current=[]
        for clip in model['clips']:
            dataset=ROOT/clip['dataset']
            match,rally=clip['clip'].rsplit('_',1)
            labels_path=dataset/f'tennis/all/{match}/csv/{rally}_ball.csv'
            assert digest(labels_path)==clip['label_sha256']
            assert digest(dataset/f"tennis/videos/{clip['clip']}.mp4")==clip['video_sha256']
            with labels_path.open(newline='') as handle:
                source=list(csv.DictReader(handle))
            labels={int(r['Frame']):r for r in source}
            assert len(labels)==len(source)==len(clip['rows'])
            assert set(labels)=={r['frame'] for r in clip['rows']}
            assert clip['original_point_and_all_candidates_exact']
            if clip['source']=='VERIFIED_PRIOR_ORIGINAL_INFERENCE':
                assert model['name']=='original' and clip['rows']==old[clip['clip']]['rows']
            else:
                assert clip['source']=='NEW_INFERENCE'
            stride=max(1,int(math.floor(clip['fps']/30+.5)))
            assert stride==clip['stride']
            for row in clip['rows']:
                label=labels[row['frame']]
                expected=[float(label['X'])*clip['width']/1920,float(label['Y'])*clip['height']/1080] if int(label['Visibility']) else None
                assert expected==row['target_xy']
                current.append((clip['clip'],row['frame'],expected))
                windows=[]
                for slot in (2,1,0):
                    start=row['frame']-slot*stride
                    if start>=0 and start+2*stride<clip['frames']:
                        windows.append([[start,start+stride,start+2*stride],slot])
                assert row['windows']==windows and windows
                assert set(row['thresholds'])==set(rows)
                for threshold,prediction in row['thresholds'].items():
                    best=max(prediction['candidates'],key=lambda p:p['confidence'],default=None)
                    assert prediction['prediction_xy']==([best['x'],best['y']] if best else None)
                    rows[threshold].append(dict(clip=clip['clip'],frame=row['frame'],width=clip['width'],height=clip['height'],target_xy=expected,prediction_xy=prediction['prediction_xy']))
        if identities is None:
            identities=current
        assert current==identities and len(current)==600
        scores={t:calculate(r) for t,r in rows.items()}
        for t,m in scores.items():
            assert all(v==model['scores'][t]['pooled'][k] for k,v in m.items())
        selected=max(scores.items(),key=choose)[0]
        assert selected==model['selected_threshold']
        per_clip={c:calculate([r for r in rows[selected] if r['clip']==c]) for c in run['selection_clips']}
        for c,m in per_clip.items():
            assert all(v==model['scores'][selected]['per_clip'][c][k] for k,v in m.items())
        verified.append(dict(name=model['name'],selected_threshold=selected,scores=scores,selected_per_clip=per_clip))
    before,after=run['evaluations']
    original_points={(c['clip'],r['frame']):r['thresholds'][before['selected_threshold']]['prediction_xy'] for c in before['clips'] for r in c['rows']}
    changes=[]
    for c in after['clips']:
        for r in c['rows']:
            a=original_points[c['clip'],r['frame']]
            b=r['thresholds'][after['selected_threshold']]['prediction_xy']
            ac,bc=correct(a,r['target_xy'],c['width'],c['height']),correct(b,r['target_xy'],c['width'],c['height'])
            if ac!=bc:
                changes.append(dict(clip=c['clip'],frame=r['frame'],target_xy=r['target_xy'],before_xy=a,after_xy=b,
                                    kind=('gained_' if bc else 'lost_')+('absence' if r['target_xy'] is None else 'visible'),
                                    dataset=c['dataset'],video_sha256=c['video_sha256']))
    selected,used=[],Counter()
    for r in sorted(changes,key=lambda r:(r['clip'],r['frame'])):
        if used[r['kind']]<2:
            selected.append(r)
            used[r['kind']]+=1
    a,b=[m['scores'][m['selected_threshold']] for m in verified]
    assert run['adapted_wins_internal_selection']==(b['f1']>a['f1'])
    review=dict(complete=True,qualification_evidence=False,manifest_sha256=digest(folder/'manifest.json'),reviewer_sha256=digest(Path(__file__)),
                tensor_changes=changed,all_nonhead_tensors_exact=True,labels_and_windows_exact=True,independent_scores_exact=True,
                models=verified,changes=changes,paired_changes=dict(Counter(r['kind'] for r in changes)),selected_review=selected,
                next_gate=run['next_gate'])
    out.mkdir(parents=True)
    (out/'review.json').write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in review.items() if k not in ('changes','models','selected_review')},indent=2))


if __name__=='__main__':
    main()
