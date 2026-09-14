"""Verify aligned training inputs without training or changing any labels."""
import csv
import json
import math
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.data.prepare_wasb_spaced_training import digest, read


def main():
    import torch
    from scripts.train.finetune_wasb import SparseBallDataset

    out=ROOT/'outputs/vision_upgrade_audit/wasb_spaced_training_data60'
    if out.exists():
        raise FileExistsError(out)
    cache=ROOT/'artifacts/training/vision_upgrade/wasb_spaced_cache60'
    report=read(cache/'manifest.json')
    assert report['complete'] and len(report['clips'])==60
    provenance=report['provenance']
    for name, checksum in provenance['code_hashes'].items():
        assert digest(ROOT/name)==checksum
    assert digest(ROOT/'docs/experiments/WASB_SPACED_TRAINING_DATA_PROTOCOL.md')==provenance['protocol_sha256']
    prior=ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04'
    assert digest(prior/'manifest.json')==provenance['prior_manifest_sha256']
    assert digest(prior/'training_samples.json')==provenance['prior_samples_sha256']
    audit_path=ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60/report.json'
    assert digest(audit_path)==provenance['audit_sha256']
    audit=read(audit_path)
    source_clips={c['clip']:c for c in audit['clips']}
    assert digest(cache/'training_samples.json')==report['samples_sha256']
    old=read(prior/'training_samples.json')
    new=read(cache/'training_samples.json')
    assert len(new)==len(old)==2999
    metadata={c['clip']:c for c in report['clips']}
    content_hashes={p:h for c in report['clips'] for p,h in c['cache_hashes'].items()}
    for p,h in content_hashes.items():
        assert digest(ROOT/p)==h
    labels={}
    for name,checksum in provenance['dataset_manifests'].items():
        folder=ROOT/name
        assert digest(folder/'manifest.json')==checksum
        for item in read(folder/'manifest.json')['files']:
            assert digest(folder/item['path'])==item['sha256']
    for c in audit['clips']:
        match,rally=c['clip'].rsplit('_',1)
        with (ROOT/c['dataset']/f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as handle:
            rows=list(csv.DictReader(handle))
        labels[c['clip']]={int(r['Frame']):r for r in rows}
        assert len(rows)==len(labels[c['clip']])==metadata[c['clip']]['labels']
    used_paths,identities=set(),set()
    for a,b in zip(old,new,strict=True):
        for key in ('clip','frame','split','source_visible'):
            assert a[key]==b[key]
        key=b['clip'],b['frame']
        assert key not in identities
        identities.add(key)
        c=source_clips[b['clip']]
        stride=max(1,int(math.floor(c['fps']/30+.5)))
        indices=[b['frame']+i*stride for i in (-2,-1,0,1,2)]
        assert b['stride']==stride and b['fps']==c['fps'] and b['context_indices']==indices
        assert 0<=indices[0]<indices[-1]<c['frames']
        label=labels[b['clip']][b['frame']]
        point=[float(label['X'])*c['width']/1920,float(label['Y'])*c['height']/1080] if int(label['Visibility']) else None
        assert (point is not None)==b['source_visible']
        w,h=c['width'],c['height']
        tw,th=round(w*.6),round(h*.6)
        boxes=[(0,0,w,h)]+[(x,y,x+tw,y+th) for y in (0,h-th) for x in (0,w-tw)]
        assert len(b['views'])==5
        for v,(av,bv,box) in enumerate(zip(a['views'],b['views'],boxes,strict=True)):
            assert av['target']==bv['target']
            left,top,right,bottom=box
            expected=None if point is None or not(left<=point[0]<right and top<=point[1]<bottom) else [(point[0]-left)*512/(right-left),(point[1]-top)*288/(bottom-top)]
            assert (expected is None)==(bv['target'] is None)
            if expected is not None:
                assert max(abs(x-y) for x,y in zip(expected,bv['target'],strict=True))<.00001
            assert len(bv['context_paths'])==5
            for index,path in zip(indices,bv['context_paths'],strict=True):
                path=Path(path)
                assert path.name==f'{index:06}.jpg' and path.parent.name==f'view_{v}' and path.parent.parent.name==b['clip']
                relative=str(path.relative_to(ROOT))
                assert relative in content_hashes
                used_paths.add(relative)
            if stride==1:
                assert av==bv
    assert used_paths==set(content_hashes)
    assert identities=={(c,frame) for c,rows in labels.items() for frame in rows}
    assert all(c['all_jpegs_decoded'] and c['source_spotcheck_exact'] for c in report['clips'])
    assert len(content_hashes)==14431*5
    chosen=[]
    for split,stride in [('train',2),('selection',2),('train',1)]:
        row=min((s for s in new if s['split']==split and s['stride']==stride and s['source_visible']),key=lambda s:(s['clip'],s['frame']))
        chosen.append(row)
    smoke=[]
    for sample in chosen:
        slots=set()
        dataset=SparseBallDataset([sample['views'][0]])
        for seed in range(12):
            random.seed(seed)
            frames,target,slot=dataset[0]
            assert frames.shape==(9,288,512) and target.shape==(288,512)
            assert torch.isfinite(frames).all() and torch.isfinite(target).all()
            assert set(torch.unique(target).tolist())<={0.,1.} and target.sum()>0
            assert sample['context_indices'][2-slot:5-slot][slot]==sample['frame']
            slots.add(slot)
        assert slots=={0,1,2}
        smoke.append(dict(clip=sample['clip'],frame=sample['frame'],split=sample['split'],stride=sample['stride'],checked_draws=12,slots=sorted(slots)))
    result=dict(complete=True,qualification_evidence=False,manifest_sha256=digest(cache/'manifest.json'),
                samples_sha256=digest(cache/'training_samples.json'),reviewer_sha256=digest(Path(__file__)),
                labels_targets_splits_and_order_exact=True,all_context_paths_and_hashes_exact=True,
                unchanged_stride_one_views_exact=True,decoded_cache_file_count=len(content_hashes),
                historical_cache_source_validation='900 deterministic crop spot checks; not exhaustive source-pixel verification.',
                summary=report['summary'],actual_training_loader_smoke=smoke,
                selected_visual_examples=[dict(clip=s['clip'],frame=s['frame'],split=s['split'],stride=s['stride']) for s in chosen],
                model_trained=False)
    out.mkdir(parents=True)
    (out/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
