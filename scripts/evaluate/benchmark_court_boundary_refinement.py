"""Replay one boundary-preserving revision against every saved court case."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from scripts.evaluate.refine_court_line_geometry import refine as original_refine, LINES
from scripts.evaluate.refine_court_boundary_geometry import refine
from scripts.evaluate.probe_court_line_segments import propose
from scripts.evaluate.benchmark_court_line_refinement import digest
from scripts.evaluate.benchmark_court_refinement_labels import score


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=ROOT/'outputs/vision_upgrade_audit';cases=[];sources={}
    for group,folder in [('selection','court_refinement_selection01'),('external_source_check','court_refinement_raw_labels01'),('pilot','court_line_refinement01')]:
        path=root/folder/'report.json';r=json.loads(path.read_text());assert r['complete'];sources[str(path.relative_to(ROOT))]=digest(path)
        for name,h in r['code_hashes'].items():assert digest(ROOT/name)==h
        for old in r['cases'] if group=='pilot' else r['images']:
            cases.append((group,old))
    assert len(cases)==64
    args.output.mkdir(parents=True)
    report=dict(complete=False,qualification_evidence=False,scope='REUSED_DEVELOPMENT_COMPARISON; NOT_INDEPENDENT_ACCURACY',sources=sources,
        protocol_sha256=digest(ROOT/'docs/experiments/COURT_BOUNDARY_REFINEMENT_PROTOCOL.md'),
        code_hashes={p:digest(ROOT/p) for p in ['scripts/evaluate/benchmark_court_boundary_refinement.py','scripts/evaluate/refine_court_boundary_geometry.py','scripts/evaluate/refine_court_line_geometry.py','scripts/evaluate/probe_court_line_segments.py','scripts/evaluate/audit_court_line_support.py','scripts/evaluate/audit_caltennis_projection.py','scripts/evaluate/benchmark_court_refinement_labels.py']},cases=[])
    cache={}
    for index,(group,old) in enumerate(cases):
        identity=old.get('id',old.get('identity'));entry=dict(index=index,group=group,id=identity,model=old.get('model'),frame=old.get('frame'),original=old)
        valid=('original_points' in old) if group=='pilot' else old['initial_calibration']['is_valid']
        if not valid:
            entry.update(initial_valid=False,accepted=False,reason='INITIAL_GEOMETRIC_CALIBRATION_INVALID')
            if group!='pilot':entry['scores']={name:old['scores']['projected'] for name in ('projected','original','boundary')}
            report['cases'].append(entry);continue
        if group=='pilot':
            path=Path(old['video']);assert digest(path)==old['video_sha256'];key=(str(path),old['frame'])
            if key not in cache:
                cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,old['frame']);ok,im=cap.read();cap.release();assert ok
                cache[key]=cv2.resize(im,(960,round(im.shape[0]*960/im.shape[1])))
            image=cache[key];points=np.array(old['original_points']);expected=np.array(old['refined_points']);old_evidence=old;labels=None
        else:
            path=ROOT/'data/external/court_heatmap_pilot/images'/f'{identity}.png';assert digest(path)==old['input_sha256']
            image=cv2.resize(cv2.imread(str(path)),(960,540));points=np.array([x['prediction'] for x in old['scores']['projected']['landmarks']])*.75
            expected=np.array([x['prediction'] for x in old['scores']['refined']['landmarks']])*.75;old_evidence=old['refinement'];labels=[x['label'] for x in old['scores']['projected']['landmarks']]
        segments=[x['segment'] for x in propose(image) if x['support_fraction']>=.5]
        replay,replayed=original_refine(points,segments,image.shape[1],image.shape[0]);replay_difference=float(np.max(np.abs(replay-expected)))
        assert replayed['accepted']==old_evidence['accepted'] and replay_difference<1e-5,(identity,replay_difference)
        actual,evidence=refine(points,segments,image.shape[1],image.shape[0])
        entry.update(initial_valid=True,accepted=evidence['accepted'],reason=evidence['reason'],evidence=evidence,segments=segments,
            original_points=points.tolist(),previous_refined_points=expected.tolist(),boundary_refined_points=actual.tolist(),
            original_replay_maximum_difference_reference_px=replay_difference,
            changed_from_previous=bool(evidence['accepted']!=old_evidence['accepted'] or np.max(np.abs(actual-expected))>.1))
        if labels is not None:
            entry['scores']={name:score((p/.75).tolist(),labels) for name,p in [('projected',points),('original',expected),('boundary',actual)]}
            for name,oldname in [('projected','projected'),('original','refined')]:
                for k in ('tp','fp','fn'):assert entry['scores'][name][k]==old['scores'][oldname][k]
        panels=[]
        for name,candidate in [('Projected',points),('Previous refinement',expected),('Boundary constraint',actual)]:
            panel=cv2.resize(image,(640,round(image.shape[0]*640/960)))
            for a,b in LINES:cv2.line(panel,tuple(np.rint(candidate[a]*2/3).astype(int)),tuple(np.rint(candidate[b]*2/3).astype(int)),(30,30,230),1)
            if labels is not None:
                for label in labels:cv2.circle(panel,tuple(np.rint(np.array(label)*.5).astype(int)),4,(30,255,30),1)
            panel=cv2.copyMakeBorder(panel,32,0,0,0,cv2.BORDER_CONSTANT,value=(20,20,20))
            cv2.putText(panel,f'{index} {identity} {name}',(5,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            cv2.putText(panel,'Green unchanged label / red projected court',(5,27),cv2.FONT_HERSHEY_SIMPLEX,.35,(240,240,240),1);panels.append(panel)
        output=args.output/f'case_{index:02d}.jpg';assert cv2.imwrite(str(output),np.concatenate(panels),[cv2.IMWRITE_JPEG_QUALITY,85])
        entry.update(image=output.name,image_sha256=digest(output),visually_inspected=False);report['cases'].append(entry)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(index=index,id=identity,accepted=evidence['accepted'],changed=entry['changed_from_previous'],tp={k:v['tp'] for k,v in entry.get('scores',{}).items()})),flush=True)
    pooled={}
    for group in ('selection','external_source_check'):
        pooled[group]={}
        for stage in ('projected','original','boundary'):
            records=[x['scores'][stage] for x in report['cases'] if x['group']==group];counts={k:sum(x[k] for x in records) for k in ('tp','fp','fn')}
            errors=[p['distance_native_px'] for x in records for p in x['landmarks'] if p['eligible'] and p['distance_native_px'] is not None]
            pooled[group][stage]={**counts,'precision':counts['tp']/(counts['tp']+counts['fp']),'recall':counts['tp']/(counts['tp']+counts['fn']),'available':len(errors),'mean_error_px':float(np.mean(errors)),'median_error_px':float(np.median(errors))}
    report.update(complete=True,pooled=pooled);(args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(pooled),flush=True)

if __name__=='__main__':main()
