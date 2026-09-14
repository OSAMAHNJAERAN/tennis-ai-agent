"""Select internal-only visual rejection thresholds while preserving proposal order."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.evaluate.replay_ball_temporal_detours import digest
from scripts.evaluate.compare_ball_resolution import compare
from scripts.evaluate.summarize_ball_validation import summarize
from scripts.train.train_feature_ball_verifier import load_features,predict
from src.detection.feature_ball_verifier import FeatureBallVerifier
from src.evaluation.point_metrics import evaluate_points


def accepted_order_rows(rows,scores,threshold):
    if not 0<threshold<1:
        raise ValueError('Invalid threshold')
    output=[]
    for row in rows:
        if [c['rank'] for c in row['candidates']]!=list(range(len(row['candidates']))):
            raise ValueError('Changed detector ordering')
        if any(not np.isfinite(scores[c['sample_index']]) for c in row['candidates']):
            raise ValueError('Missing candidate score')
        chosen=next((c for c in row['candidates'] if scores[c['sample_index']]>=threshold),None)
        output.append({k:row[k] for k in ('clip','frame','width','height','target_xy')}|
                      {'prediction_xy':[chosen['x'],chosen['y']] if chosen else None})
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--training-features',type=Path,required=True)
    parser.add_argument('--validation-features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    run_path=args.training_run/'manifest.json'
    run=json.loads(run_path.read_text())
    if not run['complete'] or [e['epoch'] for e in run['epochs']]!=list(range(1,9)):
        raise ValueError('Complete original training required')
    if digest(args.training_features/'manifest.json')!=run['feature_manifest_sha256']:
        raise ValueError('Training feature identity changed')
    for name,checksum in run['code_hashes'].items():
        if digest(ROOT/name)!=checksum:raise ValueError('Training source changed')
    fm,source,appearance,metadata,_=load_features(args.training_features)
    if fm['split']!='TRAINING_ONLY':raise ValueError('Training cache required')
    rows=[r for r in source['rows'] if r['clip'] in run['heldout_clips']]
    if set(r['clip'] for r in rows)!=set(run['heldout_clips']):raise ValueError('Missing internal selection clips')
    ids=np.array([c['sample_index'] for r in rows for c in r['candidates']],dtype=np.int64)
    torch.use_deterministic_algorithms(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=FeatureBallVerifier().to(device)
    best_key=(-1.,-1.,-1.);evaluations=[];selected=None
    args.output.mkdir(parents=True)
    for epoch in run['epochs']:
        path=args.training_run/f'epoch_{epoch["epoch"]:02}.pt'
        if digest(path)!=epoch['checkpoint_sha256']:raise ValueError('Head checkpoint changed')
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True)['model_state_dict'],strict=True)
        scores=predict(model,appearance,metadata,ids,device)
        for threshold in [.3,.5,.7,.9]:
            predictions=accepted_order_rows(rows,scores,threshold);metrics=evaluate_points(predictions)
            evaluations.append({'epoch':epoch['epoch'],'threshold':threshold,'metrics':metrics})
            key=tuple(metrics[k] if metrics[k] is not None else -1. for k in ('f1','precision','recall'))
            if key>best_key:
                best_key=key;selected={'epoch':epoch['epoch'],'threshold':threshold,'checkpoint':str(path.resolve()),
                                     'checkpoint_sha256':digest(path),'internal_metrics':metrics}
    selection={'complete':True,'qualification_evidence':False,'policy':'FIRST_ACCEPTED_ORIGINAL_DETECTOR_ORDER',
               'selection_scope':'ORIGINAL_FOUR_INTERNAL_TRAINING_CLIPS_ONLY','training_manifest_sha256':digest(run_path),
               'script_sha256':digest(__file__),'selected':selected,'evaluations':evaluations}
    selection_path=args.output/'selection.json'
    selection_path.write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
    del appearance,metadata
    vf,external,a,m,_=load_features(args.validation_features)
    if vf['split']!='VALIDATION_ONLY' or vf['configuration']!=fm['configuration'] or vf['encoder_sha256']!=fm['encoder_sha256']:
        raise ValueError('External feature configuration mismatch')
    if external['configuration']!=source['configuration'] or external['checkpoint_sha256']!=source['checkpoint_sha256']:
        raise ValueError('External proposal settings differ')
    if any(r['clip'] in run['train_clips']+run['heldout_clips'] for r in external['rows']):
        raise ValueError('External overlap')
    model.load_state_dict(torch.load(selected['checkpoint'],map_location=device,weights_only=True)['model_state_dict'],strict=True)
    scores=predict(model,a,m,np.arange(len(a)),device)
    predictions=accepted_order_rows(external['rows'],scores,selected['threshold'])
    result={'complete':True,'qualification_evidence':False,'policy':selection['policy'],
            'selection_sha256':digest(selection_path),'external_feature_manifest_sha256':digest(args.validation_features/'manifest.json'),
            'selected':selected,'labeled_rows':predictions,**summarize(predictions),'candidate_scores':scores.tolist()}
    (args.output/'external.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'selected_epoch':selected['epoch'],'threshold':selected['threshold'],
                     'pooled':{k:v for k,v in result['pooled'].items() if k!='localization_errors_reference_px'}}))


if __name__=='__main__':main()
