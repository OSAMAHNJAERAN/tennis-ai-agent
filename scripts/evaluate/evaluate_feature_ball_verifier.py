"""Score a frozen training-selected pretrained-context head on original labels."""
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
from scripts.evaluate.summarize_ball_validation import summarize
from scripts.train.train_feature_ball_verifier import load_features, predict
from scripts.train.train_ball_candidate_verifier import score_rows
from src.detection.feature_ball_verifier import FeatureBallVerifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fm, source, appearance, metadata, _ = load_features(args.features)
    path = args.training_run/'manifest.json'
    run = json.loads(path.read_text())
    if not run['complete'] or [e['epoch'] for e in run['epochs']]!=list(range(1,9)) or fm['split']!='VALIDATION_ONLY':
        raise ValueError('Need complete training and separate validation')
    if fm['configuration']!=run['feature_configuration'] or fm['encoder_sha256']!=run['encoder_sha256'] or source['configuration']!=run['preparation_configuration'] or source['checkpoint_sha256']!=run['base_checkpoint_sha256']:
        raise ValueError('Feature/proposal configuration mismatch')
    if any(r['clip'] in set(run['train_clips']+run['heldout_clips']) for r in source['rows']):
        raise ValueError('Training/validation overlap')
    selected = run['selected']
    if digest(selected['checkpoint'])!=selected['checkpoint_sha256']:
        raise ValueError('Selected head changed')
    for name,checksum in run['code_hashes'].items():
        if digest(ROOT/name)!=checksum:
            raise ValueError('Training/scoring code changed')
    torch.use_deterministic_algorithms(True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeatureBallVerifier().to(device)
    model.load_state_dict(torch.load(selected['checkpoint'],map_location=device,weights_only=True)['model_state_dict'],strict=True)
    scores = predict(model,appearance,metadata,np.arange(len(appearance)),device)
    metrics,rows = score_rows(source['rows'],scores,selected['threshold'])
    result = {'complete':True,'qualification_evidence':False,'scope':'FROZEN_CONTEXT_HEAD_REUSED_EXTERNAL_LABELS_UNCHANGED',
              'training_manifest_sha256':digest(path),'feature_manifest_sha256':digest(args.features/'manifest.json'),
              'evaluator_sha256':digest(__file__),'selected_checkpoint_sha256':selected['checkpoint_sha256'],
              'threshold':selected['threshold'],'metrics':metrics,'strata':summarize(rows),'labeled_rows':rows,
              'candidate_scores':scores.tolist(),'weights_only_loading':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k!='localization_errors_reference_px'}))


if __name__=='__main__':
    main()
