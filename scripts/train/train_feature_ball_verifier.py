"""Train a feature head using only the original training and selection clips."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from scripts.evaluate.replay_ball_temporal_detours import digest
from scripts.train.train_ball_candidate_verifier import training_partition, score_rows
from src.detection.feature_ball_verifier import FeatureBallVerifier, APPEARANCE_DIM, ENCODER_SHA256


def load_features(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if not manifest['complete'] or manifest['encoder_sha256'] != ENCODER_SHA256:
        raise ValueError('Require completed expected encoder cache')
    for name, checksum in manifest['code_hashes'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Feature extraction code changed')
    source_path = Path(manifest['source_manifest'])
    if digest(source_path) != manifest['source_manifest_sha256'] or digest(directory/'appearance.npy') != manifest['feature_sha256']:
        raise ValueError('Feature/source hash mismatch')
    source = json.loads(source_path.read_text())
    if source['split'] != manifest['split'] or source['candidate_count'] != manifest['candidate_count']:
        raise ValueError('Source identity mismatch')
    if digest(source_path.parent/'candidates.npz') != source['cache_sha256']:
        raise ValueError('Original cache changed')
    appearance = np.load(directory/'appearance.npy',allow_pickle=False)
    with np.load(source_path.parent/'candidates.npz',allow_pickle=False) as values:
        metadata, targets = values['features'], values['targets']
    count = source['candidate_count']
    expected_clips = sorted({r['clip'] for r in source['rows']})
    if sorted(manifest['completed_clips']) != expected_clips:
        raise ValueError('Incomplete or duplicated completed clips')
    seen_rows = {(r['clip'], r['frame']) for r in source['rows']}
    if len(seen_rows) != len(source['rows']):
        raise ValueError('Duplicate labeled source row')
    if appearance.shape != (count,APPEARANCE_DIM) or appearance.dtype != np.float16 or not np.isfinite(appearance).all():
        raise ValueError('Invalid feature matrix')
    if metadata.shape != (count,2) or not np.isfinite(metadata).all() or targets.shape != (count,) or not np.isin(targets,[-1,0,1]).all():
        raise ValueError('Invalid original arrays')
    seen = set()
    for row in source['rows']:
        for c in row['candidates']:
            index = c['sample_index']
            if index in seen or not 0<=index<count or c['training_target']!=targets[index]:
                raise ValueError('Candidate target mapping mismatch')
            seen.add(index)
    if seen != set(range(count)):
        raise ValueError('Missing candidates')
    return manifest, source, appearance, metadata, targets


def predict(model, appearance, metadata, indices, device):
    scores = np.full(len(appearance),np.nan,dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(indices),256):
            ids = indices[start:start+256]
            values = model(torch.from_numpy(appearance[ids]).to(device).float(),
                           torch.from_numpy(metadata[ids]).to(device).float()).sigmoid().cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError('Nonfinite score')
            scores[ids] = values
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    feature_manifest, source, appearance, metadata, targets = load_features(args.features)
    train_clips, heldout_clips = training_partition(source)
    train_rows = [r for r in source['rows'] if r['clip'] in train_clips]
    heldout_rows = [r for r in source['rows'] if r['clip'] in heldout_clips]
    ids = np.array([c['sample_index'] for r in train_rows for c in r['candidates'] if c['training_target']>=0],dtype=np.int64)
    heldout_ids = np.array([c['sample_index'] for r in heldout_rows for c in r['candidates']],dtype=np.int64)
    counts = np.bincount(targets[ids],minlength=2)
    if not counts.all() or len(ids)!=4719:
        raise ValueError('Expected frozen original training sample counts')
    torch.manual_seed(7)
    np.random.seed(7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeatureBallVerifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    dataset = TensorDataset(torch.from_numpy(appearance[ids]),torch.from_numpy(metadata[ids]),torch.from_numpy(targets[ids]))
    sampler = WeightedRandomSampler((1./counts[targets[ids]]).tolist(),len(ids),replacement=True)
    loader = DataLoader(dataset,batch_size=64,sampler=sampler,num_workers=0)
    args.output.mkdir(parents=True)
    run = {'complete':False,'qualification_evidence':False,'scope':'FROZEN_ENCODER_CONTEXT_PILOT_ORIGINAL_LABELS',
           'feature_manifest_sha256':digest(args.features/'manifest.json'),'encoder_sha256':ENCODER_SHA256,
           'feature_configuration':feature_manifest['configuration'],'source_manifest_sha256':feature_manifest['source_manifest_sha256'],
           'base_checkpoint_sha256':source['checkpoint_sha256'],'preparation_configuration':source['configuration'],
           'train_clips':train_clips,'heldout_clips':heldout_clips,'seed':7,'deterministic_algorithms':True,
           'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'device':device},
           'parameters':sum(p.numel() for p in model.parameters()),'epochs_requested':8,'batch_size':64,
           'training_class_counts':counts.tolist(),'draws_per_epoch':len(ids),'learning_rate':.001,'weight_decay':.0001,
           'thresholds':[.3,.5,.7,.9],'selection':'INTERNAL_FRAME_F1_THEN_PRECISION_THEN_RECALL',
           'code_hashes':{p:digest(ROOT/p) for p in ('scripts/train/train_feature_ball_verifier.py',
                    'src/detection/feature_ball_verifier.py','scripts/train/train_ball_candidate_verifier.py','src/evaluation/point_metrics.py')},'epochs':[]}
    best_key = (-1.,-1.,-1.)
    for epoch in range(1,9):
        started = time.perf_counter()
        model.train()
        total_loss, total = 0.,0
        for a,m,y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(a.to(device).float(),m.to(device).float())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,y.to(device).float())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total_loss += loss.item()*len(y)
            total += len(y)
        path = args.output/f'epoch_{epoch:02}.pt'
        torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'encoder_sha256':ENCODER_SHA256},path)
        scores = predict(model,appearance,metadata,heldout_ids,device)
        evaluations = []
        for threshold in run['thresholds']:
            metrics, rows = score_rows(heldout_rows,scores,threshold)
            evaluations.append({'threshold':threshold,'metrics':metrics})
            key = tuple(metrics[k] if metrics[k] is not None else -1. for k in ('f1','precision','recall'))
            if key>best_key:
                best_key = key
                run['selected'] = {'epoch':epoch,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path),
                                   'threshold':threshold,'internal_metrics':metrics,'labeled_rows':rows}
        run['epochs'].append({'epoch':epoch,'mean_loss':total_loss/total,'seconds':time.perf_counter()-started,
                              'checkpoint_sha256':digest(path),'evaluations':evaluations})
        run['complete'] = epoch==8
        (args.output/'manifest.json').write_text(json.dumps(run,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'epoch':epoch,'loss':total_loss/total,'selected_epoch':run['selected']['epoch'],
                          'threshold':run['selected']['threshold'],'internal_f1':best_key[0]}),flush=True)


if __name__=='__main__':
    main()
