"""Extract frozen features from hash-verified existing candidate observations."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from scripts.evaluate.replay_ball_temporal_detours import digest
from src.detection.ball_candidate_verifier import reference_rgb
from src.detection.feature_ball_verifier import context_patch, ENCODER_SHA256, APPEARANCE_DIM
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_path = args.source_cache/'manifest.json'
    source = json.loads(source_path.read_text())
    if source['split'] not in ('TRAINING_ONLY', 'VALIDATION_ONLY') or digest(args.source_cache/'candidates.npz') != source['cache_sha256']:
        raise ValueError('Invalid source cache')
    dataset = Path(source['dataset'])
    manifest_path = dataset/'manifest.json'
    if digest(manifest_path) != source['dataset_manifest_sha256'] or digest(args.encoder) != ENCODER_SHA256:
        raise ValueError('Dataset or encoder changed')
    for item in json.loads(manifest_path.read_text())['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset file changed')
    count = source['candidate_count']
    if sorted(c['sample_index'] for r in source['rows'] for c in r['candidates']) != list(range(count)):
        raise ValueError('Invalid candidate mapping')
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = resnet50(weights=None)
    encoder.load_state_dict(torch.load(args.encoder, map_location='cpu', weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False).eval().to(device)
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    args.output.mkdir(parents=True)
    feature_path = args.output/'appearance.npy'
    features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float16, shape=(count, APPEARANCE_DIM))
    written = np.zeros(count, dtype=bool)
    result = {'complete': False, 'qualification_evidence': False, 'split': source['split'],
              'source_manifest': str(source_path.resolve()), 'source_manifest_sha256': digest(source_path),
              'encoder_sha256': ENCODER_SHA256, 'candidate_count': count, 'appearance_dimension': APPEARANCE_DIM,
              'configuration': {'reference_size': [960,540], 'context_side': 64, 'transform': 'IMAGENET1K_V2_OFFICIAL',
                                'features': 'CURRENT_PAST_ABSOLUTE_DIFFERENCE', 'storage_dtype': 'float16', 'encoder_frozen': True},
              'code_hashes': {p: digest(ROOT/p) for p in ('scripts/data/prepare_ball_context_features.py',
                 'src/detection/feature_ball_verifier.py', 'src/detection/ball_candidate_verifier.py', 'src/utils/video_frame_sequence.py')},
              'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__), 'torchvision': torchvision.__version__,
                          'numpy': np.__version__, 'device': device, 'deterministic_algorithms': True}, 'completed_clips': []}
    output_manifest = args.output/'manifest.json'
    output_manifest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    started = time.perf_counter()
    for clip in sorted({r['clip'] for r in source['rows']}):
        with VideoFrameSequence(str(dataset/f'tennis/videos/{clip}.mp4')) as frames:
            for row in sorted((r for r in source['rows'] if r['clip']==clip), key=lambda r:r['frame']):
                if row['width']!=frames.metadata.width or row['height']!=frames.metadata.height or not 0<=row['past_frame']<=row['frame']<len(frames):
                    raise ValueError('Frame context mismatch')
                candidates = row['candidates']
                if not candidates:
                    continue
                current, past = reference_rgb(frames[row['frame']]), reference_rgb(frames[row['past_frame']])
                patches = [context_patch(frame, [c['x'], c['y']], [row['width'], row['height']]) for frame in (current,past) for c in candidates]
                values = []
                with torch.inference_mode():
                    for start in range(0,len(patches),32):
                        images = torch.from_numpy(np.stack(patches[start:start+32])).to(device).float()/255
                        values.append(encoder(transform(images)).cpu().numpy())
                encoded = np.concatenate(values)
                n = len(candidates)
                vectors = np.concatenate([encoded[:n],encoded[n:],np.abs(encoded[:n]-encoded[n:])],axis=1).astype(np.float16)
                if not np.isfinite(vectors).all():
                    raise FloatingPointError('Nonfinite features')
                ids = [c['sample_index'] for c in candidates]
                features[ids], written[ids] = vectors, True
        features.flush()
        result['completed_clips'].append(clip)
        output_manifest.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'clip':clip,'encoded_candidates':int(written.sum())}),flush=True)
    if not written.all():
        raise ValueError('Incomplete extraction')
    features.flush()
    del features
    result.update(complete=True,feature_sha256=digest(feature_path),seconds_excluding_encoder_load=time.perf_counter()-started)
    output_manifest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'complete':True,'candidates':count,'seconds':result['seconds_excluding_encoder_load']}),flush=True)


if __name__=='__main__':
    main()
