"""Frozen sparse-label TOTNet pilot; no production integration or threshold fitting."""
import argparse
from collections import deque
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/research/TOTNet'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE / 'vendor'))

import cv2
import numpy as np
import torch
from src.evaluation.point_metrics import evaluate_points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(frame):
    resized = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_LANCZOS4)
    normalized = (resized / 255. - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
    return normalized.transpose(2, 0, 1).astype(np.float32)


def metrics(rows):
    return {str(t): evaluate_points(rows, tolerance_px=t) for t in (2., 4., 8.)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/external/racketvision_validation')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/vision_upgrade_audit/totnet_ball_pilot01')
    args = parser.parse_args()
    if (args.output/'report.json').exists():
        raise FileExistsError('Inspect existing run; do not silently overwrite')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['split'] == 'VALIDATION_ONLY'
    for entry in manifest['files']:
        assert digest(args.dataset/entry['path']) == entry['sha256'], entry['path']
    source_manifest = json.loads((SOURCE/'source_manifest.json').read_text(encoding='utf-8'))
    for entry in source_manifest['files']:
        assert digest(SOURCE/'source'/entry['path']) == entry['sha256']
    checkpoint = ROOT/'artifacts/models/ball/totnet_research/tennis_state_dict.pt'
    assert digest(checkpoint) == '66395b83760d533e8c9a43eb3b5a00e7b24a018d74f6d9232489dae6224dd7a2'
    official = import_file('totnet_official', SOURCE/'source/src/model/TOTNet.py')
    model = official.TemporalConvNet(input_shape=(288,512), spatial_channels=64, num_frames=5)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval().to('cuda')
    torch.backends.cudnn.benchmark = False
    transform = import_file('totnet_official_transform', SOURCE/'source/src/data_process/transformation.py')
    sample = np.random.default_rng(2024).integers(0,256,(359,641,3),dtype=np.uint8)
    resized, xy, visible = transform.Resize((288,512),p=1)([sample],np.array([30.,40.]),1)
    normalized, _, _ = transform.Normalize(p=1)(resized,xy,visible)
    assert np.array_equal(preprocess(sample), normalized[0].transpose(2,0,1).astype(np.float32))
    report = {'complete':False, 'qualification_evidence':False, 'dataset_manifest_sha256':digest(manifest_path),
              'checkpoint_sha256':digest(checkpoint), 'source_revision':source_manifest['revision'],
              'script_sha256':digest(Path(__file__)), 'official_preprocess_pixel_exact':True,
              'torch_version':torch.__version__, 'cv2_version':cv2.__version__,
              'protocol':'docs/experiments/TOTNET_BALL_PILOT.md', 'inference_scope':'explicit labeled frames with five real consecutive inputs',
              'input_shape':[1,5,3,288,512], 'target':'last frame', 'clips':[]}
    all_raw, all_absent = [], []
    def save():
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for match,rally in manifest['selected_clips']:
        csv_path = args.dataset/f'tennis/all/{match}/csv/{rally}_ball.csv'
        with csv_path.open(newline='',encoding='utf-8') as f:
            labels = {int(row['Frame']):row for row in csv.DictReader(f)}
        video = args.dataset/f'tennis/videos/{match}_{rally}.mp4'
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS); expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        history = deque(maxlen=5); rows = []; absent_rows = []; predictions = []
        frame_index=0; elapsed_inference=0.; boundary=0; start=time.perf_counter()
        with torch.inference_mode():
            while True:
                ok, frame = cap.read()
                if not ok: break
                height,width = frame.shape[:2]
                history.append(preprocess(frame))
                if frame_index in labels:
                    label=labels[frame_index]; point=None; peak=None; grid=None
                    if len(history)==5:
                        tensor=torch.from_numpy(np.stack(history)[None]).to('cuda')
                        torch.cuda.synchronize(); t=time.perf_counter()
                        output=model(tensor)
                        torch.cuda.synchronize(); elapsed_inference+=time.perf_counter()-t
                        assert output.shape==(1,512*288) and torch.isfinite(output).all()
                        assert abs(output.sum().item()-1.)<1e-4
                        index=int(output.argmax().item()); peak=float(output[0,index].item())
                        grid=[index%512,index//512]; point=[grid[0]*width/512,grid[1]*height/288]
                    else: boundary+=1
                    target=[float(label['X'])*width/1920,float(label['Y'])*height/1080] if int(label['Visibility']) else None
                    row={'frame_index':frame_index,'width':width,'height':height,'target_xy':target,'prediction_xy':point}
                    rows.append(row); absent_rows.append({**row,'prediction_xy':None if grid==[0,0] else point})
                    predictions.append({**row,'peak_probability':peak,'grid_xy':grid,
                                        'input_frame_indices':list(range(frame_index-4,frame_index+1)) if len(history)==5 else []})
                frame_index+=1
        cap.release()
        assert frame_index==expected and len(rows)==len(labels)
        all_raw.extend(rows); all_absent.extend(absent_rows)
        clip={'match':match,'rally':rally,'decoded_frames':frame_index,'fps':fps,'labels':len(rows),
              'boundary_abstentions':boundary,'model_inference_seconds':elapsed_inference,
              'decode_preprocess_inference_seconds':time.perf_counter()-start,
              'raw_argmax_metrics':metrics(rows),'origin_absent_metrics':metrics(absent_rows),'predictions':predictions}
        report['clips'].append(clip); save()
        print(json.dumps({'clip':match,'labeled_frames':len(rows),'raw_4px':clip['raw_argmax_metrics']['4.0'],
                          'inference_seconds':elapsed_inference}),flush=True)
    report.update(complete=True,raw_argmax_metrics=metrics(all_raw),origin_absent_metrics=metrics(all_absent),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
    save()
    print(json.dumps({'complete':True,'raw_4px':report['raw_argmax_metrics']['4.0'],
                      'origin_absent_4px':report['origin_absent_metrics']['4.0']}),flush=True)


if __name__=='__main__':
    main()
