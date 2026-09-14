"""Compare frozen heatmap operating points on reserved training matches only."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.train.finetune_wasb_spatial import split_training_clips
from scripts.evaluate.benchmark_ball_temporal_spacing import aligned_windows, infer_candidates
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.detection.tiled_wasb_candidates import overlapping_tiles, merge_by_confidence
from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.video_frame_sequence import VideoFrameSequence

THRESHOLDS = (.10,.20,.35,.50,.70)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer_thresholds(detector,frames,target,stride):
    windows = aligned_windows(len(frames),target,stride)
    context = {i:frames[i] for i in sorted({i for indices,_ in windows for i in indices})}
    height,width = context[target].shape[:2]
    boxes = [(0,0,width,height),*overlapping_tiles(width,height,.6)]
    proposals = {str(t):[] for t in THRESHOLDS}
    previous = detector.threshold
    try:
        for view,(left,top,right,bottom) in enumerate(boxes):
            maps = []
            for indices,slot in windows:
                heatmaps,shape = detector.predict_heatmaps([context[i][top:bottom,left:right] for i in indices])
                assert len(heatmaps) == 3
                maps.append(heatmaps[slot])
            averaged = np.mean(maps,axis=0)
            for threshold in THRESHOLDS:
                detector.threshold = threshold
                decoded = detector.decode_heatmaps([averaged],shape)[0]
                proposals[str(threshold)].extend(dict(view=view,box=[left,top,right,bottom],mass_rank=rank,
                                                       x=p.x_px+left,y=p.y_px+top,confidence=p.confidence)
                                                 for rank,p in enumerate(decoded))
    finally:
        detector.threshold = previous
    results = {}
    for threshold,candidates in proposals.items():
        merged = merge_by_confidence([BallObservation(c['x'],c['y'],c['confidence']) for c in candidates],(width,height))
        results[threshold] = dict(candidates=candidates,prediction_xy=[merged[0].x_px,merged[0].y_px] if merged else None)
    return results,windows


def selection_key(item):
    threshold, metrics = item
    m = metrics['pooled']
    return (m['f1'] if m['f1'] is not None else -1,m['precision'] if m['precision'] is not None else -1,
            m['recall'] if m['recall'] is not None else -1,-abs(float(threshold)-.2),-float(threshold))


def main():
    out = ROOT/'outputs/vision_upgrade_audit/wasb_training_threshold01'
    if out.exists():
        raise FileExistsError(out)
    training_path = ROOT/'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json'
    training = json.loads(training_path.read_text())
    assert training['complete']
    for path,checksum in training['code_hashes'].items():
        assert digest(Path(path)) == checksum
    manifests,datasets = [],[]
    for name,checksum in training['dataset_manifests'].items():
        dataset = ROOT/name
        assert digest(dataset/'manifest.json') == checksum
        manifest = json.loads((dataset/'manifest.json').read_text())
        for item in manifest['files']:
            assert digest(dataset/item['path']) == item['sha256']
        manifests.append(manifest)
        datasets.append(dataset)
    split = split_training_clips(manifests)
    assert sorted(k for k,v in split.items() if v == 'selection') == training['selection_clips']
    selected = []
    for dataset,manifest in zip(datasets,manifests,strict=True):
        publisher_train = {tuple(x) for x in json.loads((dataset/'tennis/info/train.json').read_text())}
        publisher_val = {x[0] for x in json.loads((dataset/'tennis/info/val.json').read_text())}
        for match,rally in manifest['selected_clips']:
            assert (match,rally) in publisher_train and match not in publisher_val
            if split[f'{match}_{rally}'] == 'selection':
                selected.append((dataset,match,rally))
    assert len(selected) == 8
    models = [('original',ROOT/training['initial_checkpoint'],training['initial_checkpoint_sha256']),
              ('spatial_epoch03',ROOT/training['selected_checkpoint'],training['epochs'][-1]['checkpoint_sha256'])]
    assert training['selected_checkpoint'] == training['epochs'][-1]['checkpoint']
    out.mkdir(parents=True)
    report = dict(complete=False,qualification_evidence=False,scope='TRAINING_SPLIT_THRESHOLD_SELECTION',
                  training_manifest_sha256=digest(training_path),protocol_sha256=digest(ROOT/'docs/experiments/WASB_TRAINING_THRESHOLD_PROTOCOL.md'),
                  code_hashes={p:digest(ROOT/p) for p in ['scripts/evaluate/select_wasb_training_threshold.py',
                      'src/detection/wasb_ball_detector.py','src/detection/tiled_wasb_candidates.py',
                      'scripts/evaluate/benchmark_ball_temporal_spacing.py','src/evaluation/point_metrics.py']},
                  thresholds=list(THRESHOLDS),models=[])
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for name,checkpoint,checksum in models:
        assert digest(checkpoint) == checksum
        detector = WASBBallDetector(checkpoint,threshold=.2,device='cuda')
        model = dict(name=name,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=checksum,complete=False,clips=[])
        report['models'].append(model)
        for dataset,match,rally in selected:
            clip = f'{match}_{rally}'
            video = dataset/f'tennis/videos/{clip}.mp4'
            label_path = dataset/f'tennis/all/{match}/csv/{rally}_ball.csv'
            with label_path.open(newline='') as handle:
                labels = list(csv.DictReader(handle))
            started = time.perf_counter()
            result = dict(clip=clip,dataset=str(dataset.relative_to(ROOT)),video_sha256=digest(video),label_sha256=digest(label_path),rows=[])
            with VideoFrameSequence(str(video),cache_size=12) as frames:
                width,height,fps = frames.metadata.width,frames.metadata.height,frames.metadata.fps
                stride = max(1,int(math.floor(fps/30+.5)))
                result.update(width=width,height=height,fps=fps,stride=stride,frames=len(frames))
                seen = set()
                for label in labels:
                    index,visible = int(label['Frame']),int(label['Visibility'])
                    assert index not in seen and 0 <= index < len(frames) and visible in (0,1)
                    seen.add(index)
                    predictions,windows = infer_thresholds(detector,frames,index,stride)
                    if len(seen) == 1:
                        original,candidates,original_windows = infer_candidates(detector,frames,index,stride)
                        reference = [original[0].x_px,original[0].y_px] if original else None
                        assert predictions['0.2']['prediction_xy'] == reference
                        assert predictions['0.2']['candidates'] == candidates and windows == original_windows
                        result['original_point_and_all_candidates_exact'] = True
                    target = [float(label['X'])*width/1920,float(label['Y'])*height/1080] if visible else None
                    result['rows'].append(dict(frame=index,target_xy=target,windows=windows,thresholds=predictions))
            result['seconds'] = time.perf_counter()-started
            model['clips'].append(result)
            save()
            print(json.dumps(dict(model=name,clip=clip,labels=len(labels),seconds=result['seconds'])),flush=True)
        assert sum(len(c['rows']) for c in model['clips']) == 400
        scores = {threshold:summarize([dict(clip=c['clip'],frame=r['frame'],width=c['width'],height=c['height'],
                                           target_xy=r['target_xy'],prediction_xy=r['thresholds'][threshold]['prediction_xy'])
                                      for c in model['clips'] for r in c['rows']]) for threshold in map(str,THRESHOLDS)}
        model.update(complete=True,scores=scores,selected_threshold=max(scores.items(),key=selection_key)[0])
        save()
        print(json.dumps(dict(model=name,selected_threshold=model['selected_threshold'],metrics={k:v for k,v in scores[model['selected_threshold']]['pooled'].items() if k != 'localization_errors_reference_px'})),flush=True)
        del detector
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
