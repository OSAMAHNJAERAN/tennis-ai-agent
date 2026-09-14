"""Collect original-model response evidence on a frozen training-selection subset."""
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts.evaluate.benchmark_ball_temporal_spacing import aligned_windows
from src.detection.tiled_wasb_candidates import overlapping_tiles, merge_by_confidence
from src.detection.wasb_ball_detector import WASBBallDetector
from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.video_frame_sequence import VideoFrameSequence

OUT = Path('outputs/vision_upgrade_audit/wasb_proposal_response01')
SOURCE = Path('artifacts/training/vision_upgrade/wasb_spaced_head_pilot05/manifest.json')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(point, target, width, height):
    return math.hypot((point[0]-target[0])*512/width, (point[1]-target[1])*288/height)


def response_stats(maps, inverse, box, target, width, height, threshold=.2):
    left, top, right, bottom = box
    yy, xx = np.mgrid[:maps.shape[1], :maps.shape[2]]
    x = inverse[0, 0]*xx + inverse[0, 1]*yy + inverse[0, 2] + left
    y = inverse[1, 0]*xx + inverse[1, 1]*yy + inverse[1, 2] + top
    valid = (x >= left) & (x < right) & (y >= top) & (y < bottom)
    averaged = maps.mean(axis=0)
    result = dict(average_global_peak=float(averaged[valid].max()),
                  slot_global_peaks=[float(m[valid].max()) for m in maps])
    if target is None:
        return result
    mask = valid & (((x-target[0])*512/width)**2 + ((y-target[1])*288/height)**2 <= 16)
    result.update(tolerance_pixel_count=int(mask.sum()),
                  average_target_peak=float(averaged[mask].max()) if mask.any() else None,
                  slot_target_peaks=[float(m[mask].max()) for m in maps] if mask.any() else [],
                  average_target_active_pixels=int(((averaged > threshold) & mask).sum()))
    return result


def category(views):
    if any(v.get('average_target_active_pixels', 0) for v in views):
        return 'COMPONENT_DISPLACEMENT'
    if any(any(p > .2 for p in v.get('slot_target_peaks', [])) for v in views):
        return 'TEMPORAL_ATTENUATION'
    return 'WEAK_RESPONSE'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    source = json.loads(SOURCE.read_text())
    assert source['complete'] and not source['advance']
    original = next(c for c in source['comparators'] if c['name'] == 'original')
    selection = []
    for clip in original['clips']:
        correct_control = absent_control = False
        for row in sorted(clip['rows'], key=lambda r:r['frame']):
            target = row['target_xy']
            covered = target is not None and any(error((p['x'],p['y']),target,clip['width'],clip['height']) <= 4 for p in row['candidates'])
            role = None
            if target is not None and not covered:
                role = 'PROPOSAL_MISS'
            elif target is not None and not correct_control and row['prediction_xy'] is not None and error(row['prediction_xy'],target,clip['width'],clip['height']) <= 4:
                role = 'CORRECT_CONTROL'
                correct_control = True
            elif target is None and not absent_control:
                role = 'ABSENT_CONTROL'
                absent_control = True
            if role:
                selection.append(dict(clip=clip['clip'], dataset=clip['dataset'], width=clip['width'],
                                      height=clip['height'], fps=clip['fps'], stride=clip['stride'],
                                      video_sha256=clip['video_sha256'], label_sha256=clip['label_sha256'], role=role, row=row))
    OUT.mkdir(parents=True)
    selected = dict(source_sha256=digest(SOURCE), protocol_sha256=digest('docs/experiments/WASB_PROPOSAL_RESPONSE_PROTOCOL.md'),entries=selection)
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    checkpoint = Path('artifacts/models/ball/wasb_tennis_best.pth.tar')
    assert digest(checkpoint) == source['provenance']['initial_checkpoint_sha256']
    for p,h in source['provenance']['code_hashes'].items():
        assert digest(p) == h, p
    assert torch.cuda.is_available()
    detector = WASBBallDetector(checkpoint, threshold=.2)
    report = dict(complete=False, qualification_evidence=False, selection_sha256=digest(OUT/'selection.json'),
                  checkpoint_sha256=digest(checkpoint),code_hashes={p:digest(p) for p in [__file__,
                      'src/detection/wasb_ball_detector.py','src/detection/tiled_wasb_candidates.py',
                      'src/utils/video_frame_sequence.py','artifacts/research/WASB-SBDT/src/utils/image.py']},entries=[])
    def save():
        (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    for clip in sorted({e['clip'] for e in selection}):
        group = [e for e in selection if e['clip'] == clip]
        dataset = Path(group[0]['dataset'])
        video = dataset/'tennis/videos'/f'{clip}.mp4'
        assert digest(video) == group[0]['video_sha256']
        label = dataset/'tennis/ball'/f'{clip}.csv'
        # Locate the exact publisher label path through its verified dataset inventory.
        manifest = json.loads((dataset/'manifest.json').read_text())
        labels = [dataset/f['path'] for f in manifest['files'] if f['sha256'] == group[0]['label_sha256']]
        assert len(labels) == 1 and digest(labels[0]) == group[0]['label_sha256']
        frames = VideoFrameSequence(str(video))
        started = time.perf_counter()
        try:
            for entry in group:
                row = entry['row']; target = row['target_xy']; index = row['frame']
                windows = aligned_windows(len(frames),index,entry['stride'])
                assert json.loads(json.dumps(windows)) == row['windows']
                context = {i:frames[i] for i in sorted({i for indices,_ in windows for i in indices})}
                width,height = entry['width'],entry['height']
                boxes = [(0,0,width,height),*overlapping_tiles(width,height,.6)]
                proposals=[]; views=[]; cache={}
                for view,box in enumerate(boxes):
                    left,top,right,bottom = box; maps=[]
                    for indices,slot in windows:
                        heatmaps,shape = detector.predict_heatmaps([context[i][top:bottom,left:right] for i in indices])
                        maps.append(heatmaps[slot])
                    maps=np.stack(maps)
                    inverse=detector.geometry.get_affine_transform(np.array([(right-left)/2,(bottom-top)/2],np.float32),max(right-left,bottom-top),0,(512,288),inv=1)
                    stats=response_stats(maps,inverse,box,target,width,height)
                    slots=[]
                    for slot_map in maps:
                        candidates=detector.decode_heatmaps([slot_map],shape)[0]
                        slots.append(target is not None and any(error((p.x_px+left,p.y_px+top),target,width,height)<=4 for p in candidates))
                    stats.update(view=view,box=list(box),inverse=inverse.tolist(),slot_correct_proposals=slots)
                    views.append(stats);cache[f'view{view}']=maps
                    for rank,p in enumerate(detector.decode_heatmaps([maps.mean(axis=0)],shape)[0]):
                        proposals.append(dict(view=view,box=list(box),mass_rank=rank,x=p.x_px+left,y=p.y_px+top,confidence=p.confidence))
                assert len(proposals)==len(row['candidates'])
                for got,want in zip(proposals,row['candidates']):
                    assert all(got[k]==want[k] for k in ('view','box','mass_rank'))
                    assert all(abs(got[k]-want[k])<=1e-5 for k in ('x','y','confidence')),(clip,index,got,want)
                merged=merge_by_confidence([BallObservation(p['x'],p['y'],p['confidence']) for p in proposals],(width,height))
                prediction=[merged[0].x_px,merged[0].y_px] if merged else None
                assert (prediction is None)==(row['prediction_xy'] is None)
                if prediction is not None: assert max(abs(a-b) for a,b in zip(prediction,row['prediction_xy']))<=1e-5
                path=OUT/f'{clip}_{index:06}.npz';np.savez_compressed(path,**cache)
                report['entries'].append(dict(clip=clip,frame=index,role=entry['role'],target_xy=target,
                    views=views,category=category(views) if entry['role']=='PROPOSAL_MISS' else entry['role'],
                    heatmaps=path.name,heatmaps_sha256=digest(path),prediction_reproduced=True))
                save()
        finally:
            frames.close()
        print(json.dumps(dict(clip=clip,labels=len(group),seconds=time.perf_counter()-started)),flush=True)
    report.update(complete=True,counts={k:sum(e['category']==k for e in report['entries']) for k in sorted({e['category'] for e in report['entries']})})
    save(); print(json.dumps(dict(complete=True,counts=report['counts'])))


if __name__ == '__main__':
    main()
