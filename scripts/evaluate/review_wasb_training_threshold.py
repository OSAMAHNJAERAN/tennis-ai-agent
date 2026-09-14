"""Independently score and verify the completed training-only threshold comparison."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(rows):
    tp=fp=fn=tn=wrong=absent_fp=miss=0
    for row in rows:
        point,target=row['prediction_xy'],row['target_xy']
        if target is None:
            tn += int(point is None)
            fp += int(point is not None)
            absent_fp += int(point is not None)
        elif point is None:
            fn += 1
            miss += 1
        elif math.hypot((point[0]-target[0])*512/row['width'],(point[1]-target[1])*288/row['height']) <= 4:
            tp += 1
        else:
            fp += 1
            fn += 1
            wrong += 1
    return dict(true_positives=tp,false_positives=fp,false_negatives=fn,true_negatives=tn,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                absent_false_detections=absent_fp,wrong_visible_localizations=wrong,visible_abstentions=miss)


def choose(item):
    threshold,m=item
    return (m['f1'] if m['f1'] is not None else -1,m['precision'] if m['precision'] is not None else -1,
            m['recall'] if m['recall'] is not None else -1,-abs(float(threshold)-.2),-float(threshold))


def main():
    out=ROOT/'outputs/vision_upgrade_audit/wasb_training_threshold01'
    target=out/'review.json'
    if target.exists():
        raise FileExistsError(target)
    report=json.loads((out/'report.json').read_text())
    assert report['complete'] and len(report['models']) == 2
    assert report['protocol_sha256'] == digest(ROOT/'docs/experiments/WASB_TRAINING_THRESHOLD_PROTOCOL.md')
    for name,checksum in report['code_hashes'].items():
        assert digest(ROOT/name) == checksum
    training_path=ROOT/'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json'
    assert digest(training_path) == report['training_manifest_sha256']
    training=json.loads(training_path.read_text())
    comparisons=[]
    identities=None
    for model in report['models']:
        assert model['complete'] and len(model['clips']) == 8
        assert sorted(c['clip'] for c in model['clips']) == training['selection_clips']
        assert digest(ROOT/model['checkpoint']) == model['checkpoint_sha256']
        by_threshold={str(t):[] for t in report['thresholds']}
        current_identities=[]
        for clip in model['clips']:
            assert clip['original_point_and_all_candidates_exact']
            dataset=ROOT/clip['dataset']
            match,rally=clip['clip'].rsplit('_',1)
            video=dataset/f"tennis/videos/{clip['clip']}.mp4"
            labels_path=dataset/f'tennis/all/{match}/csv/{rally}_ball.csv'
            assert digest(video) == clip['video_sha256'] and digest(labels_path) == clip['label_sha256']
            with labels_path.open(newline='') as handle:
                labels={int(r['Frame']):r for r in csv.DictReader(handle)}
            assert len(labels) == len(clip['rows']) and set(labels) == {r['frame'] for r in clip['rows']}
            stride=clip['stride']
            assert stride == max(1,int(math.floor(clip['fps']/30+.5)))
            for row in clip['rows']:
                current_identities.append((clip['clip'],row['frame'],row['target_xy']))
                label=labels[row['frame']]
                expected=[float(label['X'])*clip['width']/1920,float(label['Y'])*clip['height']/1080] if int(label['Visibility']) else None
                assert expected == row['target_xy']
                windows=[]
                for slot in (2,1,0):
                    start=row['frame']-slot*stride
                    if start>=0 and start+2*stride<clip['frames']:
                        windows.append([[start,start+stride,start+2*stride],slot])
                assert windows and windows == row['windows']
                for threshold,result in row['thresholds'].items():
                    # Greedy suppression retains the highest-confidence proposal first.
                    proposals=result['candidates']
                    top=max(proposals,key=lambda p:p['confidence']) if proposals else None
                    assert result['prediction_xy'] == ([top['x'],top['y']] if top else None)
                    by_threshold[threshold].append(dict(clip=clip['clip'],frame=row['frame'],width=clip['width'],height=clip['height'],
                                                        target_xy=row['target_xy'],prediction_xy=result['prediction_xy']))
        assert len(current_identities) == 400
        if identities is None:
            identities=current_identities
        assert identities == current_identities
        scores={t:calculate(rows) for t,rows in by_threshold.items()}
        for t,metrics in scores.items():
            assert all(value == model['scores'][t]['pooled'][key] for key,value in metrics.items())
        selected=max(scores.items(),key=choose)[0]
        assert selected == model['selected_threshold']
        per_clip={clip:calculate([r for r in by_threshold[selected] if r['clip']==clip]) for clip in training['selection_clips']}
        comparisons.append(dict(model=model['name'],selected_threshold=selected,scores=scores,selected_per_clip=per_clip,
                                original_path_checks=8,seconds=sum(c['seconds'] for c in model['clips'])))
    original,adapted=comparisons
    original_metrics=original['scores'][original['selected_threshold']]
    adapted_metrics=adapted['scores'][adapted['selected_threshold']]
    # Compare measured detection metrics; threshold proximity is only a within-model tie-break.
    adapted_wins=tuple(adapted_metrics[k] or 0 for k in ('f1','precision','recall')) > tuple(original_metrics[k] or 0 for k in ('f1','precision','recall'))
    review=dict(complete=True,qualification_evidence=False,report_sha256=digest(out/'report.json'),
                reviewer_sha256=digest(Path(__file__)),independent_scoring_exact=True,labels_exact_and_paired=True,
                inference_reference_checks=16,models=comparisons,adapted_wins_internal_selection=adapted_wins,
                next_gate='FROZEN_EXTERNAL_PAIRED_COMPARISON' if adapted_wins else 'KEEP_ADAPTED_CHECKPOINT_REJECTED',
                limitation='Training-split model/threshold selection; no external accuracy qualification.')
    target.write_text(json.dumps(review,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(review,indent=2))


if __name__ == '__main__':
    main()
