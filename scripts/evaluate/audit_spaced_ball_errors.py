"""Attribute remaining errors without changing labels, candidates or predictions."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.review_ball_spacing_final import load_final


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def distance(point, target, width, height):
    return math.hypot((point[0]-target[0])*512/width, (point[1]-target[1])*288/height)


def outcome(point, target, width, height):
    if target is None:
        return 'absent_false_detection' if point is not None else 'correct_absence'
    if point is None:
        return 'visible_abstention'
    return 'correct_ball' if distance(point,target,width,height) <= 4 else 'wrong_location'


def main():
    base = ROOT/'artifacts/validation/vision_upgrade'
    out = ROOT/'outputs/vision_upgrade_audit/ball_spaced_error_audit01'
    if out.exists():
        raise FileExistsError(out)
    sources, rows = {}, []
    def source(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return read(path)
    for group, folder in [('expansion12','racketvision_validation_expansion12'),('additional6','racketvision_validation_additional6')]:
        final_path = base/f'{group}_temporal_spacing_detours01.json'
        final = source(final_path)
        verified, stream, clips = load_final(final_path)
        assert verified == final
        sparse = source(base/f'{group}_temporal_spacing_pilot01.json')
        assert stream['sparse_sha256'] == digest(base/f'{group}_temporal_spacing_pilot01.json')
        dataset = ROOT/'data/external'/folder
        manifest = source(dataset/'manifest.json')
        assert digest(dataset/'manifest.json') == sparse['dataset_manifest_sha256'] == stream['dataset_manifest_sha256']
        for item in manifest['files']:
            assert digest(dataset/item['path']) == item['sha256']
        details = {c['clip']:{r['frame']:r for r in c['candidate_details']} for c in sparse['clips']}
        for clip_name, clip in clips.items():
            match, rally = clip_name.rsplit('_',1)
            with (dataset/f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as handle:
                original_labels = {int(r['Frame']):r for r in csv.DictReader(handle)}
            scored = [r for r in final['labeled_rows'] if r['clip'] == clip_name]
            assert set(original_labels) == {r['frame'] for r in scored}
            for row in scored:
                index, target = row['frame'], row['target_xy']
                width,height = row['width'],row['height']
                label = original_labels[index]
                expected_target = [float(label['X'])*width/1920,float(label['Y'])*height/1080] if int(label['Visibility']) else None
                assert target == expected_target
                points = {name:values[index] for name,values in clip['predictions'].items()}
                points['final'] = clip['final_points'][index]
                assert points['final'] == row['prediction_xy']
                outcomes = {name:outcome(point,target,width,height) for name,point in points.items()}
                detail = details[clip_name].get(index)
                candidate_info = None
                if detail is not None:
                    ordered = sorted(detail['candidates'],key=lambda c:-c['confidence'])
                    correct = [i for i,c in enumerate(ordered) if target is not None and distance([c['x'],c['y']],target,width,height) <= 4]
                    assert detail['has_correct_candidate'] == (bool(correct) if target is not None else None)
                    candidate_info = dict(count=len(ordered),correct_count=len(correct),
                                          first_correct_confidence_rank=correct[0]+1 if correct else None,
                                          best_confidence=ordered[0]['confidence'] if ordered else None,
                                          correct_candidate_confidence=ordered[correct[0]]['confidence'] if correct else None)
                category = outcomes['final']
                if category in ('visible_abstention','wrong_location'):
                    if outcomes['raw'] == 'correct_ball':
                        category = 'raw_correct_lost_by_filters'
                    elif candidate_info is None:
                        category = 'raw_miss_candidate_coverage_unknown'
                    elif candidate_info['correct_count']:
                        category = 'raw_miss_correct_candidate_available'
                    else:
                        category = 'raw_miss_no_correct_candidate'
                nearby = {str(j):clip['final_points'][j] for j in range(max(0,index-1),min(clip['frames'],index+2))}
                rows.append(dict(**row,dataset=folder,fps=clip['fps'],stride=clip['stride'],
                                 stages=points,outcomes=outcomes,category=category,candidates=candidate_info,
                                 adjacent_final_points=nearby))
    assert len(rows) == 900 and len({(r['clip'],r['frame']) for r in rows}) == 900
    counts = Counter(r['outcomes']['final'] for r in rows)
    tp,wrong,missing,afp,tn = [counts[k] for k in ('correct_ball','wrong_location','visible_abstention','absent_false_detection','correct_absence')]
    assert (tp,wrong+afp,wrong+missing,tn) == (803,43,26,42)
    categories = Counter(r['category'] for r in rows)
    errors = sorted([r for r in rows if r['outcomes']['final'] not in ('correct_ball','correct_absence')],key=lambda r:(r['clip'],r['frame']))
    selected, used = [], Counter()
    for row in errors:
        if used[row['category']] < 2:
            selected.append(dict(clip=row['clip'],frame=row['frame'],category=row['category']))
            used[row['category']] += 1
    transitions = {}
    for a,b in [('raw','stationary'),('stationary','pixel_motion'),('pixel_motion','final')]:
        transitions[a+'->'+b] = dict(Counter(r['outcomes'][a]+'->'+r['outcomes'][b] for r in rows))
    report = dict(complete=True,qualification_evidence=False,sources=sources,
                  script_sha256=digest(Path(__file__)),protocol_sha256=digest(ROOT/'docs/experiments/BALL_SPACED_ERROR_AUDIT_PROTOCOL.md'),
                  labels_exact=True,final_stream_reconstruction_exact=True,
                  metrics=dict(tp=tp,fp=wrong+afp,fn=wrong+missing,tn=tn,precision=tp/(tp+wrong+afp),recall=tp/(tp+wrong+missing)),
                  outcomes=dict(counts),categories=dict(categories),transitions=transitions,
                  candidate_evidence_frames=sum(r['candidates'] is not None for r in rows),
                  candidate_evidence_visible_frames=sum(r['candidates'] is not None and r['target_xy'] is not None for r in rows),
                  visible_frames_with_correct_candidate=sum(r['candidates'] is not None and r['candidates']['correct_count']>0 for r in rows),
                  selected_review=selected,all_error_frames=errors,rows=rows)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','all_error_frames')},indent=2))


if __name__ == '__main__':
    main()
