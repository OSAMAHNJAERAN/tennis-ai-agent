"""Verify RT-DETR downstream artifacts and render paired detector comparisons."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from scripts.evaluate.audit_player_racket_evidence_gaps import matches
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    folder = base/'rtdetr_racket_selection_pilot01'
    candidate_path = folder/'report.json'
    baseline_path = base/'tracker_original_racket_selection_clipped01/report.json'
    persons_path = base/'rtdetr_player_proposals_pilot01/report.json'
    candidate, baseline, persons = [json.loads(p.read_text(encoding='utf-8')) for p in (candidate_path, baseline_path, persons_path)]
    if not all(r['complete'] for r in (candidate, baseline, persons)) or digest(persons_path) != candidate['person_report_sha256']:
        raise ValueError('Incomplete or changed reports')
    for report in (candidate, baseline, persons):
        for name, expected in report['code_hashes'].items():
            if digest(ROOT/name) != expected:
                raise ValueError('Source code changed')
    if digest(folder/'frozen_protocol.md') != candidate['protocol_sha256']:
        raise ValueError('Protocol changed')
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'manifest.json') != candidate['dataset_manifest_sha256'] or candidate['dataset_manifest_sha256'] != baseline['dataset_manifest_sha256']:
        raise ValueError('Dataset alignment changed')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    result = {'complete': False, 'qualification_evidence': False, 'candidate_report_sha256': digest(candidate_path),
        'baseline_report_sha256': digest(baseline_path), 'person_report_sha256': digest(persons_path),
        'review_script_sha256': digest(Path(__file__)), 'sequences': {}, 'aggregate': {}}
    for sequence, info in candidate['sequences'].items():
        count, fps = info['frames'], info['fps']
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', count)]
        entry = result['sequences'][sequence] = {'variants': {}, 'identity_review': []}
        source_entry = persons['sequences'][sequence]['trackers']['botsort']
        person_path = persons_path.parent/source_entry['predictions_file']
        if digest(person_path) != source_entry['predictions_sha256']:
            raise ValueError('Persons changed')
        source = json.loads(person_path.read_text())
        selected_pair = {}
        for recipe, report, parent in [('YOLO11m', baseline, baseline_path.parent), ('RTDETR', candidate, folder)]:
            tracker = report['sequences'][sequence]['trackers']['botsort']
            if digest(parent/tracker['observations_file']) != tracker['observations_sha256']:
                raise ValueError('Racket cache changed')
            for variant, value in tracker['variants'].items():
                path = parent/value['selected_file']
                if digest(path) != value['selected_sha256']:
                    raise ValueError('Selection changed')
                selected = json.loads(path.read_text())
                if recipe == 'RTDETR':
                    for observed, chosen in zip(source, selected, strict=True):
                        by_id = {r['id']: r for r in observed}
                        for row in chosen:
                            original = by_id[row.get('source_id', row['id'])]
                            if original['box'] != row['box'] or original['confidence'] != row['confidence']:
                                raise ValueError('Fabricated selected observation')
                metrics = value['metrics']['summary']
                key = recipe+'/'+variant
                entry['variants'][key] = metrics
                aggregate = result['aggregate'].setdefault(key, {'TP': 0, 'FP': 0, 'FN': 0, 'IDSW': 0})
                for output, metric in [('TP', 'CLR_TP'), ('FP', 'CLR_FP'), ('FN', 'CLR_FN'), ('IDSW', 'IDSW')]:
                    aggregate[output] += metrics[metric]
                if variant == 'flow_handoff':
                    selected_pair[recipe] = selected
        differences = [len(matches(b, g))-len(matches(a, g)) for a,b,g in zip(selected_pair['YOLO11m'], selected_pair['RTDETR'], gt, strict=True)]
        entry['gained_frame_matches'] = sum(max(0, n) for n in differences)
        entry['lost_frame_matches'] = sum(max(0, -n) for n in differences)
        def draw(frame, index, only_identity=None):
            panels = []
            for recipe in ('YOLO11m', 'RTDETR'):
                panel = frame.copy()
                for row in gt[index]:
                    x1,y1,x2,y2 = map(round, row['box'])
                    cv2.rectangle(panel, (x1,y1), (x2,y2), (225,70,225), 1)
                for row in selected_pair[recipe][index]:
                    if only_identity is not None and recipe == 'RTDETR' and row['id'] != only_identity:
                        continue
                    x1,y1,x2,y2 = map(round, row['box'])
                    cv2.rectangle(panel, (x1,y1), (x2,y2), (40,235,230), 2)
                    cv2.putText(panel, str(row['id']), (x1,max(12,y1-3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (40,235,230), 1)
                panel = cv2.resize(panel, (640,360), interpolation=cv2.INTER_AREA)
                cv2.rectangle(panel, (0,0), (640,25), (15,20,25), -1)
                cv2.putText(panel, f'{sequence} frame {index+1} | {recipe} + BoT-SORT + original racket', (5,17), cv2.FONT_HERSHEY_SIMPLEX, .4, (245,245,245), 1)
                panels.append(panel)
            return cv2.hconcat(panels)
        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            contact = []
            for index in sorted({0, count//2, count-1}):
                contact.append(cv2.resize(draw(frames[index], index), (960,270)))
            path = folder/f'{sequence}_contact.jpg'
            cv2.imwrite(str(path), cv2.vconcat(contact), [cv2.IMWRITE_JPEG_QUALITY, 75])
            entry['contact'] = {'path': path.name, 'sha256': digest(path)}
            identities = sorted({r['id'] for rows in selected_pair['RTDETR'] for r in rows})
            for identity in identities:
                indices = [i for i,rows in enumerate(selected_pair['RTDETR']) if any(r['id'] == identity for r in rows)]
                matched = 0
                for index in indices:
                    chosen = selected_pair['RTDETR'][index]
                    matched += any(chosen[pi]['id'] == identity for pi,gi in matches(chosen, gt[index]))
                samples = [indices[0], indices[len(indices)//2], indices[-1]]
                tiles = [cv2.resize(draw(frames[i], i, identity), (960,270)) for i in samples]
                path = folder/f'{sequence}_identity{identity}.jpg'
                cv2.imwrite(str(path), cv2.vconcat(tiles), [cv2.IMWRITE_JPEG_QUALITY, 75])
                entry['identity_review'].append({'identity': identity, 'selected_frames': len(indices), 'matched_frames': matched,
                    'unmatched_frames': len(indices)-matched, 'sample_frames_1based': [i+1 for i in samples], 'path': path.name, 'sha256': digest(path)})
            if sequence == 'tennis_V02':
                path = folder/'paired_rtdetr_V02.mp4'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280,360))
                if not writer.isOpened():
                    raise RuntimeError('Video writer failed')
                try:
                    for index, frame in enumerate(frames):
                        writer.write(draw(frame,index))
                finally:
                    writer.release()
                cap = cv2.VideoCapture(str(path))
                decoded_fps, decoded = cap.get(cv2.CAP_PROP_FPS), 0
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok:
                            break
                        decoded += 1
                        if decoded == 741:
                            preview = cv2.resize(image, (960,270))
                            cv2.imwrite(str(folder/'decoded_rtdetr_frame741.jpg'), preview, [cv2.IMWRITE_JPEG_QUALITY,75])
                finally:
                    cap.release()
                if decoded != count or abs(decoded_fps-fps) > .001:
                    raise ValueError('Incomplete video decode')
                entry['video'] = {'path':path.name,'sha256':digest(path),'decoded_frames':decoded,'fps':decoded_fps}
    for totals in result['aggregate'].values():
        totals['precision'] = totals['TP']/(totals['TP']+totals['FP']) if totals['TP']+totals['FP'] else 0
        totals['recall'] = totals['TP']/(totals['TP']+totals['FN'])
    result['complete'] = True
    (folder/'final_review.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'aggregate':result['aggregate'],'identities':{s:e['identity_review'] for s,e in result['sequences'].items()}},indent=2))


if __name__ == '__main__':
    main()
