"""Diagnostic review of every racket support for V03 BoT-SORT source 79."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from scripts.evaluate.audit_uvy_videos import digest
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    folder = base/'tracker_racket_selection_clipped01'
    report = json.loads((folder/'report.json').read_text())
    people_path = base/'uvy_tracker_architecture_clipped01/report.json'
    people = json.loads(people_path.read_text())
    if not report['complete'] or digest(people_path) != report['person_report_sha256']:
        raise ValueError('Incomplete or changed inputs')
    cache = people['sequences']['tennis_V03']['trackers']['botsort']
    path = people_path.parent/cache['predictions_file']
    if digest(path) != cache['predictions_sha256']:
        raise ValueError('People changed')
    detections = json.loads(path.read_text())
    candidate = report['sequences']['tennis_V03']['trackers']['botsort']
    path = folder/candidate['variants']['flow_handoff']['selected_file']
    if digest(path) != candidate['variants']['flow_handoff']['selected_sha256']:
        raise ValueError('Selection changed')
    selected = json.loads(path.read_text())
    supports = [r for r in candidate['racket_supports'] if r['source_id'] == 79]
    indices = [i for i, rows in enumerate(selected) if any(r['id'] == 79 for r in rows)]
    cases = [{'frame': i, 'kind': 'first_mid_last_selection'} for i in [indices[0], indices[len(indices)//2], indices[-1]]]
    cases += [{'frame': r['frame'], 'kind': 'all_racket_supports', 'racket': r} for r in supports]
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    with VideoFrameSequence(str(dataset/manifest['sequences']['tennis_V03']['video'])) as frames:
        for number, case in enumerate(cases):
            index = case['frame']
            frame = frames[index].copy()
            person = next(r for r in detections[index] if r['id'] == 79)
            box = tuple(map(round, person['box']))
            cv2.rectangle(frame, box[:2], box[2:], (40, 235, 230), 3)
            if 'racket' in case:
                racket = tuple(map(round, case['racket']['racket_box']))
                cv2.rectangle(frame, racket[:2], racket[2:], (250, 90, 250), 3)
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.rectangle(frame, (0, 0), (640, 26), (15, 20, 25), -1)
            label = f'V03 frame {index+1}: source 79 yellow; racket proposal pink'
            cv2.putText(frame, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1)
            target = folder/f'false_track79_case{number+1}.jpg'
            cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            case.update({'image': target.name, 'sha256': digest(target), 'person': person})
    result = {'complete': True, 'qualification_evidence': False, 'report_sha256': digest(folder/'report.json'),
        'script_sha256': digest(Path(__file__)), 'selected_source79_frames': len(indices), 'cases': cases}
    (folder/'false_track79_review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'selected_frames': len(indices), 'supports': len(supports), 'images': len(cases)}))


if __name__ == '__main__':
    main()
