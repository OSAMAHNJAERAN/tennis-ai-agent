"""Render all racket supports behind the two RT-DETR false selected tracks."""
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
    folder = base/'rtdetr_racket_selection_pilot01'
    report_path = folder/'report.json'
    people_path = base/'rtdetr_player_proposals_pilot01/report.json'
    report, people = [json.loads(p.read_text()) for p in (report_path, people_path)]
    if not report['complete'] or digest(people_path) != report['person_report_sha256']:
        raise ValueError('Incomplete or changed inputs')
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
        raise ValueError('Dataset changed')
    result = {'complete': True, 'report_sha256': digest(report_path), 'script_sha256': digest(Path(__file__)), 'cases': []}
    for sequence, identity in [('tennis_V01',1449), ('tennis_V03',227)]:
        tracker = report['sequences'][sequence]['trackers']['botsort']
        cache = people['sequences'][sequence]['trackers']['botsort']
        path = people_path.parent/cache['predictions_file']
        if digest(path) != cache['predictions_sha256']:
            raise ValueError('Person cache changed')
        observations = json.loads(path.read_text())
        evidence = [r for r in tracker['racket_supports'] if r['source_id'] == identity]
        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            for support in evidence:
                index = support['frame']
                frame = frames[index].copy()
                person = next(r for r in observations[index] if r['id'] == identity)
                x1,y1,x2,y2 = person['box']
                height = y2-y1
                left = max(0, int(min(x1, support['racket_box'][0])-height))
                top = max(0, int(min(y1, support['racket_box'][1])-height*.5))
                right = min(frame.shape[1], int(max(x2, support['racket_box'][2])+height))
                bottom = min(frame.shape[0], int(max(y2, support['racket_box'][3])+height*.5))
                for box,color in [(person['box'],(40,235,230)),(support['racket_box'],(230,70,230))]:
                    q = tuple(map(round,box))
                    cv2.rectangle(frame,q[:2],q[2:],color,1)
                crop = frame[top:bottom,left:right]
                scale = min(640/crop.shape[1],400/crop.shape[0])
                crop = cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)),interpolation=cv2.INTER_NEAREST)
                crop = cv2.copyMakeBorder(crop,30,0,0,0,cv2.BORDER_CONSTANT,value=(15,20,25))
                cv2.putText(crop,f'{sequence} frame {index+1}, source {identity}; yellow person / pink racket',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.38,(245,245,245),1)
                path = folder/f'{sequence}_false_support_{index+1}.jpg'
                cv2.imwrite(str(path),crop,[cv2.IMWRITE_JPEG_QUALITY,80])
                result['cases'].append({'sequence':sequence,'identity':identity,'frame_1based':index+1,'support':support,
                    'person':person,'context_crop_xyxy':[left,top,right,bottom],'path':path.name,'sha256':digest(path)})
    (folder/'false_support_review.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'cases':len(result['cases'])}))


if __name__ == '__main__':
    main()
