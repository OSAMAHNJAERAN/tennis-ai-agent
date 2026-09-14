"""Review the original-racket paired tracker control."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from PIL import Image, ImageDraw
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from scripts.evaluate.audit_player_racket_evidence_gaps import matches
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    folder = base/'tracker_original_racket_selection_clipped01'
    report_path = folder/'report.json'
    people_path = base/'uvy_tracker_architecture_clipped01/report.json'
    old_path = base/'uvy_duplicate_handoff_pilot01/report.json'
    report, people, old = [json.loads(p.read_text()) for p in (report_path, people_path, old_path)]
    if not all(r['complete'] for r in (report, people, old)) or digest(people_path) != report['person_report_sha256']:
        raise ValueError('Incomplete or changed reports')
    for name, expected in report['code_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Changed code: {name}')
    if digest(folder/'frozen_protocol.md') != report['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    result = {'complete': False, 'qualification_evidence': False, 'report_sha256': digest(report_path),
        'person_report_sha256': digest(people_path), 'previous_original_report_sha256': digest(old_path),
        'review_script_sha256': digest(Path(__file__)), 'sequences': {}, 'aggregate': {}, 'videos': {}}
    panels = []
    for sequence, info in report['sequences'].items():
        count, fps = info['frames'], info['fps']
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', count)]
        entry = result['sequences'][sequence] = {'trackers': {}}
        selected_by_kind = {}
        for kind, value in info['trackers'].items():
            cache = people['sequences'][sequence]['trackers'][kind]
            path = people_path.parent/cache['predictions_file']
            if digest(path) != value['person_predictions_sha256'] or digest(path) != cache['predictions_sha256']:
                raise ValueError('Persons changed')
            persons = json.loads(path.read_text())
            observation_path = folder/value['observations_file']
            if digest(observation_path) != value['observations_sha256']:
                raise ValueError('Rackets changed')
            spans, current, longest = {}, {}, {}
            for frame_index, (rows, truth) in enumerate(zip(persons, gt, strict=True)):
                keys = {(truth[gi]['id'], rows[pi]['id']) for pi, gi in matches(rows, truth)}
                current = {key: current.get(key, 0)+1 for key in keys}
                for key, n in current.items():
                    longest[key] = max(longest.get(key, 0), n)
            for identity in {r['id'] for rows in gt for r in rows}:
                options = [(n, source) for (gid, source), n in longest.items() if gid == identity]
                n, source = max(options, default=(0, None))
                spans[str(identity)] = {'longest_consecutive_matched_frames': n, 'source_id': source, 'duration_seconds': n/fps}
            entry['trackers'][kind] = {'matched_spans': spans, 'variants': {}}
            for variant, candidate in value['variants'].items():
                path = folder/candidate['selected_file']
                if digest(path) != candidate['selected_sha256']:
                    raise ValueError('Selected output changed')
                selected = json.loads(path.read_text())
                for rows, chosen in zip(persons, selected, strict=True):
                    by_id = {r['id']: r for r in rows}
                    for row in chosen:
                        original = by_id[row.get('source_id', row['id'])]
                        if row['box'] != original['box'] or row['confidence'] != original['confidence']:
                            raise ValueError('Source observation changed')
                metrics = candidate['metrics']['summary']
                entry['trackers'][kind]['variants'][variant] = {'metrics': metrics, 'observations_verified': True}
                key = kind+'/'+variant
                total = result['aggregate'].setdefault(key, {'TP': 0, 'FP': 0, 'FN': 0, 'IDSW': 0})
                for out, field in [('TP', 'CLR_TP'), ('FP', 'CLR_FP'), ('FN', 'CLR_FN'), ('IDSW', 'IDSW')]:
                    total[out] += metrics[field]
                if variant == 'flow_handoff':
                    selected_by_kind[kind] = selected
                    if kind == 'bytetrack':
                        previous_path = old_path.parent/old['sequences'][sequence]['selected_file']
                        if digest(previous_path) != old['sequences'][sequence]['selected_sha256']:
                            raise ValueError('Old trained selection changed')
                        previous = json.loads(previous_path.read_text())
                        entry['previous_original_byte_exact_frames'] = sum(a == b for a, b in zip(selected, previous, strict=True))

        def paired(frame, index):
            tiles = []
            for kind in ('bytetrack', 'botsort'):
                tile = frame.copy()
                for row in gt[index]:
                    x1, y1, x2, y2 = map(round, row['box'])
                    cv2.rectangle(tile, (x1, y1), (x2, y2), (220, 70, 220), 1)
                for row in selected_by_kind[kind][index]:
                    x1, y1, x2, y2 = map(round, row['box'])
                    cv2.rectangle(tile, (x1, y1), (x2, y2), (40, 235, 230), 2)
                    cv2.putText(tile, str(row['id']), (x1, max(12, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (40, 235, 230), 1)
                tile = cv2.resize(tile, (640, 360), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tile, (0, 0), (640, 24), (15, 20, 25), -1)
                cv2.putText(tile, f'{sequence}  frame {index+1}  {kind} + flow/handoff', (8, 17), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
                tiles.append(tile)
            return cv2.hconcat(tiles)

        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            for index in sorted({0, count//2, count-1}):
                panel = paired(frames[index], index)
                panels.append(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
            for index in ({'tennis_V01': [662], 'tennis_V02': [], 'tennis_V03': [381]}[sequence]):
                extra = paired(frames[index], index)
                extra = cv2.resize(extra, (960, 270), interpolation=cv2.INTER_AREA)
                target = folder/f'gain_review_{sequence}_frame{index+1}.jpg'
                cv2.imwrite(str(target), extra, [cv2.IMWRITE_JPEG_QUALITY, 75])
                entry.setdefault('gain_review_images', []).append({'path': target.name, 'sha256': digest(target), 'frame': index+1})
            if sequence == 'tennis_V03':
                video_path = folder/'paired_tracker_V03.mp4'
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 360))
                if not writer.isOpened():
                    raise RuntimeError('Video writer failed')
                try:
                    for index, frame in enumerate(frames):
                        writer.write(paired(frame, index))
                finally:
                    writer.release()
                decoded = 0
                cap = cv2.VideoCapture(str(video_path))
                decoded_fps = cap.get(cv2.CAP_PROP_FPS)
                try:
                    while True:
                        ok, image = cap.read()
                        if not ok:
                            break
                        decoded += 1
                        if decoded == 421:
                            cv2.imwrite(str(folder/'decoded_tracker_frame421.jpg'), image)
                finally:
                    cap.release()
                if decoded != count or abs(decoded_fps-fps) > .001:
                    raise ValueError('Incomplete video')
                result['videos'][sequence] = {'path': video_path.name, 'sha256': digest(video_path), 'decoded_frames': decoded, 'fps': decoded_fps}
    for total in result['aggregate'].values():
        total['precision'] = total['TP']/(total['TP']+total['FP'])
        total['recall'] = total['TP']/(total['TP']+total['FN'])
    board = Image.new('RGB', (1920, 610), '#141b24')
    draw = ImageDraw.Draw(board)
    draw.text((12, 8), 'Paired tracker review: yellow = selected candidate, pink = original publisher GT. Each tile: ByteTrack left / BoT-SORT right.', fill='white')
    for index, panel in enumerate(panels):
        board.paste(panel.resize((640, 180)), ((index%3)*640, 35+(index//3)*190))
    board_path = folder/'paired_tracker_contact.jpg'
    board.save(board_path, quality=92)
    result['contact_sheet'] = {'path': board_path.name, 'sha256': digest(board_path)}
    result['complete'] = True
    (folder/'final_review.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'aggregate': result['aggregate'], 'videos': result['videos'], 'previous_byte_exact': {s: e['previous_original_byte_exact_frames'] for s,e in result['sequences'].items()}}, indent=2))


if __name__ == '__main__':
    main()
