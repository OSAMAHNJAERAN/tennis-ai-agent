"""Inspect newly selected identities and every V02 racket support."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import box_iou_matrix
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    output = base/'uvy_trained_racket_players_pilot01'
    report = json.loads((output/'report.json').read_text())
    baseline_path = base/'uvy_duplicate_handoff_pilot01/report.json'
    baseline = json.loads(baseline_path.read_text())
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if not report['complete'] or digest(baseline_path) != report['baseline_report_sha256']:
        raise ValueError('Incomplete or changed baseline')
    summary = {'scope': 'NEW_IDENTITY_ERROR_REVIEW; SOURCE_LABELS_UNCHANGED', 'qualification_evidence': False,
               'report_sha256': digest(output/'report.json'), 'script_sha256': digest(__file__), 'sequences': {}}
    cases = []
    for sequence, info in report['sequences'].items():
        path = output/info['selected_file']
        if digest(path) != info['selected_sha256']:
            raise ValueError('Selected observations changed')
        rows = json.loads(path.read_text())
        raw_path = base/'uvy_person_tracked640'/f'{sequence}_proposals.json'
        if digest(raw_path) != info['person_predictions_sha256']:
            raise ValueError('Person source changed')
        raw = json.loads(raw_path.read_text())
        for selected, observed in zip(rows, raw, strict=True):
            by_id = {r['id']: r for r in observed}
            for row in selected:
                original = by_id[row['source_id']]
                if row['box'] != original['box'] or row['confidence'] != original['confidence']:
                    raise ValueError('Selected observation was synthesized or changed')
        new_ids = [int(i) for i, stats in info['track_statistics'].items()
                   if stats['eligible'] and not baseline['sequences'][sequence]['track_statistics'].get(i, {}).get('eligible', False)]
        gt = [[r for r in frame if r['class'] == 1] for frame in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(rows))]
        details = []
        for identity in new_ids:
            selected_frames = [i for i, frame in enumerate(rows) if any(r['id'] == identity for r in frame)]
            ious = []
            for index in selected_frames:
                player = next(r for r in rows[index] if r['id'] == identity)
                iou = box_iou_matrix([player['box']], [r['box'] for r in gt[index]])
                ious.append(float(iou.max()) if iou.size else 0)
            samples = [selected_frames[i] for i in sorted({0, len(selected_frames)//2, len(selected_frames)-1})] if selected_frames else []
            details.append({'canonical_id': identity, 'selected_frames': len(selected_frames), 'best_gt_iou50_frames': sum(i >= .5 for i in ious),
                            'median_best_gt_iou': float(np.median(ious)) if ious else None, 'review_frames': samples,
                            'metric_scope': 'Per-box best GT IoU diagnostic, not one-to-one tracking metrics'})
            cases.extend((sequence, index, identity, None) for index in samples)
        summary['sequences'][sequence] = {'newly_eligible': details, 'all_selected_boxes_are_exact_observations': True}
    # Every V02 support includes one observation on a source too short-supported to be selected.
    for support in report['sequences']['tennis_V02']['racket_supports']:
        cases.append(('tennis_V02', support['frame'], support['source_id'], support))
    board = Image.new('RGB', (960, 240*((len(cases)+2)//3)), '#222222')
    draw = ImageDraw.Draw(board)
    for number, (sequence, index, identity, support) in enumerate(cases):
        info = report['sequences'][sequence]
        selected = json.loads((output/info['selected_file']).read_text())
        raw = json.loads((base/'uvy_person_tracked640'/f'{sequence}_proposals.json').read_text())
        persons = selected[index] if support is None else raw[index]
        player = next(r for r in persons if r['id'] == identity)
        truth = [r for r in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(raw))[index] if r['class'] == 1]
        ious = box_iou_matrix([player['box']], [r['box'] for r in truth])
        best = truth[int(ious.argmax())] if ious.size and ious.max() > 0 else None
        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            image = Image.fromarray(frames[index][:, :, ::-1])
        painter = ImageDraw.Draw(image)
        if best:
            painter.rectangle(best['box'], outline='#ff40d0', width=1)
        painter.rectangle(player['box'], outline='#00ffff', width=1)
        if support:
            painter.rectangle(support['racket_box'], outline='#ffff00', width=1)
        x1, y1, x2, y2 = player['box']
        radius = max(50, y2-y1)
        x, y = (x1+x2)/2, (y1+y2)/2
        crop = (max(0, int(x-radius)), max(0, int(y-radius)), min(image.width, int(x+radius)), min(image.height, int(y+radius)))
        image = image.crop(crop)
        image.thumbnail((300, 200))
        left, top = number%3*320, number//3*240
        board.paste(image, (left, top+35))
        draw.text((left+4, top+4), f'{sequence} frame{index+1} ID{identity}', fill='white')
        draw.text((left+4, top+18), f'IoU {float(ious.max()) if ious.size else 0:.3f}; cyan pred / pink GT', fill='white')
    image_path = output/'new_identity_and_support_review.jpg'
    board.save(image_path, quality=75)
    summary['review_image_sha256'] = digest(image_path)
    summary['review_cases'] = cases
    (output/'new_identity_review.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary['sequences']))


if __name__ == '__main__':
    main()
