"""Record paired player counts and visualize all COCO true-positive losses."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.detection_metrics import match_detections


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    first = base/'uvy_racket_supported_players_pilot01'
    second = base/'uvy_racket_supported_players_pilot02_nearest'
    output = second/'paired_review.json'
    if output.exists():
        raise FileExistsError(output)
    old = json.loads((first/'report.json').read_text())
    new = json.loads((second/'report.json').read_text())
    legacy_path = base/'uvy_player_tracking_baseline/report.json'
    legacy = json.loads(legacy_path.read_text())
    controls_path = base/'racket_nearest_ownership_controls_final.json'
    controls = json.loads(controls_path.read_text())
    if not all(r['complete'] for r in (old, new, legacy, controls)):
        raise ValueError('Incomplete comparison input')
    report = {'qualification_evidence': False, 'sequences': {},
              'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
                               (first/'report.json', second/'report.json', legacy_path, controls_path)},
              'script_sha256': digest(__file__)}
    for sequence, candidate in new['sequences'].items():
        before = json.loads((first/old['sequences'][sequence]['selected_file']).read_text())
        after = json.loads((second/candidate['selected_file']).read_text())
        for folder, info in ((first, old['sequences'][sequence]), (second, candidate)):
            if digest(folder/info['selected_file']) != info['selected_sha256']:
                raise ValueError('Selected source changed')
        gt = [[r for r in frame if r['class'] == 1] for frame in read_gt(
            ROOT/'data/external/uvy_tennis_videos/UVY'/sequence/'gt/gt.txt', candidate['frames'])]
        changes = []
        for index, (a, b, truth) in enumerate(zip(before, after, gt, strict=True)):
            ac = match_detections([r['box'] for r in a], [r['box'] for r in truth], iou_threshold=.5)
            bc = match_detections([r['box'] for r in b], [r['box'] for r in truth], iou_threshold=.5)
            if a != b:
                changes.append({'frame': index, 'before_ids': [r['id'] for r in a], 'after_ids': [r['id'] for r in b],
                                'tp_delta': bc.true_positives-ac.true_positives,
                                'fp_delta': bc.false_positives-ac.false_positives})
        report['sequences'][sequence] = {
            'legacy': legacy['sequences'][sequence]['metrics']['legacy_roles']['summary'],
            'court': legacy['sequences'][sequence]['metrics']['upgraded_roles']['summary'],
            'racket_original': old['sequences'][sequence]['metrics']['summary'],
            'racket_nearest': candidate['metrics']['summary'],
            'all_frame_tp_counts_unchanged': all(r['tp_delta'] == 0 for r in changes), 'changes': changes}
    coco_source_path = base/'coco_racket_global_assignment640.json'
    coco_source = json.loads(coco_source_path.read_text())
    if digest(coco_source_path) != controls['datasets']['coco']['source_sha256']:
        raise ValueError('COCO inference source changed')
    previous = {r['image_id']: r for r in coco_source['per_image_iou50']}
    losses = [r['image_id'] for r in controls['datasets']['coco']['per_image_iou50']
              if r['tp'] < previous[r['image_id']]['tp']]
    board = Image.new('RGB', (960, 360*len(losses)), '#222222')
    draw = ImageDraw.Draw(board)
    metadata = {r['id']: r for r in json.loads((ROOT/'data/external/coco_tennis_val2017/manifest.json').read_text())['images']}
    for row, identity in enumerate(losses):
        item = metadata[identity]
        path = ROOT/'data/external/coco_tennis_val2017'/item['path']
        if digest(path) != item['sha256']:
            raise ValueError('Review image changed')
        before = next(r for r in coco_source['images'] if r['image_id'] == identity)
        after = next(r for r in controls['datasets']['coco']['per_image'] if r['image'] == identity)
        for column, (label, observations) in enumerate((('original', before['observations']), ('nearest', after['observations']))):
            image = Image.open(path).convert('RGB')
            painter = ImageDraw.Draw(image)
            for person_id, box in before['player_boxes'].items():
                painter.rectangle(box, outline='#00d0ff', width=1)
                painter.text((box[0], max(0, box[1]-12)), person_id, fill='#00d0ff')
            for person_id, value in observations.items():
                if value['bbox_xyxy'] is not None:
                    box = value['bbox_xyxy']
                    painter.rectangle(box, outline='#ffe000', width=3)
                    painter.text((box[0], max(0, box[1]-12)), f'R{person_id}', fill='#ffe000')
            image.thumbnail((480, 330))
            board.paste(image, (column*480, row*360+30))
            draw.text((column*480+5, row*360+7), f'{identity} {label} | cyan people / yellow rackets', fill='white')
    board_path = second/'coco_loss_review.jpg'
    board.save(board_path, quality=70)
    report.update(coco_lost_tp_images=losses, loss_review_sha256=digest(board_path))
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'all_sequences_frame_tp_unchanged': all(v['all_frame_tp_counts_unchanged'] for v in report['sequences'].values()),
                      'coco_lost_tp_images': losses}))


if __name__ == '__main__':
    main()
