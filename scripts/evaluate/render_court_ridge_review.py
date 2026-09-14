"""Render all changed cases and all lost matches for one completed ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['no_white', 'no_sides', 'ridge_only'], required=True)
    args = parser.parse_args()
    output = ROOT/f'outputs/vision_upgrade_audit/court_ridge_ablation_{args.mode}01'
    report = json.loads((output/'report.json').read_text())
    assert report['complete']
    changes = [row for row in report['cases'] if row.get('changed_from_previous')]
    boards = []
    for start in range(0, len(changes), 4):
        group = changes[start:start+4]
        board = Image.new('RGB', (1280, 1568), (20, 20, 20))
        for index, row in enumerate(group):
            image = Image.open(output/row['image'])
            # These saved changes all have 960x540 reference images.
            assert image.size == (640, 1176)
            board.paste(image.crop((0, 392, 640, 1176)), ((index % 2)*640, (index//2)*784))
        path = output/f'changed_review_{start//4+1:02}.jpg'
        board.resize((960, 1176)).save(path, quality=67)
        boards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           indices=[row['index'] for row in group], visually_inspected=False))
    losses = []
    for row in report['cases']:
        if 'scores' not in row:
            continue
        before = row['scores']['original']['landmarks']
        after = row['scores']['ablation']['landmarks']
        for identity, (a, b) in enumerate(zip(before, after)):
            if a['correct'] and not b['correct']:
                losses.append((row, identity, a, b))
    panels, loss_details = [], []
    for row, identity, before, after in losses:
        source = cv2.imread(str(ROOT/'data/external/court_heatmap_pilot/images'/f"{row['id']}.png"))
        pair = []
        for stage, point in [('previous', before), (args.mode, after)]:
            label = np.array(point['label'])
            prediction = np.array(point['prediction'])
            left = max(0, min(1280-192, int(label[0])-96))
            top = max(0, min(720-160, int(label[1])-80))
            crop = source[top:top+160, left:left+192].copy()
            cv2.circle(crop, tuple(np.rint(label-[left, top]).astype(int)), 5, (30,255,30), 1)
            cv2.circle(crop, tuple(np.rint(prediction-[left, top]).astype(int)), 3, (30,30,255), 1)
            crop = cv2.copyMakeBorder(crop, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,20,20))
            for text, y in [(f"{row['id']} ID{identity}", 13), (f"{stage} {point['distance_native_px']:.3f}px", 29)]:
                cv2.putText(crop, text, (3,y), cv2.FONT_HERSHEY_SIMPLEX, .31, (255,255,255), 1)
            pair.append(crop)
        panels.append(np.concatenate(pair, axis=1))
        loss_details.append(dict(index=row['index'], id=row['id'], identity=identity,
                                 before_px=before['distance_native_px'], after_px=after['distance_native_px']))
    loss_boards = []
    for start in range(0, len(panels), 4):
        path = output/f'lost_matches_{start//4+1:02}.jpg'
        assert cv2.imwrite(str(path), np.concatenate(panels[start:start+4]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        loss_boards.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                landmarks=loss_details[start:start+4], visually_inspected=False))
    result = dict(report_sha256=hashlib.sha256((output/'report.json').read_bytes()).hexdigest(),
                  mode=args.mode, changed_boards=boards, loss_boards=loss_boards)
    (output/'visual_review.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
