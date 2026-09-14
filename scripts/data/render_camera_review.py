"""Render a labeled contact sheet for acquired training/validation source review."""

import json
from pathlib import Path

import cv2
import numpy as np


def main():
    roots = [Path('data/external/racketvision_validation'), Path('data/external/racketvision_training20')]
    clips = [(root, match, rally) for root in roots
             for match, rally in json.loads((root / 'manifest.json').read_text())['selected_clips']]
    canvas = np.full((((len(clips) + 3) // 4) * 210, 4 * 360, 3), 245, np.uint8)
    for index, (root, match, rally) in enumerate(clips):
        cap = cv2.VideoCapture(str(root / f'tennis/videos/{match}_{rally}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f'Cannot decode {match}')
        y, x = index // 4 * 210, index % 4 * 360
        canvas[y:y + 180, x:x + 360] = cv2.resize(frame, (360, 180))
        label = ('VAL ' if root == roots[0] else 'TRAIN ') + match
        cv2.putText(canvas, label, (x + 8, y + 201), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    if not cv2.imwrite('outputs/vision_upgrade_audit/data_camera_contact_sheet.jpg', canvas):
        raise RuntimeError('Contact sheet write failed')


if __name__ == '__main__':
    main()
