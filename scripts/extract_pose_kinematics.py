import cv2
import json
import numpy as np
from ultralytics import YOLO
from src.utils.bbox_utils import BBox
from src.detection.player_detector import PlayerDetector

model = YOLO("yolo11n-pose.pt")
player_det = PlayerDetector(model_path="yolo11m.pt")

with open('data/benchmarks/cross_match_final_holdout/videos.json') as f:
    vids = json.load(f)['videos']
with open('data/benchmarks/cross_match_final_holdout/ground_truth_shots.json') as f:
    gt_shots = json.load(f)['shots']

print("=== POSE KEYPOINT KINEMATICS EXTRACTION ===")

for vid in ['video_08', 'video_09', 'video_10']:
    vpath = vids[vid]['path']
    cap = cv2.VideoCapture(vpath)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    print(f"\n--- {vid} ({len(frames)} frames) ---")
    
    # Run player detection
    player_boxes_p1 = []
    player_boxes_p2 = []
    for f in frames:
        boxes = player_det.detect(f)
        if len(boxes) >= 2:
            s_boxes = sorted(boxes, key=lambda b: b.y2)
            player_boxes_p2.append(s_boxes[0]) # far
            player_boxes_p1.append(s_boxes[-1]) # near
        elif len(boxes) == 1:
            player_boxes_p1.append(boxes[0])
            player_boxes_p2.append(None)
        else:
            player_boxes_p1.append(None)
            player_boxes_p2.append(None)
        
    for g in gt_shots[vid]:
        f_idx = g['frame_index']
        p_id = g['player_id']
        gt_type = g['shot_type']
        side = g['court_side']
        
        boxes = player_boxes_p1 if p_id == 1 else player_boxes_p2
        bbox = boxes[f_idx]
        if bbox is None:
            print(f"Hit f{f_idx} [{gt_type:8s}] P{p_id} {side:4s} -> NO PLAYER BOX")
            continue
            
        frame = frames[f_idx]
        fh, fw = frame.shape[:2]
        bw = bbox.x2 - bbox.x1
        bh = bbox.y2 - bbox.y1
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        
        crop_margin = 0.35
        x1_c = max(0, int(cx - (bw * (1.0 + crop_margin)) / 2.0))
        x2_c = min(fw, int(cx + (bw * (1.0 + crop_margin)) / 2.0))
        y1_c = max(0, int(cy - (bh * (1.0 + crop_margin)) / 2.0))
        y2_c = min(fh, int(cy + (bh * (1.0 + crop_margin)) / 2.0))
        
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        res = model.predict(crop, verbose=False)
        if not res or not hasattr(res[0], 'keypoints') or res[0].keypoints is None or len(res[0].keypoints.data) == 0:
            print(f"Hit f{f_idx} [{gt_type:8s}] P{p_id} {side:4s} -> NO POSE DETECTED")
            continue
            
        kpts = res[0].keypoints.data[0].cpu().numpy()
        l_sh, r_sh = kpts[5], kpts[6]
        l_wr, r_wr = kpts[9], kpts[10]
        
        sh_mid_x = (l_sh[0] + r_sh[0]) / 2.0
        sh_w = max(5.0, np.linalg.norm(r_sh[:2] - l_sh[:2]))
        
        # Relative wrist displacement
        r_wr_dx = (r_wr[0] - sh_mid_x) / sh_w
        l_wr_dx = (l_wr[0] - sh_mid_x) / sh_w
        
        print(f"Hit f{f_idx:4d} [{gt_type:8s}] P{p_id} {side:4s} | Box: {int(bw)}x{int(bh)} | "
              f"L_Sh: conf {l_sh[2]:.2f}, R_Sh: conf {r_sh[2]:.2f} | "
              f"L_Wr: conf {l_wr[2]:.2f} (dx={l_wr_dx:+.2f}), R_Wr: conf {r_wr[2]:.2f} (dx={r_wr_dx:+.2f}) | "
              f"Shoulder X: L={l_sh[0]:.1f}, R={r_sh[0]:.1f}")
