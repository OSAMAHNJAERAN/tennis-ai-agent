import json
import math
import numpy as np

def main():
    with open('data/benchmarks/ball_baseline/ground_truth.json') as f:
        gt = json.load(f)
    gt_frames = gt['frames']

    with open('outputs/phase2_yolo11_final/trajectories.json') as f:
        traj_data = json.load(f)
    ball_traj = traj_data['ball_trajectory']

    errors_with_info = []

    for i in range(len(ball_traj)):
        p = ball_traj[i]
        g = gt_frames[i]
        
        if g['candidate_bbox'] and p['x_px'] is not None:
            gt_cx = (g['candidate_bbox'][0] + g['candidate_bbox'][2]) / 2.0
            gt_cy = (g['candidate_bbox'][1] + g['candidate_bbox'][3]) / 2.0
            err = math.hypot(p['x_px'] - gt_cx, p['y_px'] - gt_cy)
            errors_with_info.append({
                'frame': i,
                'timestamp_s': p['timestamp_seconds'],
                'error_px': err,
                'category': g['category'],
                'state': p['state'],
                'pred_x': p['x_px'],
                'pred_y': p['y_px'],
                'gt_x': gt_cx,
                'gt_y': gt_cy
            })

    errors = [e['error_px'] for e in errors_with_info]

    print("=== PHASE 2.1 TRAJECTORY LOCALIZATION ERROR ANALYSIS ===")
    print(f"Total evaluated points: {len(errors)}")
    print(f"Mean Error:    {np.mean(errors):.3f} px")
    print(f"Median Error:  {np.median(errors):.3f} px")
    print(f"P90 Error:     {np.percentile(errors, 90):.3f} px")
    print(f"P95 Error:     {np.percentile(errors, 95):.3f} px")
    print(f"P99 Error:     {np.percentile(errors, 99):.3f} px")
    print(f"Max Error:     {np.max(errors):.3f} px")

    # Sort to find worst 10 outlier frames
    errors_with_info.sort(key=lambda x: x['error_px'], reverse=True)
    print("\nTop 10 Outlier Frames:")
    print(f"{'Frame':<6} | {'Time (s)':<8} | {'Error (px)':<10} | {'Category':<22} | {'State':<12} | {'Pred (x,y)':<18} | {'GT (x,y)':<18}")
    print("-" * 105)
    for item in errors_with_info[:10]:
        px_str = f"({item['pred_x']:.1f}, {item['pred_y']:.1f})"
        gt_str = f"({item['gt_x']:.1f}, {item['gt_y']:.1f})"
        print(f"{item['frame']:6d} | {item['timestamp_s']:8.3f} | {item['error_px']:10.2f} | {item['category']:22s} | {item['state']:12s} | {px_str:<18s} | {gt_str:<18s}")

if __name__ == '__main__':
    main()
