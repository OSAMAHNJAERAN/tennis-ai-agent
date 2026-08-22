import json
import math
from src.events.trajectory_derivatives import TrajectoryDerivativeCalculator

def main():
    with open('outputs/phase2_yolo11_final/trajectories.json') as f:
        traj = json.load(f)['ball_trajectory']

    pos = [(p['x_px'], p['y_px']) if p['x_px'] is not None and p['y_px'] is not None else None for p in traj]
    ts = [p['timestamp_seconds'] for p in traj]

    derivs = TrajectoryDerivativeCalculator.compute_derivatives(pos, ts)

    gt_frames = [23, 62, 84, 138, 157, 196]

    print(f"{'Frame':<6} | {'Time (s)':<8} | {'Speed (px/s)':<12} | {'Accel (px/s2)':<14} | {'Curvature':<12} | {'DirChange (deg)':<16} | {'GT'}")
    print("-" * 85)
    for f in range(len(traj)):
        d = derivs[f]
        dir_deg = math.degrees(d.direction_change_rad) if d.direction_change_rad is not None else 0.0
        acc = d.accel_mag_px_s2 or 0.0
        curv = d.curvature or 0.0
        spd = d.speed_px_s or 0.0
        is_gt = ' <=== GT EVENT' if f in gt_frames else ''
        if f in gt_frames or dir_deg > 15.0 or acc > 400.0 or curv > 0.001:
            print(f"{f:6d} | {d.timestamp_s:8.3f} | {spd:12.1f} | {acc:14.1f} | {curv:12.5f} | {dir_deg:16.1f} | {is_gt}")

if __name__ == '__main__':
    main()
