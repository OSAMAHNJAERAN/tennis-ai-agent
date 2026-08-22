import json

def main():
    with open('outputs/phase2_yolo11_final/trajectories.json') as f:
        traj = json.load(f)['ball_trajectory']
    with open('outputs/phase2_yolo11_final/detections.json') as f:
        det = json.load(f)['frames']

    print(f"{'Frame':<6} | {'Time (s)':<8} | {'Ball (x, y)':<18} | {'P1 [x1,y1,x2,y2]':<24} | {'P2 [x1,y1,x2,y2]':<24}")
    print("-" * 90)
    for f in range(0, len(traj), 10):
        p = traj[f]
        b_str = f"({p['x_px']:.1f}, {p['y_px']:.1f})" if p['x_px'] is not None else "MISSING"
        item = det[f]
        p1 = item.get('player_1', {}).get('bbox', [0,0,0,0])
        p2 = item.get('player_2', {}).get('bbox', [0,0,0,0])
        p1_s = f"[{p1[0]:.0f},{p1[1]:.0f},{p1[2]:.0f},{p1[3]:.0f}]"
        p2_s = f"[{p2[0]:.0f},{p2[1]:.0f},{p2[2]:.0f},{p2[3]:.0f}]"
        print(f"{f:6d} | {p['timestamp_seconds']:8.3f} | {b_str:<18s} | {p1_s:<24s} | {p2_s:<24s}")

if __name__ == '__main__':
    main()
