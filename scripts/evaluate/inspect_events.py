import json
from src.tracking.temporal_ball_tracker import TemporalBallPoint, BallState
from src.utils.bbox_utils import BBox
from src.events.event_detector import TennisEventDetector

def main():
    with open('outputs/phase2_yolo11_final/trajectories.json') as f:
        traj = json.load(f)['ball_trajectory']
    ball_points = [
        TemporalBallPoint(
            frame_index=p['frame_index'],
            timestamp_seconds=p['timestamp_seconds'],
            x_px=p['x_px'],
            y_px=p['y_px'],
            court_x_m=p['court_x_m'],
            court_y_m=p['court_y_m'],
            speed_kmh=p['speed_kmh'],
            confidence=p['confidence'],
            state=BallState(p['state'])
        ) for p in traj
    ]

    with open('outputs/phase2_yolo11_final/detections.json') as f:
        det = json.load(f)['frames']

    p1_boxes = [None]*len(traj)
    p2_boxes = [None]*len(traj)
    for item in det:
        f_idx = item['frame_index']
        if item.get('player_1') and item['player_1'].get('bbox'):
            bb = item['player_1']['bbox']
            p1_boxes[f_idx] = BBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])
        if item.get('player_2') and item['player_2'].get('bbox'):
            bb = item['player_2']['bbox']
            p2_boxes[f_idx] = BBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3])

    with open('data/benchmarks/tennis_events/ground_truth.json') as f:
        gt_events = json.load(f)['events']

    det_inst = TennisEventDetector()
    events = det_inst.detect_events(ball_points, p1_boxes, p2_boxes)

    print("=== GROUND TRUTH EVENTS ===")
    for g in gt_events:
        print(f"GT {g['event_id']}: Frame {g['frame']:3d} ({g['timestamp_s']:6.3f}s) | {g['event_type']:14s} (Player {g['player_id']}) | Pos: ({g['x_px']}, {g['y_px']})")

    print("\n=== DETECTED EVENTS ===")
    for e in events:
        print(f"EV {e.event_id}: Frame {e.frame_index:3d} ({e.timestamp_s:6.3f}s) | {e.event_type.value:14s} (Player {e.player_id}) | Pos: ({e.ball_position_px[0]:.1f}, {e.ball_position_px[1]:.1f}) | Conf: {e.confidence:.2f}")

if __name__ == '__main__':
    main()
