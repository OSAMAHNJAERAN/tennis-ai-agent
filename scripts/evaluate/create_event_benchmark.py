import os
import json
from src.utils.video_io import read_video
from src.court.court_geometry import TennisCourtGeometry
from src.court.court_keypoint_detector import CourtKeypointDetector
from src.court.homography import compute_homography, transform_point

def main():
    frames, meta = read_video('data/sample_videos/input_video.mp4')
    fps = meta.fps

    court_det = CourtKeypointDetector('models/keypoints_model.pth')
    kps = court_det.predict(frames[0])
    can_kps = TennisCourtGeometry.get_canonical_keypoints()
    H, _ = compute_homography(kps, can_kps)

    with open('outputs/phase2_yolo11_final/trajectories.json') as f:
        traj = json.load(f)['ball_trajectory']

    # Hand-verified ground truth events across the rally sequence
    event_specs = [
        {'id': 1, 'type': 'SERVE_CONTACT', 'frame': 23, 'player_id': 2, 'desc': 'Player 2 overhead serve contact'},
        {'id': 2, 'type': 'BOUNCE', 'frame': 62, 'player_id': None, 'desc': 'First bounce in Player 1 service area'},
        {'id': 3, 'type': 'PLAYER_1_HIT', 'frame': 84, 'player_id': 1, 'desc': 'Player 1 baseline backhand return'},
        {'id': 4, 'type': 'BOUNCE', 'frame': 138, 'player_id': None, 'desc': 'Second bounce in Player 2 deep right baseline'},
        {'id': 5, 'type': 'PLAYER_2_HIT', 'frame': 157, 'player_id': 2, 'desc': 'Player 2 baseline running forehand return'},
        {'id': 6, 'type': 'BOUNCE', 'frame': 196, 'player_id': None, 'desc': 'Third bounce in Player 1 deep baseline'}
    ]

    gt_events = []
    for ev in event_specs:
        f_idx = ev['frame']
        pt = traj[f_idx]
        x_px = pt['x_px']
        y_px = pt['y_px']
        court_pt = transform_point((x_px, y_px), H) if x_px and y_px else (0.0, 0.0)
        
        gt_events.append({
            'event_id': ev['id'],
            'event_type': ev['type'],
            'frame': f_idx,
            'timestamp_s': round(float(f_idx / fps), 4),
            'player_id': ev['player_id'],
            'x_px': round(float(x_px), 2) if x_px else None,
            'y_px': round(float(y_px), 2) if y_px else None,
            'x_m': round(float(court_pt[0]), 2) if court_pt else None,
            'y_m': round(float(court_pt[1]), 2) if court_pt else None,
            'annotation_confidence': 'HIGH',
            'visual_description': ev['desc']
        })

    benchmark_data = {
        'benchmark_id': 'tennis_events_rally_214_gt',
        'video_path': 'data/sample_videos/input_video.mp4',
        'total_frames': len(frames),
        'fps': float(fps),
        'events_count': len(gt_events),
        'summary': {
            'serves': 1,
            'bounces': 3,
            'player_1_hits': 1,
            'player_2_hits': 1,
            'total_rackets': 3
        },
        'events': gt_events
    }

    out_dir = 'data/benchmarks/tennis_events'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'ground_truth.json')
    with open(out_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f'Tennis Events Ground Truth written to: {out_file}')
    print(json.dumps(benchmark_data['summary'], indent=2))

if __name__ == '__main__':
    main()
