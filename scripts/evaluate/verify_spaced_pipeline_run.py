"""Compare actual pipeline exports with a frozen, fully filtered ball benchmark."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.review_ball_spacing_final import digest, load_final, read
from src.evaluation.point_metrics import evaluate_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    destination = args.run / 'ball_benchmark_parity.json'
    if destination.exists():
        raise FileExistsError(destination)
    benchmark, stream, clips = load_final(args.benchmark)
    clip = clips[args.clip]
    validation = read(args.run / 'run_validation.json')
    assert digest(args.dataset/'manifest.json') == stream['dataset_manifest_sha256']
    video = args.dataset/f'tennis/videos/{args.clip}.mp4'
    manifest = read(args.dataset/'manifest.json')
    video_record = next(item for item in manifest['files'] if item['path'].replace('\\','/') == f'tennis/videos/{args.clip}.mp4')
    assert digest(video) == video_record['sha256'] == validation['input_sha256']
    assert digest(validation['input_video']) == validation['input_sha256']
    detections = read(args.run / 'detections.json')
    spacing = read(args.run / 'temporal_spacing_audit.json')
    detours = read(args.run / 'temporal_detour_audit.json')
    expected_detours = next(c for c in benchmark['clips'] if c['clip'] == args.clip)
    assert validation['output_decoded_frames'] == clip['frames'] == len(detections['frames'])
    assert validation['output_fps'] == clip['fps'] == detections['metadata']['fps']
    assert [detections['metadata']['width'], detections['metadata']['height']] == clip['size']
    assert spacing['enabled'] and spacing['native_stride'] == clip['stride']
    assert detours['enabled'] and detours['rejected_frames'] == expected_detours['rejected_frames']
    maximum = 0.
    for index, (actual, expected) in enumerate(zip(detections['frames'], clip['final_points'], strict=True)):
        assert actual['frame_index'] == index
        point = actual['ball']['position_px']
        assert actual['ball']['state'] == ('DETECTED' if point is not None else 'MISSING')
        if point is None or expected is None:
            assert point == expected, f'Missing-state mismatch at frame {index}'
        else:
            maximum = max(maximum, max(abs(x-y) for x,y in zip(point, expected, strict=True)))
    assert maximum <= 1e-4, f'Coordinate mismatch: {maximum}'
    labels = [r for r in benchmark['labeled_rows'] if r['clip'] == args.clip]
    rows = [{**r, 'prediction_xy': detections['frames'][r['frame']]['ball']['position_px']} for r in labels]
    measured = evaluate_points(rows)
    assert measured == benchmark['per_clip'][args.clip]
    state = read(args.run / 'match_state.json')
    ball_metrics = read(args.run / 'ball_metrics.json')
    assert state['authoritative_events_enabled'] is False
    assert ball_metrics['summary']['available'] is False
    assert ball_metrics['summary']['maximum_speed_kmh'] is None
    source_paths = [args.benchmark, args.run/'detections.json', args.run/'run_config.yaml',
                    args.run/'run_validation.json', args.run/'temporal_spacing_audit.json',
                    args.run/'temporal_detour_audit.json']
    sources = ['src/pipeline/phase6_pipeline.py', 'src/detection/spaced_ball_stream.py',
               'src/tracking/selected_temporal_detours.py', 'src/detection/wasb_ball_detector.py',
               'src/detection/tiled_wasb_candidates.py', 'src/tracking/selected_patch_persistence.py',
               'src/tracking/candidate_pixel_motion.py', 'src/tracking/stationary_candidate_filter.py',
               'scripts/evaluate/verify_spaced_pipeline_run.py']
    report = {'complete': True, 'qualification_evidence': False, 'clip': args.clip,
              'source_reports': {str(p.resolve()): digest(p) for p in source_paths},
              'code_hashes': {name: digest(ROOT/name) for name in sources},
              'all_native_frames': clip['frames'], 'maximum_coordinate_difference_px': maximum,
              'all_missing_states_exact': True, 'detour_mask_exact': True,
              'labeled_metrics_exact': True, 'metrics': measured,
              'ball_speed_unavailable': True, 'event_authority_withheld': True,
              'scope': 'REAL_PIPELINE_BALL_PARITY; NOT_PLAYER_COURT_RACKET_OR_PHYSICAL_ANALYTICS_QUALIFICATION'}
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('metrics','source_reports','code_hashes')}))


if __name__ == '__main__':
    main()
