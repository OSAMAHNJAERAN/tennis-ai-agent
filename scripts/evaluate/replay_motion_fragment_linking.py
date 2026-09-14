"""Evaluate the fixed boundary-IoU linking protocol on all UVY frames."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import evaluate_sequence
from src.tracking.motion_fragment_linking import link_player_fragments
from src.tracking.player_fragment_linking import remap_observations
from src.tracking.racket_supported_players import select_racket_supported_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT/'outputs/vision_upgrade_audit'
    source_path = base/'uvy_racket_supported_players_pilot02_nearest/report.json'
    people_path = base/'uvy_person_tracked640/report.json'
    source, people = [json.loads(p.read_text()) for p in (source_path, people_path)]
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if not all(r['complete'] for r in (source, people, manifest)) or digest(dataset/'manifest.json') != source['dataset_manifest_sha256']:
        raise ValueError('Incomplete or changed inputs')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    protocol = ROOT/'docs/experiments/PLAYER_MOTION_FRAGMENT_LINKING.md'
    report = {'complete': False, 'qualification_evidence': False, 'scope': 'OFFLINE_BIDIRECTIONAL_MOTION_FRAGMENT_LINKING_DEVELOPMENT_REPLAY',
              'source_report_sha256': digest(source_path), 'person_report_sha256': digest(people_path),
              'dataset_manifest_sha256': digest(dataset/'manifest.json'), 'protocol_sha256': digest(protocol),
              'configuration': {'maximum_elapsed_seconds': .2, 'minimum_bidirectional_iou': .5, 'fit_window_seconds': .1, 'minimum_fit_observations': 3, 'maximum_fit_residual_width_fraction': .25, 'tie_tolerance': 1e-9,
                                'edge_selection': 'MUTUAL_UNIQUE_MAXIMUM_IOU', 'racket_selection': source['configuration']},
              'code_hashes': {name: digest(ROOT/name) for name in (
                  'scripts/evaluate/replay_motion_fragment_linking.py', 'src/tracking/motion_fragment_linking.py', 'src/tracking/player_fragment_linking.py',
                  'src/tracking/racket_supported_players.py', 'src/evaluation/tracking_metrics.py')},
              'limitations': ['Whole-sequence source-ID linking without appearance or verified re-identification.',
                              'Unchanged publisher GT includes known omissions, loose boxes and a misclassified umpire.'],
              'sequences': {}}
    args.output.mkdir(parents=True)
    (args.output/'frozen_protocol.md').write_bytes(protocol.read_bytes())
    for sequence, info in source['sequences'].items():
        cache = people['sequences'][sequence]
        path = people_path.parent/cache['predictions_file']
        if digest(path) != cache['predictions_sha256']:
            raise ValueError('Person proposals changed')
        detections = json.loads(path.read_text())
        observations_path = source_path.parent/info['observations_file']
        selected_path = source_path.parent/info['selected_file']
        if digest(observations_path) != info['observations_sha256'] or digest(selected_path) != info['selected_sha256']:
            raise ValueError('Racket inputs changed')
        sampled = [r['frame'] for r in json.loads(observations_path.read_text())]
        original, _ = select_racket_supported_people(detections, info['fps'], sampled, info['racket_supports'])
        if original != json.loads(selected_path.read_text()):
            raise ValueError('Original selection does not reproduce')
        mapping, links = link_player_fragments(detections, info['fps'])
        remapped, evidence = remap_observations(detections, info['racket_supports'], mapping)
        selected, statistics = select_racket_supported_people(remapped, info['fps'], sampled, evidence)
        gt = [[r for r in row if r['class'] == 1] for row in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(detections))]
        metrics = evaluate_sequence(gt, selected, ROOT/'artifacts/research/TrackEval')
        output = args.output/f'{sequence}_selected.json'
        output.write_text(json.dumps(selected, allow_nan=False), encoding='utf-8')
        report['sequences'][sequence] = {'frames': len(detections), 'fps': info['fps'], 'source_mapping': mapping,
            'links': links, 'track_statistics': statistics, 'racket_supports': evidence,
            'original_selection_exact': True, 'baseline_metrics': info['metrics']['summary'], 'metrics': metrics,
            'selected_file': output.name, 'selected_sha256': digest(output),
            'person_predictions_sha256': cache['predictions_sha256']}
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'sequence': sequence, 'links': links, 'metrics': metrics['summary']}), flush=True)
    report['complete'] = True
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
