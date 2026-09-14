"""Replay fixed nearest-person racket ownership without new model inference."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import evaluate_sequence
from src.tracking.global_racket_tracking import GlobalRacketTracking
from src.tracking.nearest_racket_ownership import NearestRacketOwnership
from src.tracking.racket_supported_players import select_racket_supported_people
from src.utils.bbox_utils import BBox
from src.utils.video_frame_sequence import VideoFrameSequence


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/vision_upgrade_audit/uvy_racket_supported_players_pilot01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads((args.source/'report.json').read_text())
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    person_path = ROOT/'outputs/vision_upgrade_audit/uvy_person_tracked640/report.json'
    person = json.loads(person_path.read_text())
    if not source['complete'] or not manifest['complete'] or not person['complete']:
        raise ValueError('Incomplete input')
    if digest(dataset/'manifest.json') != source['dataset_manifest_sha256'] or digest(person_path) != source['person_report_sha256']:
        raise ValueError('Input provenance changed')
    for name, expected in source['code_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'Baseline source changed: {name}')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    if set(source['sequences']) != set(manifest['sequences']):
        raise ValueError('Incomplete sequence coverage')
    report = {'complete': False, 'qualification_evidence': False,
              'scope': 'DEVELOPMENT_REPLAY_NEAREST_PERSON_CROP_PROVENANCE_INTERSECTION',
              'source_report_sha256': digest(args.source/'report.json'),
              'dataset_manifest_sha256': digest(dataset/'manifest.json'),
              'protocol_sha256': digest(ROOT/'docs/experiments/UVY_RACKET_SUPPORTED_PLAYERS.md'),
              'baseline_code_hashes': source['code_hashes'],
              'code_hashes': {name: digest(ROOT/name) for name in (
                  'scripts/evaluate/replay_uvy_racket_ownership.py', 'src/tracking/nearest_racket_ownership.py')},
              'configuration': {**source['configuration'], 'nearest_gap_tolerance_px': 1e-9},
              'limitations': source['limitations'] + ['Nearest rectangle is not independently validated ownership.',
                                                       'Same recordings informed the failure diagnosis.'],
              'sequences': {}}
    args.output.mkdir(parents=True)
    for sequence, cache in source['sequences'].items():
        started = time.perf_counter()
        proposals_path = person_path.parent/person['sequences'][sequence]['predictions_file']
        if digest(proposals_path) != cache['person_predictions_sha256']:
            raise ValueError('Person cache changed')
        for key in ('observations', 'selected'):
            if digest(args.source/cache[f'{key}_file']) != cache[f'{key}_sha256']:
                raise ValueError('Racket cache changed')
        detections = json.loads(proposals_path.read_text())
        observations = json.loads((args.source/cache['observations_file']).read_text())
        if [r['frame'] for r in observations] != cache['sampled_frames'] or len(detections) != cache['frames']:
            raise ValueError('Frame alignment changed')
        baseline = GlobalRacketTracking(model=object())
        candidate = NearestRacketOwnership(model=object())
        evidence, original_evidence, saved = [], [], []
        for item in observations:
            index = item['frame']
            players = {r['id']: BBox(*r['box'], confidence=r['confidence'], track_id=r['id']) for r in detections[index]}
            outputs = baseline.assign_candidates(players, item['raw_candidates'], index/cache['fps'])
            if normalized(outputs) != item['assigned']:
                raise ValueError(f'Baseline assignment differs: {sequence}:{index}')
            changed = candidate.assign_candidates(players, item['raw_candidates'], index/cache['fps'])
            saved.append({'frame': index, 'assigned': changed})
            for results, target in ((outputs, original_evidence), (changed, evidence)):
                for identity, value in results.items():
                    if value['state'] == 'DETECTED':
                        target.append({'frame': index, 'source_id': identity,
                                       'confidence': value['confidence'], 'racket_box': value['bbox_xyxy']})
        reproduced, _ = select_racket_supported_people(detections, cache['fps'], cache['sampled_frames'], original_evidence)
        if reproduced != json.loads((args.source/cache['selected_file']).read_text()):
            raise ValueError('Baseline selected frames differ')
        selected, statistics = select_racket_supported_people(detections, cache['fps'], cache['sampled_frames'], evidence)
        gt = [[r for r in frame if r['class'] == 1] for frame in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', len(detections))]
        metrics = evaluate_sequence(gt, selected, ROOT/'artifacts/research/TrackEval')
        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            if len(frames) != len(detections) or frames.metadata.fps != cache['fps']:
                raise ValueError('Video alignment changed')
            for index in sorted({0, len(frames)//2, len(frames)-1}):
                image = frames[index].copy()
                for row in selected[index]:
                    x1, y1, x2, y2 = map(round, row['box'])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 230, 30), 2)
                    cv2.putText(image, f'candidate player {row["id"]}', (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 230, 30), 1)
                if not cv2.imwrite(str(args.output/f'{sequence}_{index+1:06}.jpg'), image):
                    raise OSError('Failed to save review image')
        files = {}
        for key, value in (('selected', selected), ('observations', saved)):
            path = args.output/f'{sequence}_{key}.json'
            path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
            files[f'{key}_file'], files[f'{key}_sha256'] = path.name, digest(path)
        report['sequences'][sequence] = {
            'frames': len(detections), 'fps': cache['fps'], 'baseline_assignments_exact': True,
            'baseline_selected_frames_exact': True, 'racket_supports': evidence,
            'track_statistics': statistics, 'selected_observations': sum(map(len, selected)),
            'baseline_metrics': cache['metrics']['summary'], 'metrics': metrics,
            'seconds': time.perf_counter()-started, **files}
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'sequence': sequence, 'supports': len(evidence), 'metrics': metrics['summary']}), flush=True)
    report['complete'] = True
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
