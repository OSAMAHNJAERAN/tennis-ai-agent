"""Paired downstream player selection from verified ByteTrack/BoT-SORT caches."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import ultralytics
from scripts.evaluate import benchmark_racket_pilot04_restricted as safe
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.evaluation.tracking_metrics import evaluate_sequence
from src.tracking.flow_fragment_linking import link_player_fragments
from src.tracking.duplicate_player_handoff import reconcile_handoffs
from src.tracking.racket_supported_players import select_racket_supported_people
from src.tracking.nearest_racket_ownership import NearestRacketOwnership
from src.utils.video_frame_sequence import VideoFrameSequence
from src.utils.bbox_utils import BBox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--person-report', type=Path, default=ROOT/'outputs/vision_upgrade_audit/uvy_tracker_architecture_clipped01/report.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = ROOT/'data/external/uvy_tennis_videos'
    person_path = args.person_report
    training_path = ROOT/'artifacts/training/vision_upgrade/racket_yolo11m_pilot04/report.json'
    person, training, manifest = [json.loads(p.read_text()) for p in (person_path, training_path, dataset/'manifest.json')]
    if not all(r['complete'] for r in (person, training, manifest)) or digest(dataset/'manifest.json') != person['dataset_manifest_sha256']:
        raise ValueError('Incomplete or changed inputs')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    for name, expected in person['code_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Upstream benchmark code changed')
    weights = ROOT/training['best_checkpoint']
    if digest(weights) != training['best_checkpoint_sha256'] or digest(weights) != 'c4416708def99c60dc041aaa8fa5d6bbaf3d0d39f2b016876c61e2f87812495a':
        raise ValueError('Racket checkpoint changed')
    if digest(ROOT/'yolo11m.pt') != training['initial_checkpoint_sha256'] or training['initial_checkpoint_sha256'] != person['checkpoint_sha256']:
        raise ValueError('Original checkpoint changed')
    torch, np, block, conv, head = safe.torch, safe.np, safe.block, safe.conv, safe.head
    allowed = [safe.tasks.DetectionModel, safe.IterableSimpleNamespace, safe.BboxLoss, safe.DFLoss, safe.v8DetectionLoss,
        safe.TaskAlignedAssigner, np.dtype, np._core.multiarray.scalar, np.dtypes.Float64DType, np.dtypes.Float32DType,
        torch.nn.SiLU, torch.nn.BatchNorm2d, torch.nn.ModuleList, torch.nn.Sequential, torch.nn.Conv2d,
        torch.nn.Identity, torch.nn.BCEWithLogitsLoss, torch.nn.MaxPool2d, torch.nn.Upsample, block.Attention,
        block.Bottleneck, block.C2PSA, block.C3k, block.C3k2, block.DFL, block.PSABlock, block.SPPF,
        conv.Concat, conv.Conv, conv.DWConv, head.Detect]
    with torch.serialization.safe_globals(allowed):
        model = ultralytics.YOLO(str(weights))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cv2.setNumThreads(1)
    protocol = ROOT/'docs/experiments/TRACKER_RACKET_SELECTION.md'
    sources = ['scripts/evaluate/benchmark_tracker_racket_selection.py', 'scripts/evaluate/benchmark_racket_pilot04_restricted.py',
        'src/tracking/flow_fragment_linking.py', 'src/tracking/player_fragment_linking.py',
        'src/tracking/duplicate_player_handoff.py', 'src/tracking/racket_supported_players.py',
        'src/tracking/nearest_racket_ownership.py', 'src/tracking/global_racket_tracking.py',
        'src/tracking/racket_tracking.py', 'src/evaluation/tracking_metrics.py', 'src/utils/video_frame_sequence.py',
        'scripts/evaluate/audit_uvy_videos.py']
    report = {'complete': False, 'qualification_evidence': False, 'person_report_sha256': digest(person_path),
        'training_report_sha256': digest(training_path), 'dataset_manifest_sha256': digest(dataset/'manifest.json'),
        'checkpoint_sha256': digest(weights), 'weights_only_forced': True, 'protocol_sha256': digest(protocol),
        'code_hashes': {name: digest(ROOT/name) for name in sources},
        'runtime': {'torch': str(torch.__version__), 'ultralytics': ultralytics.__version__, 'device': device},
        'configuration': {'sample_seconds': .2, 'racket_confidence': .25, 'racket_imgsz': 640,
            'minimum_racket_support_frames': 2, 'minimum_person_observation_seconds': .5,
            'ownership': 'NEAREST_PERSON', 'variants': ['raw_ids', 'flow_handoff'], 'offline': True}, 'sequences': {}}
    args.output.mkdir(parents=True)
    (args.output/'frozen_protocol.md').write_bytes(protocol.read_bytes())
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    for sequence, info in person['sequences'].items():
        count, fps = info['frames'], info['fps']
        sampled = sorted({round(k*.2*fps) for k in range(math.ceil(count/fps/.2)) if round(k*.2*fps) < count})
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', count)]
        entry = report['sequences'][sequence] = {'frames': count, 'fps': fps, 'sampled_frames': sampled, 'trackers': {}}
        for kind, cache in info['trackers'].items():
            path = person_path.parent/cache['predictions_file']
            if digest(path) != cache['predictions_sha256']:
                raise ValueError('Person cache changed')
            detections = json.loads(path.read_text())
            if len(detections) != count:
                raise ValueError('Frame mismatch')
            tracker = NearestRacketOwnership(model=model, device=device, confidence=.25, imgsz=640)
            evidence, observations = [], []
            started = time.perf_counter()
            with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
                if len(frames) != count or frames.metadata.fps != fps:
                    raise ValueError('Video alignment changed')
                for number, index in enumerate(sampled):
                    players = {r['id']: BBox(*r['box'], confidence=r['confidence'], track_id=r['id']) for r in detections[index]}
                    outputs = tracker.observe(frames[index], players, index/fps)
                    for identity, value in outputs.items():
                        if value['state'] == 'DETECTED':
                            evidence.append({'frame': index, 'source_id': identity, 'confidence': value['confidence'], 'racket_box': value['bbox_xyxy']})
                    observations.append({'frame': index, 'assigned': outputs,
                        'raw_candidates': [{**r, 'sources': sorted(r['sources'])} for r in tracker.last_candidates]})
                    if number % 50 == 0:
                        print(json.dumps({'sequence': sequence, 'tracker': kind, 'samples': number+1, 'total': len(sampled)}), flush=True)
                inference_seconds = time.perf_counter()-started
                started = time.perf_counter()
                mapping, links, diagnostics = link_player_fragments(detections, fps, frames)
                remapped, supports, mapping, handoffs, rejected, suppressions = reconcile_handoffs(detections, fps, evidence, mapping)
                linking_seconds = time.perf_counter()-started
            obs_path = args.output/f'{sequence}_{kind}_rackets.json'
            obs_path.write_text(json.dumps(observations, allow_nan=False), encoding='utf-8')
            value = entry['trackers'][kind] = {'person_predictions_sha256': digest(path), 'racket_supports': evidence,
                'observations_file': obs_path.name, 'observations_sha256': digest(obs_path),
                'decode_racket_seconds': inference_seconds, 'linking_seconds': linking_seconds,
                'source_mapping': mapping, 'flow_links': links, 'flow_diagnostics': diagnostics, 'handoffs': handoffs,
                'rejected_handoffs': rejected, 'suppressions': suppressions, 'combined_supports': supports, 'variants': {}}
            for variant, rows, support in [('raw_ids', detections, evidence), ('flow_handoff', remapped, supports)]:
                selected, statistics = select_racket_supported_people(rows, fps, sampled, support)
                for original, chosen in zip(detections, selected, strict=True):
                    by_id = {r['id']: r for r in original}
                    for r in chosen:
                        source = by_id[r.get('source_id', r['id'])]
                        if r['box'] != source['box'] or r['confidence'] != source['confidence']:
                            raise ValueError('Selected observation fabricated or changed')
                selected_path = args.output/f'{sequence}_{kind}_{variant}.json'
                selected_path.write_text(json.dumps(selected, allow_nan=False), encoding='utf-8')
                metrics = evaluate_sequence(gt, selected, ROOT/'artifacts/research/TrackEval')
                value['variants'][variant] = {'selected_file': selected_path.name, 'selected_sha256': digest(selected_path),
                    'all_selected_observations_exact_source': True, 'track_statistics': statistics, 'metrics': metrics}
                print(json.dumps({'sequence': sequence, 'tracker': kind, 'variant': variant, 'metrics': metrics['summary']}), flush=True)
            save()
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
