"""Paired tracker benchmark with one shared confidence-.10 detection cache."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import ultralytics
from ultralytics.engine.results import Boxes
from ultralytics.nn import tasks
from ultralytics.nn.modules import block, conv, head
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from scripts.evaluate.audit_player_racket_evidence_gaps import matches
from scripts.evaluate.trace_player_identity_switches import trace
from src.utils.video_frame_sequence import VideoFrameSequence


def diagnose(gt, predictions):
    identities = defaultdict(Counter)
    gt_counts = Counter(r['id'] for frame in gt for r in frame)
    for persons, labels in zip(predictions, gt, strict=True):
        for pi, gi in matches(persons, labels):
            identities[labels[gi]['id']][persons[pi]['id']] += 1
    switches, clear, _ = trace(gt, predictions, ROOT/'artifacts/research/TrackEval')
    return {'gt_boxes': sum(gt_counts.values()), 'oracle_matched_proposals': sum(sum(v.values()) for v in identities.values()),
            'source_ids': len({r['id'] for frame in predictions for r in frame}),
            'official_clear_id_switches': clear['IDSW'], 'switch_events': switches,
            'per_gt': {str(identity): {'gt_frames': n, 'matched_frames': sum(identities[identity].values()),
                'matched_source_ids': dict(identities[identity]),
                'dominant_source_matched_frames': max(identities[identity].values(), default=0)} for identity, n in gt_counts.items()},
            'scope': 'LABEL_ASSISTED_PERSON_COVERAGE_AND_FRAGMENTATION; NOT_ACTIVE_PLAYER_PRECISION_OR_QUALIFICATION'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    old_path = ROOT/'outputs/vision_upgrade_audit/uvy_person_tracked640/report.json'
    old = json.loads(old_path.read_text())
    if not manifest['complete'] or not old['complete'] or digest(dataset/'manifest.json') != old['dataset_manifest_sha256']:
        raise ValueError('Incomplete or changed dataset')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset file changed')
    weights = ROOT/'yolo11m.pt'
    if digest(weights) != old['checkpoint_sha256'] or digest(weights) != 'd5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95':
        raise ValueError('Original model changed')
    allowed = [tasks.DetectionModel, IterableSimpleNamespace, torch.nn.SiLU, torch.nn.BatchNorm2d,
               torch.nn.ModuleList, torch.nn.Sequential, torch.nn.Conv2d, torch.nn.Identity, torch.nn.MaxPool2d,
               torch.nn.Upsample, block.Attention, block.Bottleneck, block.C2PSA, block.C3k, block.C3k2,
               block.DFL, block.PSABlock, block.SPPF, conv.Concat, conv.Conv, conv.DWConv, head.Detect]
    with torch.serialization.safe_globals(allowed):
        model = ultralytics.YOLO(str(weights))
    installed = Path(ultralytics.__file__).parent
    configs = {kind: installed/'cfg/trackers'/f'{kind}.yaml' for kind in ('bytetrack', 'botsort')}
    source_files = ['trackers/byte_tracker.py', 'trackers/bot_sort.py', 'trackers/basetrack.py',
                    'trackers/utils/gmc.py', 'trackers/utils/matching.py', 'trackers/utils/kalman_filter.py']
    protocol = ROOT/'docs/experiments/UVY_BOTSORT_COMPARISON.md'
    report = {'complete': False, 'qualification_evidence': False, 'dataset_manifest_sha256': digest(dataset/'manifest.json'),
        'checkpoint_sha256': digest(weights), 'old_person_report_sha256': digest(old_path), 'weights_only_forced': True,
        'protocol_sha256': digest(protocol), 'ultralytics_version': ultralytics.__version__, 'torch_version': str(torch.__version__),
        'configuration': {'imgsz': 640, 'confidence': .1, 'iou': .7, 'max_det': 300, 'tracker_initialization_frame_rate': 30},
        'tracker_configs': {kind: {'sha256': digest(path), 'contents': YAML.load(path)} for kind, path in configs.items()},
        'installed_source_hashes': {name: digest(installed/name) for name in source_files},
        'code_hashes': {name: digest(ROOT/name) for name in ('scripts/evaluate/benchmark_uvy_tracker_architectures.py',
            'scripts/evaluate/audit_player_racket_evidence_gaps.py', 'scripts/evaluate/trace_player_identity_switches.py',
            'src/evaluation/tracking_metrics.py', 'src/utils/video_frame_sequence.py')}, 'sequences': {}}
    args.output.mkdir(parents=True)
    (args.output/'frozen_protocol.md').write_bytes(protocol.read_bytes())
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    # Detect all videos before constructing a tracker; retain complete raw caches.
    for sequence, info in manifest['sequences'].items():
        started = time.perf_counter()
        raw = []
        with VideoFrameSequence(str(dataset/info['video'])) as frames:
            for i, frame in enumerate(frames):
                result = model.predict(frame, classes=[0], imgsz=640, conf=.1, iou=.7, max_det=300,
                                       augment=False, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False)[0]
                raw.append(result.boxes.data.cpu().numpy().tolist())
                if i % 250 == 0:
                    print(json.dumps({'stage': 'detect', 'sequence': sequence, 'done': i+1, 'total': len(frames)}), flush=True)
            fps, count = frames.metadata.fps, len(frames)
        path = args.output/f'{sequence}_raw.json'
        path.write_text(json.dumps(raw, allow_nan=False), encoding='utf-8')
        report['sequences'][sequence] = {'frames': count, 'fps': fps, 'raw_file': path.name, 'raw_sha256': digest(path),
            'decode_detection_seconds': time.perf_counter()-started, 'trackers': {}}
        save()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for kind, tracker_class in (('bytetrack', BYTETracker), ('botsort', BOTSORT)):
        for sequence, info in manifest['sequences'].items():
            entry = report['sequences'][sequence]
            raw = json.loads((args.output/entry['raw_file']).read_text())
            tracker = tracker_class(IterableSimpleNamespace(**YAML.load(configs[kind])), frame_rate=30)
            predictions = []
            started = time.perf_counter()
            with VideoFrameSequence(str(dataset/info['video'])) as frames:
                for frame, detections in zip(frames, raw, strict=True):
                    boxes = Boxes(np.asarray(detections, dtype=np.float32).reshape(-1, 6), frame.shape[:2])
                    tracks = tracker.update(boxes, frame)
                    predictions.append([{'id': int(t[4]), 'box': [float(v) for v in t[:4]], 'confidence': float(t[5])} for t in tracks])
            seconds = time.perf_counter()-started
            gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', entry['frames'])]
            metrics = diagnose(gt, predictions)
            path = args.output/f'{sequence}_{kind}_proposals.json'
            path.write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
            value = {'decode_tracking_seconds': seconds, 'diagnostics': metrics, 'predictions_file': path.name, 'predictions_sha256': digest(path)}
            if kind == 'bytetrack':
                previous_path = old_path.parent/old['sequences'][sequence]['predictions_file']
                if digest(previous_path) != old['sequences'][sequence]['predictions_sha256']:
                    raise ValueError('Old person cache changed')
                previous = json.loads(previous_path.read_text())
                value['old_cache_exact_frames'] = sum(a == b for a, b in zip(previous, predictions, strict=True))
                value['old_cache_total_frames'] = len(previous)
            entry['trackers'][kind] = value
            save()
            print(json.dumps({'stage': kind, 'sequence': sequence, 'seconds': seconds, 'coverage': metrics['oracle_matched_proposals'],
                              'source_ids': metrics['source_ids'], 'switches': metrics['official_clear_id_switches'],
                              'old_exact_frames': value.get('old_cache_exact_frames')}), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
