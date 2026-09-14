"""Measure a verified RT-DETR person detector with the frozen BoT-SORT recipe."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '1'
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics.nn import tasks
from ultralytics.nn.modules import block, conv, head, transformer
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from scripts.evaluate.audit_player_racket_evidence_gaps import matches
from scripts.evaluate.benchmark_uvy_tracker_architectures import diagnose
from src.utils.video_frame_sequence import VideoFrameSequence


def coverage(detections, gt):
    return sum(len(matches([{'box': row[:4]} for row in rows], truth)) for rows, truth in zip(detections, gt, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = ROOT/'data/external/uvy_tennis_videos'
    checkpoint_manifest_path = ROOT/'artifacts/models/person/rtdetr_research/manifest.json'
    baseline_path = ROOT/'outputs/vision_upgrade_audit/uvy_tracker_architecture_clipped01/report.json'
    manifest, checkpoint_manifest, baseline = [json.loads(p.read_text()) for p in
        (dataset/'manifest.json', checkpoint_manifest_path, baseline_path)]
    if not all(r['complete'] for r in (manifest, checkpoint_manifest, baseline)) or digest(dataset/'manifest.json') != baseline['dataset_manifest_sha256']:
        raise ValueError('Incomplete or changed inputs')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    weights = ROOT/checkpoint_manifest['checkpoint']
    if digest(weights) != '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f' or digest(weights) != checkpoint_manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    allowed = [tasks.DetectionModel, torch.nn.MultiheadAttention, torch.nn.Sequential,
        torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.LayerNorm, torch.nn.ModuleList,
        torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.ReLU,
        torch.nn.GELU, torch.nn.Upsample, torch.nn.Dropout, torch.nn.SiLU, torch.nn.Identity, torch.nn.Embedding,
        conv.Concat, conv.DWConv, conv.LightConv, conv.Conv, conv.RepConv, block.HGBlock, block.HGStem, block.RepC3,
        head.RTDETRDecoder, transformer.MSDeformAttn, transformer.MLP, transformer.DeformableTransformerDecoder,
        transformer.AIFI, transformer.DeformableTransformerDecoderLayer]
    allowed_names = {f'{cls.__module__}.{cls.__name__}' for cls in allowed}
    declared = torch.serialization.get_unsafe_globals_in_checkpoint(weights)
    if not set(declared) <= allowed_names:
        raise ValueError('Unreviewed checkpoint globals')
    with torch.serialization.safe_globals(allowed):
        model = ultralytics.RTDETR(str(weights))
    if model.names[0] != 'person':
        raise ValueError('Unexpected model classes')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cv2.setNumThreads(1)
    installed = Path(ultralytics.__file__).parent
    config_path = installed/'cfg/trackers/botsort.yaml'
    if digest(config_path) != baseline['tracker_configs']['botsort']['sha256']:
        raise ValueError('Tracker configuration changed')
    protocol = ROOT/'docs/experiments/RTDETR_PLAYER_PROPOSALS.md'
    sources = ['scripts/evaluate/benchmark_rtdetr_player_proposals.py', 'scripts/evaluate/benchmark_uvy_tracker_architectures.py',
        'scripts/evaluate/audit_uvy_videos.py', 'scripts/evaluate/audit_player_racket_evidence_gaps.py',
        'scripts/evaluate/trace_player_identity_switches.py', 'src/evaluation/tracking_metrics.py', 'src/utils/video_frame_sequence.py']
    report = {'complete': False, 'qualification_evidence': False, 'checkpoint_sha256': digest(weights),
        'checkpoint_manifest_sha256': digest(checkpoint_manifest_path), 'weights_only_forced': True,
        'checkpoint_declared_globals': sorted(declared), 'dataset_manifest_sha256': digest(dataset/'manifest.json'),
        'baseline_report_sha256': digest(baseline_path), 'protocol_sha256': digest(protocol),
        'code_hashes': {name: digest(ROOT/name) for name in sources},
        'installed_source_hashes': {name: digest(installed/name) for name in ['models/rtdetr/predict.py', 'models/rtdetr/model.py',
            'nn/tasks.py', 'nn/modules/head.py', 'nn/modules/transformer.py', 'data/augment.py', 'trackers/bot_sort.py',
            'trackers/byte_tracker.py', 'trackers/utils/gmc.py', 'engine/results.py', 'utils/ops.py']},
        'configuration': {'detector': 'RTDETR_L', 'imgsz': 640, 'confidence': .1, 'classes': [0], 'max_det': 300,
            'nms': False, 'preprocessing': 'SCALE_FILL_SQUARE', 'output_image_boundary_clipping': True,
            'tracker_initialization_frame_rate': 30},
        'tracker_configs': {'botsort': {'sha256': digest(config_path), 'contents': YAML.load(config_path)}},
        'runtime': {'torch': str(torch.__version__), 'ultralytics': ultralytics.__version__, 'device': device}, 'sequences': {}}
    args.output.mkdir(parents=True)
    (args.output/'frozen_protocol.md').write_bytes(protocol.read_bytes())
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    for sequence, info in manifest['sequences'].items():
        raw, degenerate = [], 0
        started = time.perf_counter()
        with VideoFrameSequence(str(dataset/info['video'])) as frames:
            for index, frame in enumerate(frames):
                result = model.predict(frame, classes=[0], imgsz=640, conf=.1, max_det=300,
                    augment=False, device=device, verbose=False)[0]
                detections = result.boxes.data.cpu().numpy().copy()
                height, width = frame.shape[:2]
                detections[:, [0, 2]] = detections[:, [0, 2]].clip(0, width)
                detections[:, [1, 3]] = detections[:, [1, 3]].clip(0, height)
                valid = (detections[:, 2] > detections[:, 0]) & (detections[:, 3] > detections[:, 1])
                degenerate += int((~valid).sum())
                detections = detections[valid]
                if not np.isfinite(detections).all() or np.any(detections[:, 5] != 0):
                    raise ValueError('Invalid or non-person output')
                raw.append(detections.tolist())
                if index % 250 == 0:
                    print(json.dumps({'stage': 'detect', 'sequence': sequence, 'frames': index+1, 'total': len(frames)}), flush=True)
            count, fps = len(frames), frames.metadata.fps
            actual_shape = list(model.predictor.pre_transform([frames[0]])[0].shape)
        seconds = time.perf_counter()-started
        path = args.output/f'{sequence}_raw.json'
        path.write_text(json.dumps(raw, allow_nan=False), encoding='utf-8')
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', count)]
        old_raw_path = baseline_path.parent/baseline['sequences'][sequence]['raw_file']
        if digest(old_raw_path) != baseline['sequences'][sequence]['raw_sha256']:
            raise ValueError('YOLO raw cache changed')
        old_raw = json.loads(old_raw_path.read_text())
        entry = report['sequences'][sequence] = {'frames': count, 'fps': fps, 'image_size': [width, height],
            'actual_transformed_image_shape': actual_shape, 'raw_file': path.name, 'raw_sha256': digest(path),
            'degenerate_clipped_boxes': degenerate, 'raw_matched_proposals': coverage(raw, gt),
            'baseline_raw_matched_proposals': coverage(old_raw, gt), 'decode_detection_seconds': seconds, 'trackers': {}}
        save()
        print(json.dumps({'sequence': sequence, 'raw_matches': entry['raw_matched_proposals'], 'YOLO_raw_matches': entry['baseline_raw_matched_proposals'], 'seconds': seconds}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for sequence, info in manifest['sequences'].items():
        entry = report['sequences'][sequence]
        raw = json.loads((args.output/entry['raw_file']).read_text())
        tracker = BOTSORT(IterableSimpleNamespace(**YAML.load(config_path)), frame_rate=30)
        predictions = []
        started = time.perf_counter()
        with VideoFrameSequence(str(dataset/info['video'])) as frames:
            for frame, rows in zip(frames, raw, strict=True):
                tracks = tracker.update(Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), frame.shape[:2]), frame)
                height, width = frame.shape[:2]
                predictions.append([{'id': int(t[4]), 'box': [float(max(0, min(limit, x))) for x, limit in zip(t[:4], [width, height, width, height], strict=True)], 'confidence': float(t[5])} for t in tracks])
        seconds = time.perf_counter()-started
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', entry['frames'])]
        diagnostics = diagnose(gt, predictions)
        path = args.output/f'{sequence}_botsort_proposals.json'
        path.write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
        entry['trackers']['botsort'] = {'predictions_file': path.name, 'predictions_sha256': digest(path),
            'decode_tracking_seconds': seconds, 'diagnostics': diagnostics,
            'baseline_diagnostics': baseline['sequences'][sequence]['trackers']['botsort']['diagnostics']}
        save()
        print(json.dumps({'sequence': sequence, 'BoT_matches': diagnostics['oracle_matched_proposals'], 'switches': diagnostics['official_clear_id_switches'], 'seconds': seconds}), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()
