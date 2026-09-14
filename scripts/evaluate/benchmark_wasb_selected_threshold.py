"""Paired sparse test of the original checkpoint's training-selected threshold."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.evaluate.benchmark_ball_temporal_spacing import aligned_windows, infer_candidates
from scripts.evaluate.summarize_ball_validation import summarize
from src.detection.wasb_ball_detector import WASBBallDetector
from src.detection.tiled_wasb_candidates import overlapping_tiles, merge_by_confidence
from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.video_frame_sequence import VideoFrameSequence


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def infer_pair(detector, frames, target, stride, selected_threshold):
    """Share model forward passes; independently decode both operating points."""
    if not 0 < selected_threshold < 1 or selected_threshold == .2:
        raise ValueError('Expected a distinct selected heatmap threshold')
    windows = aligned_windows(len(frames), target, stride)
    context = {i: frames[i] for i in sorted({i for indices, _ in windows for i in indices})}
    height, width = context[target].shape[:2]
    boxes = [(0, 0, width, height), *overlapping_tiles(width, height, .6)]
    proposals = {'control': [], 'selected': []}
    previous = detector.threshold
    try:
        for view, (left, top, right, bottom) in enumerate(boxes):
            maps = []
            for indices, slot in windows:
                heatmaps, shape = detector.predict_heatmaps([context[i][top:bottom, left:right] for i in indices])
                if len(heatmaps) != 3:
                    raise ValueError('Expected three aligned model outputs')
                maps.append(heatmaps[slot])
            averaged = np.mean(maps, axis=0)
            for name, threshold in [('control', .2), ('selected', selected_threshold)]:
                detector.threshold = threshold
                decoded = detector.decode_heatmaps([averaged], shape)[0]
                proposals[name].extend(dict(view=view, box=[left, top, right, bottom], mass_rank=rank,
                                            x=p.x_px + left, y=p.y_px + top, confidence=p.confidence)
                                       for rank, p in enumerate(decoded))
    finally:
        detector.threshold = previous
    result = {}
    for name, candidates in proposals.items():
        merged = merge_by_confidence([BallObservation(p['x'], p['y'], p['confidence']) for p in candidates], (width, height))
        result[name] = dict(candidates=candidates, prediction_xy=[merged[0].x_px, merged[0].y_px] if merged else None)
    return result, windows


def same_point(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) <= .0001


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'outputs/vision_upgrade_audit/wasb_selected_threshold01'
    target = out / 'report.json'
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    selection_path = ROOT / 'outputs/vision_upgrade_audit/wasb_training_threshold01/report.json'
    review_path = selection_path.with_name('review.json')
    audit_path = ROOT / 'outputs/vision_upgrade_audit/ball_spaced_error_audit01/report.json'
    selection, review, audit = read(selection_path), read(review_path), read(audit_path)
    assert selection['complete'] and review['complete'] and audit['complete']
    assert digest(selection_path) == review['report_sha256']
    for path, checksum in selection['code_hashes'].items():
        assert digest(ROOT / path) == checksum
    for path, checksum in audit['sources'].items():
        assert digest(ROOT / path) == checksum
    original = next(m for m in selection['models'] if m['name'] == 'original')
    threshold = float(original['selected_threshold'])
    assert threshold == .35  # Frozen before opening external scores.
    checkpoint = ROOT / original['checkpoint']
    assert digest(checkpoint) == original['checkpoint_sha256']
    training = read(ROOT / 'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json')
    assert digest(ROOT / 'artifacts/training/vision_upgrade/wasb_pilot03_spatial/manifest.json') == selection['training_manifest_sha256']
    training_ids = {c.rsplit('_', 1)[0] for c in training['train_clips'] + training['selection_clips']}
    assert len(audit['rows']) == 900 and len({(r['clip'], r['frame']) for r in audit['rows']}) == 900
    assert training_ids.isdisjoint({r['clip'].rsplit('_', 1)[0] for r in audit['rows']})
    manifests = {}
    for dataset in sorted({r['dataset'] for r in audit['rows']}):
        folder = ROOT / 'data/external' / dataset
        manifest = read(folder / 'manifest.json')
        assert manifest['split'] == 'VALIDATION_ONLY'
        for item in manifest['files']:
            assert digest(folder / item['path']) == item['sha256']
        manifests[dataset] = digest(folder / 'manifest.json')
    provenance = dict(selection_sha256=digest(selection_path), selection_review_sha256=digest(review_path),
                      audit_sha256=digest(audit_path), dataset_manifests=manifests,
                      checkpoint_sha256=digest(checkpoint),
                      protocol_sha256=digest(ROOT / 'docs/experiments/WASB_SELECTED_THRESHOLD_PROTOCOL.md'),
                      code_hashes={p: digest(ROOT / p) for p in [
                          'scripts/evaluate/benchmark_wasb_selected_threshold.py',
                          'scripts/evaluate/benchmark_ball_temporal_spacing.py',
                          'src/detection/wasb_ball_detector.py', 'src/detection/tiled_wasb_candidates.py',
                          'src/utils/video_frame_sequence.py', 'src/evaluation/point_metrics.py',
                          'scripts/evaluate/summarize_ball_validation.py']})
    if args.resume:
        report = read(target)
        assert not report['complete'] and report['provenance'] == provenance
    else:
        out.mkdir(parents=True)
        report = dict(complete=False, qualification_evidence=False, provenance=provenance,
                      configuration=dict(control_threshold=.2, selected_threshold=threshold, crop_fraction=.6,
                                         target_temporal_fps=30, merge_radius_reference_px=4), clips=[])
    def save():
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)
    save()
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded benchmark requires the existing CUDA environment')
    report['runtime'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                             device=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    detector = WASBBallDetector(checkpoint, device='cuda', threshold=.2)
    completed = {c['clip'] for c in report['clips']}
    assert len(completed) == len(report['clips'])
    for clip_name in sorted({r['clip'] for r in audit['rows']}):
        if clip_name in completed:
            continue
        rows = sorted([r for r in audit['rows'] if r['clip'] == clip_name], key=lambda r: r['frame'])
        folder = ROOT / 'data/external' / rows[0]['dataset']
        video = folder / f'tennis/videos/{clip_name}.mp4'
        match, rally = clip_name.rsplit('_', 1)
        label_path = folder / f'tennis/all/{match}/csv/{rally}_ball.csv'
        with label_path.open(newline='') as handle:
            labels = {int(r['Frame']): r for r in csv.DictReader(handle)}
        assert set(labels) == {r['frame'] for r in rows}
        entry = dict(clip=clip_name, dataset=rows[0]['dataset'], video_sha256=digest(video),
                     label_sha256=digest(label_path), rows=[], control_replay_max_native_px=0.)
        started = time.perf_counter()
        with VideoFrameSequence(str(video), cache_size=12) as frames:
            width, height, fps = frames.metadata.width, frames.metadata.height, frames.metadata.fps
            stride = max(1, int(math.floor(fps / 30 + .5)))
            entry.update(width=width, height=height, fps=fps, stride=stride, frames=len(frames))
            for index, row in enumerate(rows):
                assert (row['width'], row['height'], row['fps'], row['stride']) == (width, height, fps, stride)
                label = labels[row['frame']]
                expected = [float(label['X']) * width / 1920, float(label['Y']) * height / 1080] if int(label['Visibility']) else None
                assert expected == row['target_xy']
                pair, windows = infer_pair(detector, frames, row['frame'], stride, threshold)
                control = pair['control']['prediction_xy']
                assert same_point(control, row['stages']['raw'])
                if control is not None:
                    entry['control_replay_max_native_px'] = max(entry['control_replay_max_native_px'], max(abs(x-y) for x,y in zip(control, row['stages']['raw'], strict=True)))
                if index == 0:
                    prior, candidates, old_windows = infer_candidates(detector, frames, row['frame'], stride)
                    assert candidates == pair['control']['candidates'] and windows == old_windows
                    assert control == ([prior[0].x_px, prior[0].y_px] if prior else None)
                    entry['existing_inference_reference_exact'] = True
                entry['rows'].append(dict(frame=row['frame'], target_xy=expected, windows=windows, **pair))
        torch.cuda.synchronize()
        entry['seconds'] = time.perf_counter() - started
        report['clips'].append(entry)
        save()
        print(json.dumps(dict(clip=clip_name, labels=len(rows), completed_clips=len(report['clips']), seconds=entry['seconds'])), flush=True)
    assert len(report['clips']) == 18 and sum(len(c['rows']) for c in report['clips']) == 900
    scores = {name: summarize([dict(clip=c['clip'], frame=r['frame'], width=c['width'], height=c['height'],
                                    target_xy=r['target_xy'], prediction_xy=r[name]['prediction_xy'])
                              for c in report['clips'] for r in c['rows']]) for name in ('control', 'selected')}
    a, b = scores['control']['pooled'], scores['selected']['pooled']
    report.update(complete=True, scores=scores, advance_to_continuous=(b['f1'] > a['f1'] and b['precision'] >= a['precision'] and b['recall'] >= a['recall']),
                  peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    save()
    print(json.dumps(dict(complete=True, advance_to_continuous=report['advance_to_continuous'], scores={n: {k:s['pooled'][k] for k in ('true_positives','false_positives','false_negatives','precision','recall','f1')} for n,s in scores.items()})), flush=True)


if __name__ == '__main__':
    main()
