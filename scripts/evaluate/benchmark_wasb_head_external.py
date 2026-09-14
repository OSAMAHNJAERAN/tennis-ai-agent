"""Frozen external development comparison for the verified head-only pilot."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.benchmark_wasb_selected_threshold import digest, read
from scripts.evaluate.benchmark_ball_temporal_spacing import infer_candidates, aligned_windows
from scripts.evaluate.select_wasb_training_threshold import infer_thresholds
from scripts.evaluate.summarize_ball_validation import summarize


def advance_gate(scores):
    a, b = [scores[n]['pooled'] for n in ('original', 'adapted')]
    passing = {n: sum(m['precision'] is not None and m['recall'] is not None and
                      m['precision'] >= .95 and m['recall'] >= .95
                      for m in scores[n]['per_clip'].values()) for n in scores}
    return (b['f1'] > a['f1'] and b['precision'] >= a['precision'] and
            b['recall'] >= a['recall'] and passing['adapted'] >= passing['original']), passing


def main():
    import numpy as np
    import torch
    from src.detection.wasb_ball_detector import WASBBallDetector
    from src.utils.video_frame_sequence import VideoFrameSequence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT/'outputs/vision_upgrade_audit/wasb_head_external01'
    target = out/'report.json'
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    training_path = ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04/manifest.json'
    internal_path = ROOT/'outputs/vision_upgrade_audit/wasb_head_pilot04/review.json'
    baseline_path = ROOT/'outputs/vision_upgrade_audit/wasb_selected_threshold01/report.json'
    baseline_review_path = baseline_path.with_name('review.json')
    training, internal, baseline, baseline_review = map(read, [training_path, internal_path, baseline_path, baseline_review_path])
    assert all(d['complete'] for d in (training, internal, baseline, baseline_review))
    assert digest(training_path) == internal['manifest_sha256']
    assert internal['next_gate'] == 'FROZEN_EXTERNAL_COMPARISON'
    assert digest(baseline_path) == baseline_review['report_sha256']
    assert all(m['selected_threshold'] == '0.35' for m in internal['models'])
    assert baseline['configuration']['selected_threshold'] == .35
    for collection in [training['code_hashes'], baseline['provenance']['code_hashes']]:
        for path, checksum in collection.items():
            assert digest(ROOT/path) == checksum
    for name in ('initial', 'trained'):
        assert digest(ROOT/training[name+'_checkpoint']) == training[name+'_checkpoint_sha256']
    assert baseline['provenance']['checkpoint_sha256'] == training['initial_checkpoint_sha256']
    training_ids = {c.rsplit('_', 1)[0] for c in training['train_clips']+training['selection_clips']}
    assert training_ids.isdisjoint({c['clip'].rsplit('_', 1)[0] for c in baseline['clips']})
    assert len(baseline['clips']) == 18 and sum(len(c['rows']) for c in baseline['clips']) == 900
    for dataset, checksum in baseline['provenance']['dataset_manifests'].items():
        folder = ROOT/'data/external'/dataset
        assert digest(folder/'manifest.json') == checksum
        manifest = read(folder/'manifest.json')
        assert manifest['split'] == 'VALIDATION_ONLY'
        for item in manifest['files']:
            assert digest(folder/item['path']) == item['sha256']
    code = ['scripts/evaluate/benchmark_wasb_head_external.py',
            'scripts/evaluate/select_wasb_training_threshold.py',
            *baseline['provenance']['code_hashes']]
    provenance = dict(training_manifest_sha256=digest(training_path), internal_review_sha256=digest(internal_path),
                      baseline_sha256=digest(baseline_path), baseline_review_sha256=digest(baseline_review_path),
                      initial_checkpoint_sha256=training['initial_checkpoint_sha256'],
                      trained_checkpoint_sha256=training['trained_checkpoint_sha256'],
                      protocol_sha256=digest(ROOT/'docs/experiments/WASB_HEAD_EXTERNAL_PROTOCOL.md'),
                      dataset_manifests=baseline['provenance']['dataset_manifests'],
                      code_hashes={p:digest(ROOT/p) for p in code})
    if args.resume:
        report = read(target)
        assert not report['complete'] and report['provenance'] == provenance
    else:
        out.mkdir(parents=True)
        report = dict(complete=False, qualification_evidence=False, provenance=provenance,
                      configuration=dict(original_threshold=.35, adapted_threshold=.35, crop_fraction=.6,
                                         target_temporal_fps=30, merge_radius_reference_px=4), clips=[])
    def save():
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(target)
    save()
    assert torch.cuda.is_available()
    report['runtime'] = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                             device=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    detector = WASBBallDetector(ROOT/training['trained_checkpoint'], device='cuda', threshold=.35)
    completed = {c['clip'] for c in report['clips']}
    assert len(completed) == len(report['clips'])
    for old in baseline['clips']:
        if old['clip'] in completed:
            continue
        folder = ROOT/'data/external'/old['dataset']
        video = folder/f"tennis/videos/{old['clip']}.mp4"
        match, rally = old['clip'].rsplit('_', 1)
        label_path = folder/f'tennis/all/{match}/csv/{rally}_ball.csv'
        assert digest(video) == old['video_sha256'] and digest(label_path) == old['label_sha256']
        with label_path.open(newline='') as handle:
            source = list(csv.DictReader(handle))
        labels = {int(r['Frame']):r for r in source}
        assert len(source) == len(labels) == len(old['rows'])
        assert set(labels) == {r['frame'] for r in old['rows']}
        entry = {k:old[k] for k in ('clip', 'dataset', 'width', 'height', 'fps', 'stride', 'frames', 'video_sha256', 'label_sha256')}
        entry['rows'] = []
        started = time.perf_counter()
        with VideoFrameSequence(str(video), cache_size=12) as frames:
            width, height = frames.metadata.width, frames.metadata.height
            fps = frames.metadata.fps
            stride = max(1, int(math.floor(fps/30+.5)))
            assert (width, height, fps, stride, len(frames)) == tuple(old[k] for k in ('width', 'height', 'fps', 'stride', 'frames'))
            for i, row in enumerate(old['rows']):
                label = labels[row['frame']]
                expected = [float(label['X'])*width/1920, float(label['Y'])*height/1080] if int(label['Visibility']) else None
                assert expected == row['target_xy']
                assert json.loads(json.dumps(aligned_windows(len(frames), row['frame'], stride))) == row['windows']
                selected, candidates, windows = infer_candidates(detector, frames, row['frame'], stride)
                prediction = [selected[0].x_px, selected[0].y_px] if selected else None
                assert all(math.isfinite(p[k]) for p in candidates for k in ('x', 'y', 'confidence'))
                if i == 0:
                    reference, reference_windows = infer_thresholds(detector, frames, row['frame'], stride)
                    assert reference_windows == windows
                    assert reference['0.35']['candidates'] == candidates
                    assert reference['0.35']['prediction_xy'] == prediction
                    entry['internal_inference_reference_exact'] = True
                entry['rows'].append(dict(frame=row['frame'], target_xy=expected, windows=windows,
                                          original=row['selected'], control=row['control'],
                                          adapted=dict(prediction_xy=prediction, candidates=candidates)))
        torch.cuda.synchronize()
        entry['seconds'] = time.perf_counter()-started
        report['clips'].append(entry)
        save()
        print(json.dumps(dict(clip=entry['clip'], completed_clips=len(report['clips']), seconds=entry['seconds'])), flush=True)
    assert len(report['clips']) == 18 and sum(len(c['rows']) for c in report['clips']) == 900
    scores = {name:summarize([dict(clip=c['clip'], frame=r['frame'], width=c['width'], height=c['height'],
                                  target_xy=r['target_xy'], prediction_xy=r[name]['prediction_xy'])
                            for c in report['clips'] for r in c['rows']]) for name in ('original', 'adapted', 'control')}
    for name, old_name in [('original', 'selected'), ('control', 'control')]:
        assert scores[name] == baseline['scores'][old_name]
    advance, passing = advance_gate(scores)
    report.update(complete=True, scores=scores, individually_passing_clips=passing,
                  advance_to_continuous=advance, peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    save()
    print(json.dumps(dict(complete=True, advance_to_continuous=advance, passing=passing,
                         metrics={n:{k:m['pooled'][k] for k in ('true_positives', 'false_positives', 'false_negatives', 'precision', 'recall', 'f1')} for n,m in scores.items()})), flush=True)


if __name__ == '__main__':
    main()
