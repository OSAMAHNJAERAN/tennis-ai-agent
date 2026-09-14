"""Apply the installed Results.update image clipping to direct tracker caches."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate.benchmark_uvy_tracker_architectures import diagnose
from scripts.evaluate.audit_uvy_videos import digest, read_gt
from src.utils.video_frame_sequence import VideoFrameSequence
import ultralytics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = ROOT/'outputs/vision_upgrade_audit'
    source = base/'uvy_tracker_architecture_pilot01/report.json'
    prior = base/'uvy_person_tracked640/report.json'
    report, old = [json.loads(p.read_text()) for p in (source, prior)]
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    if not report['complete'] or not old['complete'] or digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
        raise ValueError('Incomplete or changed source')
    for item in manifest['files']:
        if digest(dataset/item['path']) != item['sha256']:
            raise ValueError('Dataset changed')
    for name, expected in report['code_hashes'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Original benchmark source changed')
    report['complete'] = False
    report['unclipped_source_report_sha256'] = digest(source)
    report['code_hashes']['scripts/evaluate/adapt_tracker_cache_to_image_bounds.py'] = digest(Path(__file__))
    installed = Path(ultralytics.__file__).parent
    report['installed_source_hashes']['engine/results.py'] = digest(installed/'engine/results.py')
    report['installed_source_hashes']['utils/ops.py'] = digest(installed/'utils/ops.py')
    report['adapter_correction'] = 'Clip x coordinates to [0,width], y to [0,height] as installed Results.update; preserve IDs, confidence, order and every frame. Earlier raw caches remain intact.'
    report['configuration']['output_image_boundary_clipping'] = True
    args.output.mkdir(parents=True)
    (args.output/'frozen_protocol.md').write_bytes((source.parent/'frozen_protocol.md').read_bytes())
    for sequence, info in report['sequences'].items():
        with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
            height, width = frames[0].shape[:2]
        info['image_size'] = [width, height]
        raw = source.parent/info['raw_file']
        if digest(raw) != info['raw_sha256']:
            raise ValueError('Raw detections changed')
        info['raw_file'] = str(raw)
        gt = [[r for r in rows if r['class'] == 1] for rows in read_gt(dataset/'UVY'/sequence/'gt/gt.txt', info['frames'])]
        for kind, entry in info['trackers'].items():
            path = source.parent/entry['predictions_file']
            if digest(path) != entry['predictions_sha256']:
                raise ValueError('Tracker cache changed')
            predictions = json.loads(path.read_text())
            changed = 0
            for rows in predictions:
                for row in rows:
                    clipped = [max(0., min(limit, x)) for x, limit in zip(row['box'], [width, height, width, height], strict=True)]
                    changed += clipped != row['box']
                    row['box'] = clipped
            target = args.output/path.name
            target.write_text(json.dumps(predictions, allow_nan=False), encoding='utf-8')
            entry['direct_tracker_predictions_sha256'] = entry['predictions_sha256']
            entry['predictions_sha256'] = digest(target)
            entry['clipped_boxes'] = changed
            entry['diagnostics'] = diagnose(gt, predictions)
            if kind == 'bytetrack':
                previous = prior.parent/old['sequences'][sequence]['predictions_file']
                if digest(previous) != old['sequences'][sequence]['predictions_sha256'] or predictions != json.loads(previous.read_text()):
                    raise ValueError('Clipped ByteTrack must reproduce every old frame')
                entry['old_cache_exact_frames'] = len(predictions)
            print(json.dumps({'sequence': sequence, 'tracker': kind, 'clipped_boxes': changed,
                'coverage': entry['diagnostics']['oracle_matched_proposals'], 'switches': entry['diagnostics']['official_clear_id_switches']}), flush=True)
    report['complete'] = True
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
