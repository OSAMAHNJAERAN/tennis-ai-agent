"""Reuse verified spatial crops and supply real FPS-aligned training contexts."""
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


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def real_context(length, frame, fps):
    if type(length) is not int or type(frame) is not int or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid temporal geometry')
    stride = max(1, int(math.floor(fps/30+.5)))
    indices = [frame+i*stride for i in (-2,-1,0,1,2)]
    if indices[0] < 0 or indices[-1] >= length:
        raise ValueError('All three output slots require complete real context')
    return stride, indices


def plan_samples(samples, clips):
    """Preserve targets/splits while checking the exact source-frame contract."""
    seen, planned = set(), []
    for sample in samples:
        key = sample['clip'], sample['frame']
        if key in seen:
            raise ValueError('Duplicate source training label')
        seen.add(key)
        c = clips[sample['clip']]
        if sample['split'] != c['split'] or len(sample['views']) != 5:
            raise ValueError('Training split or view count changed')
        stride, indices = real_context(c['frames'], sample['frame'], c['fps'])
        planned.append(dict(**sample, fps=c['fps'], stride=stride, context_indices=indices))
    return planned


def main():
    import cv2
    import numpy as np
    from scripts.train.finetune_wasb_spatial import crop_target
    from src.detection.tiled_wasb_candidates import overlapping_tiles
    from src.utils.video_frame_sequence import VideoFrameSequence
    import importlib.util
    geometry_path = ROOT/'artifacts/research/WASB-SBDT/src/utils/image.py'
    spec = importlib.util.spec_from_file_location('spaced_training_geometry', geometry_path)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = ROOT/'artifacts/training/vision_upgrade/wasb_spaced_cache60'
    target = out/'manifest.json'
    if out.exists() and not args.resume:
        raise FileExistsError(out)
    prior = ROOT/'artifacts/training/vision_upgrade/wasb_head_pilot04'
    run = read(prior/'manifest.json')
    audit_path = ROOT/'outputs/vision_upgrade_audit/ball_training_expansion60/report.json'
    audit = read(audit_path)
    assert run['complete'] and audit['complete'] and digest(audit_path) == run['audit_sha256']
    for p, checksum in run['code_hashes'].items():
        assert digest(ROOT/p) == checksum
    assert digest(prior/'training_samples.json') == run['samples_sha256']
    clips = {c['clip']:c for c in audit['clips']}
    samples = read(prior/'training_samples.json')
    plan = plan_samples(samples, clips)
    assert len(samples) == 2999
    for name, checksum in run['dataset_manifests'].items():
        folder = ROOT/name
        assert digest(folder/'manifest.json') == checksum
        for item in read(folder/'manifest.json')['files']:
            assert digest(folder/item['path']) == item['sha256']
    code = ['scripts/data/prepare_wasb_spaced_training.py','scripts/train/finetune_wasb_spatial.py',
            'scripts/train/finetune_wasb.py','src/detection/tiled_wasb_candidates.py',
            'src/utils/video_frame_sequence.py','artifacts/research/WASB-SBDT/src/utils/image.py']
    provenance = dict(prior_manifest_sha256=digest(prior/'manifest.json'), prior_samples_sha256=digest(prior/'training_samples.json'),
                      audit_sha256=digest(audit_path), dataset_manifests=run['dataset_manifests'],
                      protocol_sha256=digest(ROOT/'docs/experiments/WASB_SPACED_TRAINING_DATA_PROTOCOL.md'),
                      code_hashes={p:digest(ROOT/p) for p in code})
    if args.resume:
        report = read(target)
        assert not report['complete'] and report['provenance'] == provenance
        for c in report['clips']:
            for path, checksum in c['cache_hashes'].items():
                assert digest(ROOT/path) == checksum
    else:
        out.mkdir(parents=True)
        report = dict(complete=False, qualification_evidence=False, provenance=provenance,
                      runtime=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__), clips=[])
    def save():
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(target)
    save()
    completed = {c['clip'] for c in report['clips']}
    for clip in sorted(clips):
        if clip in completed:
            continue
        started = time.perf_counter()
        c = clips[clip]
        rows = [s for s in plan if s['clip'] == clip]
        dataset = ROOT/c['dataset']
        match, rally = clip.rsplit('_', 1)
        with (dataset/f'tennis/all/{match}/csv/{rally}_ball.csv').open(newline='') as handle:
            source = list(csv.DictReader(handle))
        labels = {int(r['Frame']):r for r in source}
        assert len(source) == len(labels) == len(rows)
        assert set(labels) == {r['frame'] for r in rows}
        needed = sorted({i for r in rows for i in r['context_indices']})
        available = {i for i in needed if all((prior/f'frame_cache/{clip}/view_{v}/{i:06}.jpg').exists() for v in range(5))}
        reusable = sorted(available)
        checked = sorted({reusable[i] for i in (0, len(reusable)//2, len(reusable)-1)}) if reusable else []
        new_indices = sorted(set(needed)-available)
        hashes, paths = {}, {}
        with VideoFrameSequence(str(dataset/f'tennis/videos/{clip}.mp4'), cache_size=12) as frames:
            assert (frames.metadata.width, frames.metadata.height, frames.metadata.fps, len(frames)) == tuple(c[k] for k in ('width','height','fps','frames'))
            width, height = c['width'], c['height']
            boxes = [(0,0,width,height), *overlapping_tiles(width,height,.6)]
            transforms = [crop_target(None, box, geometry)[1] for box in boxes]
            for row in rows:
                label = labels[row['frame']]
                point = [float(label['X'])*width/1920, float(label['Y'])*height/1080] if int(label['Visibility']) else None
                assert (point is not None) == row['source_visible']
                for view, box in zip(row['views'], boxes, strict=True):
                    assert crop_target(point, box, geometry)[0] == view['target']
            for index in sorted(set(checked)|set(new_indices)):
                frame = frames[index]
                for v, (box, transform) in enumerate(zip(boxes, transforms, strict=True)):
                    left, top, right, bottom = box
                    image = cv2.warpAffine(frame[top:bottom,left:right], transform, (512,288))
                    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY,95])
                    assert ok
                    if index in available:
                        assert (prior/f'frame_cache/{clip}/view_{v}/{index:06}.jpg').read_bytes() == encoded.tobytes()
                    else:
                        path = out/f'frame_cache/{clip}/view_{v}/{index:06}.jpg'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # A partial unfinished clip can be safely resumed only if any existing bytes match.
                        if path.exists():
                            assert path.read_bytes() == encoded.tobytes()
                        else:
                            path.write_bytes(encoded.tobytes())
                        assert digest(path) == hashlib.sha256(encoded.tobytes()).hexdigest()
            for index in needed:
                for v in range(5):
                    path = (prior if index in available else out)/f'frame_cache/{clip}/view_{v}/{index:06}.jpg'
                    image = cv2.imread(str(path))
                    assert image is not None and image.shape == (288,512,3) and image.dtype == np.uint8
                    relative = str(path.relative_to(ROOT))
                    hashes[relative] = digest(path)
                    paths[index,v] = str(path)
            updated = []
            for row in rows:
                views = [dict(target=v['target'], context_paths=[paths[i,n] for i in row['context_indices']]) for n,v in enumerate(row['views'])]
                updated.append({**row, 'views':views})
        report['clips'].append(dict(clip=clip, split=c['split'], fps=c['fps'], stride=rows[0]['stride'],
                                    labels=len(rows), reusable_source_frames=len(available), new_source_frames=len(new_indices),
                                    source_spotcheck_frames=checked, source_spotcheck_exact=True, all_jpegs_decoded=True,
                                    cache_hashes=hashes, samples=updated, seconds=time.perf_counter()-started))
        save()
        print(json.dumps(dict(clip=clip, completed_clips=len(report['clips']), reused=len(available), new=len(new_indices))), flush=True)
    updated_by_key={(s['clip'],s['frame']):s for c in report['clips'] for s in c['samples']}
    updated=[updated_by_key[s['clip'],s['frame']] for s in samples]
    assert len(updated)==2999 and len(report['clips'])==60
    sample_path=out/'training_samples.json'
    sample_path.write_text(json.dumps(updated,indent=2,allow_nan=False),encoding='utf-8')
    report.update(complete=True,samples_sha256=digest(sample_path),summary=dict(
        samples=len(updated),train_labels=sum(s['split']=='train' for s in updated),
        selection_labels=sum(s['split']=='selection' for s in updated),
        changed_train_labels=sum(s['stride']>1 and s['split']=='train' for s in updated),
        changed_selection_labels=sum(s['stride']>1 and s['split']=='selection' for s in updated),
        reusable_source_frames=sum(c['reusable_source_frames'] for c in report['clips']),
        new_source_frames=sum(c['new_source_frames'] for c in report['clips']),
        source_spotchecks=sum(len(c['source_spotcheck_frames'])*5 for c in report['clips']),
        seconds=sum(c['seconds'] for c in report['clips'])))
    save()
    print(json.dumps(dict(complete=True,**report['summary'])),flush=True)


if __name__=='__main__':
    main()
