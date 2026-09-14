"""Validate a saved prefix before continuing a temporal-spacing benchmark."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def resume_prefix(path, expected, baseline, sparse, root, driver_snapshot):
    path, root, driver_snapshot = Path(path), Path(root), Path(driver_snapshot)
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('complete') is not False:
        raise ValueError('Resume requires an incomplete source report')
    for key in ('scope', 'baseline_sha256', 'sparse_sha256', 'dataset_manifest_sha256',
                'checkpoint_sha256', 'configuration'):
        if saved[key] != expected[key]:
            raise ValueError(f'Resume provenance mismatch: {key}')
    driver = 'scripts/evaluate/benchmark_ball_spacing_stream.py'
    for name, checksum in saved['code_hashes'].items():
        source = driver_snapshot if name == driver else root / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Resume source mismatch: {name}')
    if set(saved['code_hashes']) != set(expected['code_hashes']) - {'scripts/evaluate/spacing_resume.py'}:
        raise ValueError('Unexpected inherited source set')
    clips = saved['clips']
    if len(clips) > len(baseline['clips']):
        raise ValueError('Too many inherited clips')
    for clip, old, pilot in zip(clips, baseline['clips'], sparse['clips']):
        for key in ('clip', 'fps', 'frames', 'size'):
            if clip[key] != old[key]:
                raise ValueError(f'Resume clip mismatch: {key}')
        if clip['clip'] != pilot['clip'] or clip['stride'] != pilot['stride']:
            raise ValueError('Resume sparse order or stride mismatch')
        for stage in ('raw', 'stationary', 'pixel_motion'):
            predictions = clip['predictions'][stage]
            rows = clip['labeled_rows'][stage]
            if len(predictions) != clip['frames'] or len(rows) != len(pilot['labeled_rows']):
                raise ValueError('Incomplete inherited predictions')
            for row, label in zip(rows, pilot['labeled_rows'], strict=True):
                for key in ('clip', 'frame', 'width', 'height', 'target_xy'):
                    if row[key] != label[key]:
                        raise ValueError('Inherited label mismatch')
                if row['prediction_xy'] != predictions[row['frame']]:
                    raise ValueError('Inherited row/stream mismatch')
                if stage == 'raw' and row['prediction_xy'] != label['prediction_xy']:
                    raise ValueError('Inherited sparse output is not exact')
        if clip['sparse_raw_max_difference_source_px'] != 0:
            raise ValueError('Require exact inherited sparse agreement')
    audit = {'source_report': str(path.resolve()),
             'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'inherited_code_hashes': saved['code_hashes'],
             'original_driver_snapshot': str(driver_snapshot.resolve()),
             'inherited_clips': [clip['clip'] for clip in clips],
             'validation': 'Exact provenance, ordered complete prefix, labels and sparse coordinates'}
    return deepcopy(clips), audit
