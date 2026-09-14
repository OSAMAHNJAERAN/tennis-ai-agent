import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate.spacing_resume import resume_prefix


@pytest.fixture
def resume_case(tmp_path):
    driver = 'scripts/evaluate/benchmark_ball_spacing_stream.py'
    snapshot = tmp_path / 'original.py'
    snapshot.write_text('original driver', encoding='utf-8')
    row = dict(clip='match1', frame=0, width=10, height=10,
               target_xy=[2, 3], prediction_xy=[2, 3])
    clip = dict(clip='match1', fps=60, frames=1, size=[10, 10], stride=2,
                sparse_raw_max_difference_source_px=0,
                predictions={s: [[2, 3]] for s in ('raw', 'stationary', 'pixel_motion')},
                labeled_rows={s: [dict(row)] for s in ('raw', 'stationary', 'pixel_motion')})
    saved = dict(complete=False, scope='test', baseline_sha256='a', sparse_sha256='b',
                 dataset_manifest_sha256='c', checkpoint_sha256='d', configuration={},
                 code_hashes={driver: hashlib.sha256(snapshot.read_bytes()).hexdigest()}, clips=[clip])
    expected = {**saved, 'code_hashes': {driver: 'new', 'scripts/evaluate/spacing_resume.py': 'helper'}}
    baseline = {'clips': [clip]}
    sparse = {'clips': [{'clip': 'match1', 'stride': 2, 'labeled_rows': [row]}]}
    path = tmp_path / 'partial.json'
    return saved, path, (expected, baseline, sparse, tmp_path, snapshot)


def test_valid_prefix_is_copied_with_provenance(resume_case):
    saved, path, args = resume_case
    path.write_text(json.dumps(saved), encoding='utf-8')
    clips, audit = resume_prefix(path, *args)
    assert clips == saved['clips']
    assert audit['inherited_clips'] == ['match1']
    assert audit['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('corruption', ['configuration', 'order', 'truncation', 'sparse', 'source', 'complete'])
def test_invalid_prefix_is_rejected(resume_case, corruption):
    saved, path, args = resume_case
    if corruption == 'configuration': saved['configuration'] = {'changed': True}
    if corruption == 'order': saved['clips'][0] = {**saved['clips'][0], 'clip': 'wrong'}
    if corruption == 'truncation': saved['clips'][0]['predictions']['raw'] = []
    if corruption == 'sparse':
        saved['clips'][0]['predictions']['raw'] = [[4, 5]]
        saved['clips'][0]['labeled_rows']['raw'][0]['prediction_xy'] = [4, 5]
    if corruption == 'source': args[-1].write_text('modified', encoding='utf-8')
    if corruption == 'complete': saved['complete'] = True
    path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError): resume_prefix(path, *args)
