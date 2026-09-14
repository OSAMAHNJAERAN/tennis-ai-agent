from copy import deepcopy

import pytest

from scripts.evaluate.compare_tiled_checkpoints import compare_reports


def report():
    row = {'clip': 'match1_000', 'frame': 0, 'width': 512, 'height': 288,
           'target_xy': [20, 30], 'prediction_xy': [20, 30]}
    stages = ('raw', 'stationary', 'pixel_motion')
    return {'complete': True, 'dataset_manifest_sha256': 'data', 'checkpoint_sha256': 'weights',
            'configuration': {'threshold': .2},
            'code_hashes': {'model': 'same', 'filter': 'same', 'metric': 'same'},
            'clips': [{'clip': 'match1_000', 'frames': 1, 'fps': 30, 'size': [512, 288],
                       'predictions': {stage: [[20, 30]] for stage in stages},
                       'labeled_rows': {stage: [deepcopy(row)] for stage in stages}}]}


def test_checkpoint_comparison_reports_real_recall_loss():
    first, second = report(), report()
    second['checkpoint_sha256'] = 'new-weights'
    second['clips'][0]['predictions']['pixel_motion'][0] = None
    second['clips'][0]['labeled_rows']['pixel_motion'][0]['prediction_xy'] = None
    result = compare_reports(first, second)
    assert result['results']['pixel_motion']['paired']['lost_correct_visible_frames'] == 1
    assert result['results']['pixel_motion']['candidate']['pooled']['false_negatives'] == 1
    assert result['compared_frames'] == 1


@pytest.mark.parametrize('changed', ['configuration', 'code', 'frames', 'labels', 'complete'])
def test_checkpoint_comparison_rejects_unpaired_evidence(changed):
    first, second = report(), report()
    if changed == 'configuration':
        second['configuration']['threshold'] = .1
    elif changed == 'code':
        second['code_hashes']['filter'] = 'new'
    elif changed == 'frames':
        second['clips'][0]['predictions']['raw'] = []
    elif changed == 'labels':
        second['clips'][0]['labeled_rows']['raw'][0]['target_xy'] = [100, 100]
    else:
        second['complete'] = False
    with pytest.raises(ValueError):
        compare_reports(first, second)
