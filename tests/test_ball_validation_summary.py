import pytest
import json

from scripts.evaluate.summarize_ball_validation import summarize, operating_point, digest


def test_duplicate_labels_rejected_instead_of_counting_twice():
    row = {'clip': 'a', 'frame': 1, 'width': 512, 'height': 288,
           'target_xy': None, 'prediction_xy': None}
    with pytest.raises(ValueError, match='Duplicate'):
        summarize([row, row])


def test_clip_macro_reveals_failure_hidden_by_unequal_label_counts():
    rows = [{'clip': 'a', 'frame': i, 'width': 512, 'height': 288,
             'target_xy': [5, 5], 'prediction_xy': [5, 5]} for i in range(20)]
    rows.append({'clip': 'b', 'frame': 0, 'width': 512, 'height': 288,
                 'target_xy': [5, 5], 'prediction_xy': [100, 100]})
    result = summarize(rows)
    assert result['pooled']['recall'] > .95
    assert result['macro_available_clip_mean']['recall'] == .5
    assert result['clips_passing_both_95_percent'] == 1


def test_nested_feature_replay_preserves_detector_and_feature_identity(tmp_path):
    model_path = tmp_path / 'model.json'
    model_path.write_text(json.dumps({'backend': 'wasb', 'wasb_heatmap_threshold': .2,
                                      'checkpoint_sha256': 'checkpoint'}))
    feature_path = tmp_path / 'features.json'
    feature_path.write_text(json.dumps({'source_report': str(model_path), 'source_sha256': digest(model_path),
                                        'feature_scope': 'CAUSAL_PIXEL_DIFFERENCE',
                                        'configuration': {'patch_radius': 3}, 'feature_code_sha256': 'code-version'}))
    replay = {'source_report': str(feature_path), 'source_sha256': digest(feature_path)}
    result = operating_point(replay, .25, 12.)
    assert result['wasb_heatmap_threshold'] == .2
    assert result['checkpoint_sha256'] == 'checkpoint'
    assert result['pixel_feature_configuration'] == {'patch_radius': 3}
    assert result['pixel_feature_code_sha256'] == 'code-version'
    model_path.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        operating_point(replay, .25, 12.)
