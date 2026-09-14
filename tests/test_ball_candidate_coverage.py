import copy

import pytest

from scripts.evaluate.audit_ball_candidate_coverage import analyze_report


def fixture_report():
    return {'clips': [{'clip': 'test', 'size': [512, 288], 'fps': 30,
                      'predictions': [[{'x': 90, 'y': 80, 'confidence': .7},
                                       {'x': 20, 'y': 30, 'confidence': .9}],
                                      [{'x': 90, 'y': 80, 'confidence': .8}], []],
                      'raw_labeled_rows': [
                          {'clip': 'test', 'frame': i, 'width': 512, 'height': 288,
                           'target_xy': target, 'prediction_xy': None}
                          for i, target in enumerate(([20, 30], None, [40, 50]))]}]}


def test_oracle_coverage_separates_ranking_missing_detection_and_absence():
    result = analyze_report(fixture_report())
    assert result['counts']['correct_lower_rank'] == 1
    assert result['counts']['no_correct_raw_candidate'] == 1
    assert result['counts']['absent_with_filtered_candidates'] == 1
    assert result['filtered_candidate_oracle_recall_ceiling'] == .5
    assert result['filtered_mass_rank_metrics']['true_positives'] == 0
    assert result['filtered_peak_confidence_rank_metrics']['true_positives'] == 1
    assert result['filtered_peak_confidence_rank_metrics']['absent_false_detections'] == 1


def test_duplicate_labels_are_not_counted_as_more_evidence():
    report = fixture_report()
    report['clips'][0]['raw_labeled_rows'].append(copy.deepcopy(report['clips'][0]['raw_labeled_rows'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        analyze_report(report)


def test_stationary_filter_removal_is_not_misreported_as_detector_miss():
    clip = fixture_report()['clips'][0]
    clip['predictions'] = [[{'x': 20, 'y': 30, 'confidence': .9}] for _ in range(12)]
    clip['raw_labeled_rows'] = [{**clip['raw_labeled_rows'][0], 'frame': 11}]
    result = analyze_report({'clips': [clip]})
    assert result['counts']['correct_candidate_removed_by_filter'] == 1
    assert result['raw_candidate_oracle_recall_ceiling'] == 1
    assert result['filtered_candidate_oracle_recall_ceiling'] == 0
