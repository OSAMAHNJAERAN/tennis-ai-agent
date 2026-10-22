"""Synthetic evaluator checks; these fixtures are not independent accuracy evidence."""
import copy
import json

import pytest

from src.prediction.evaluation import evaluate_file, evaluate_manifest, validate_manifest


def manifest():
    matches = []
    for index, (split, winner) in enumerate([('calibration', 1), ('calibration', 2), ('test', 1), ('test', 2)]):
        matches.append({'match_id': f'synthetic-{index}', 'source_group_id': f'synthetic-group-{index}',
            'split': split, 'match_date': f'2026-0{1 if split == "calibration" else 2}-{index+1:02}',
            'initial_state': {'format': 'BEST_OF_3', 'server_id': 1,
                              'sets_won_p1': 1 if winner == 1 else 0,
                              'sets_won_p2': 1 if winner == 2 else 0,
                              'current_set': 2, 'games_p1': 5 if winner == 1 else 0,
                              'games_p2': 5 if winner == 2 else 0},
            'ground_truth': {'status': 'independently_reviewed', 'winner_id': winner,
                             'source': 'synthetic_rules_fixture_only'},
            'points': [{'point_id': f'p{p}', 'timestamp_seconds': p+1, 'winner': winner,
                        'status': 'confirmed', 'provenance': 'manual'} for p in range(4)]})
    return {'schema_version': '1.0', 'usage_rights': 'authorized_for_evaluation',
            'data_kind': 'synthetic_software_test', 'matches': matches}


def test_chronological_evaluation_match_weighting_baseline_bootstrap_and_no_final_prediction():
    report = evaluate_manifest(manifest(), bootstrap_samples=30, bins=5)
    assert report['calibration_status'] == 'not_fitted_or_qualified'
    assert report['qualification_status'].startswith('unqualified')
    test = report['splits']['test']
    assert test['match_count'] == 2 and test['snapshot_count'] == 6
    assert test['match_bootstrap_95_percent_intervals']['statistical']['reliability']
    assert 0 <= test['metrics']['statistical']['brier'] <= 1
    assert test['metrics']['statistical']['log_loss'] >= 0
    assert report['preliminary_test_gates']['minimum_test_matches'] is False
    assert all(len(row['snapshots']) == 3 for row in report['matches'])
    assert all(row['snapshots'][-1]['point_id'] == 'p2' for row in report['matches'])
    assert report == evaluate_manifest(manifest(), bootstrap_samples=30, bins=5)


@pytest.mark.parametrize('mutation,reason', [
    (lambda m: m.update(usage_rights='unknown'), 'usage rights'),
    (lambda m: m['matches'][2].update(match_id='synthetic-0'), 'unique IDs'),
    (lambda m: m['matches'][2].update(source_group_id='synthetic-group-0'), 'crosses'),
    (lambda m: m['matches'][2].update(match_date='2026-01-01'), 'Chronological'),
    (lambda m: m['matches'][0]['ground_truth'].update(status='model_output'), 'independently reviewed'),
    (lambda m: m['matches'][0]['points'][1].update(point_id='p0'), 'unique'),
    (lambda m: m['matches'][0]['points'][1].update(timestamp_seconds=0), 'chronological'),
    (lambda m: m['matches'][0]['points'][1].update(provenance='automatic'), 'unreviewed'),
])
def test_manifest_rejects_leakage_and_unreviewed_ground_truth(mutation, reason):
    data = manifest()
    mutation(data)
    with pytest.raises(ValueError, match=reason):
        validate_manifest(data)


def test_evaluator_rejects_future_inputs_and_mismatch_with_actual_result():
    data = manifest()
    data['matches'][0]['initial_state']['winner_id'] = 1
    with pytest.raises(ValueError, match='future outcomes'):
        evaluate_manifest(data, bootstrap_samples=0)
    data = manifest()
    data['matches'][0]['ground_truth']['winner_id'] = 2
    with pytest.raises(ValueError, match='full match result'):
        evaluate_manifest(data, bootstrap_samples=0)


def test_file_report_binds_input_and_implementation_without_qualification(tmp_path):
    source, output = tmp_path/'synthetic.json', tmp_path/'report.json'
    source.write_text(json.dumps(manifest()))
    report = evaluate_file(source, output, bootstrap_samples=0)
    assert len(report['manifest_sha256']) == 64 and len(report['implementation_sha256']) == 3
    assert json.loads(output.read_text()) == report
    assert report['splits']['test']['match_bootstrap_95_percent_intervals'] is None
