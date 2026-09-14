from pathlib import Path

import numpy as np
import pytest

from src.evaluation.tracking_metrics import box_iou_matrix, detection_gap_diagnostics, evaluate_sequence, prepare_sequence

SOURCE = Path(__file__).resolve().parents[1] / 'artifacts/research/TrackEval'


def track(identity, box=(0, 0, 10, 10)):
    return {'id': identity, 'box': list(box)}


def test_iou_handles_empty_and_known_overlap():
    assert box_iou_matrix([], [[0, 0, 10, 10]]).shape == (0, 1)
    values = box_iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(values, [[1 / 3, 0]])


@pytest.mark.parametrize('problem', ['length', 'duplicate', 'box', 'identity'])
def test_invalid_tracking_sequences_are_rejected(problem):
    gt, pred = [[track(1)]], [[track(2)]]
    if problem == 'length':
        pred.append([])
    elif problem == 'duplicate':
        pred[0].append(track(2))
    elif problem == 'box':
        pred[0][0]['box'][2] = 0
    else:
        pred[0][0]['id'] = 1.5
    with pytest.raises(ValueError):
        prepare_sequence(gt, pred)


@pytest.mark.skipif(not SOURCE.exists(), reason='Official TrackEval research source not acquired')
def test_official_metrics_distinguish_perfect_identity_from_switch():
    gt = [[track(100)] for _ in range(4)]
    perfect = [[track(500)] for _ in range(4)]
    alias_before = 'int' in np.__dict__
    result = evaluate_sequence(gt, perfect, SOURCE)['summary']
    assert result['HOTA'] == pytest.approx(1)
    assert result['IDF1'] == pytest.approx(1)
    assert result['MOTA'] == pytest.approx(1)
    assert result['IDSW'] == 0
    switched = [[track(500)], [track(500)], [track(600)], [track(600)]]
    result = evaluate_sequence(gt, switched, SOURCE)['summary']
    assert result['IDF1'] == pytest.approx(.5)
    assert result['IDSW'] == 1
    assert result['MOTA'] == pytest.approx(.75)
    assert result['HOTA'] == pytest.approx(np.sqrt(.5))
    assert ('int' in np.__dict__) == alias_before


@pytest.mark.skipif(not SOURCE.exists(), reason='Official TrackEval research source not acquired')
def test_missing_interval_and_false_track_are_counted():
    gt = [[track(1)] for _ in range(4)]
    pred = [[track(2)], [], [], [track(2), track(3, (20, 20, 30, 30))]]
    report = evaluate_sequence(gt, pred, SOURCE)
    result = report['summary']
    assert report['detection_gaps']['longest_gap_frames'] == 2
    assert report['detection_gaps']['recovered_gaps'] == 1
    assert report['detection_gaps']['missed_annotated_frames'] == 2
    assert result['CLR_TP'] == 2
    assert result['CLR_FN'] == 2
    assert result['CLR_FP'] == 1
    # Preserve and expose the pinned upstream empty-frame Frag limitation.
    assert result['Frag'] == 0
    assert result['IDSW'] == 0
    assert result['IDF1'] == pytest.approx(4 / 7)


def test_missing_gt_interval_is_not_fabricated_as_a_tracking_gap():
    gt = [[track(1)], [], [], [track(1)]]
    pred = [[track(2)], [], [], [track(2)]]
    result = detection_gap_diagnostics(gt, pred)
    assert result['missed_annotated_frames'] == 0
    assert result['gaps'] == []


def test_initial_and_terminal_misses_are_not_claimed_as_recovered_gaps():
    gt = [[track(1)] for _ in range(5)]
    pred = [[], [], [track(2)], [], []]
    result = detection_gap_diagnostics(gt, pred)
    assert result['missed_annotated_frames'] == 4
    assert result['longest_gap_frames'] == 2
    assert result['recovered_gaps'] == 0
