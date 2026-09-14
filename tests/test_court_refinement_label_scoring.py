import pytest

from scripts.evaluate.benchmark_court_refinement_labels import score


def test_identity_threshold_missing_and_ineligible_labels():
    result=score([[17,10],[27.01,20],None,[0,0]],[[10,10],[20,20],[30,30],[-1,10]])
    assert (result['tp'],result['fp'],result['fn'])==(1,1,2)
    assert result['landmarks'][0]['correct']
    assert not result['landmarks'][3]['eligible']


def test_swapped_landmark_identities_are_not_matches():
    result=score([[100,100],[10,10]],[[10,10],[100,100]])
    assert (result['tp'],result['fp'],result['fn'])==(0,2,2)


def test_mismatched_prediction_and_label_count_fails():
    with pytest.raises(ValueError):score([[10,10]],[[10,10],[20,20]])
