from copy import deepcopy

import pytest

from scripts.train.verifier_negative_review import reviewed_negative_indices


def fixture():
    candidate = {'sample_index': 4, 'x': 2., 'y': 3., 'confidence': .6, 'rank': 0, 'training_target': 0}
    manifest = {'split': 'TRAINING_ONLY', 'selected_clips': [[f'match{i}', '000'] for i in range(20)],
                'rows': [{'clip': 'match0_000', 'frame': 9, 'target_xy': None, 'candidates': [candidate]}]}
    review = {'complete': True, 'labels_changed': False,
              'scope': 'TRAINING_ONLY_NEGATIVE_UNCERTAINTY_QUARANTINE_NO_RELABELING',
              'reviews': [{'clip': 'match0_000', 'frame': 9, 'candidate': deepcopy(candidate),
                           'judgment': 'unresolved', 'exclude_from_training_loss': True}]}
    return manifest, review


def test_quarantine_preserves_cached_labels_and_retains_clear_distractors():
    manifest, review = fixture()
    before = deepcopy(manifest)
    assert reviewed_negative_indices(manifest, review) == {4}
    assert manifest == before
    review['reviews'][0].update(judgment='clear_non_ball_distractor', exclude_from_training_loss=False)
    assert reviewed_negative_indices(manifest, review) == set()


@pytest.mark.parametrize('bad_clip', ['match16_000', 'match19_000', 'external_validation'])
def test_internal_selection_and_external_clips_cannot_enter_review(bad_clip):
    manifest, review = fixture()
    manifest['rows'][0]['clip'] = review['reviews'][0]['clip'] = bad_clip
    with pytest.raises(ValueError, match='actual training'):
        reviewed_negative_indices(manifest, review)


def test_visible_or_positive_examples_cannot_be_quarantined():
    for change in ('visible', 'positive'):
        manifest, review = fixture()
        if change == 'visible':
            manifest['rows'][0]['target_xy'] = [2., 3.]
        else:
            manifest['rows'][0]['candidates'][0]['training_target'] = 1
            review['reviews'][0]['candidate']['training_target'] = 1
        with pytest.raises(ValueError):
            reviewed_negative_indices(manifest, review)


def test_changed_identity_duplicate_review_and_policy_mismatch_rejected():
    for change in ('identity', 'duplicate', 'policy'):
        manifest, review = fixture()
        if change == 'identity':
            review['reviews'][0]['candidate']['x'] += 1
        elif change == 'duplicate':
            review['reviews'].append(deepcopy(review['reviews'][0]))
        else:
            review['reviews'][0]['exclude_from_training_loss'] = False
        with pytest.raises(ValueError):
            reviewed_negative_indices(manifest, review)
