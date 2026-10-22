"""Causal identity/state software fixtures; no image-identity accuracy claims."""
import json

import pytest

from src.tracking.match_identity import IdentityAnchor, IdentityObservation, IdentitySettings, MatchIdentity

RED, BLUE = (1., 0., 0.), (0., 0., 1.)


def person(track, position=(3., 20.), colour=RED, **kwargs):
    return IdentityObservation(track, position, colour, participant=kwargs.pop('participant', True),
                               ground_point_usable=kwargs.pop('ground_point_usable', True), **kwargs)


def anchor(layer, player='A', track=1, timestamp=0., segment='rally-1', position=(3., 20.), colour=RED, command=None):
    return layer.correct(IdentityAnchor(command or f'anchor-{player}-{track}-{timestamp}', player, track,
                                        timestamp, segment, 'Reviewed video evidence', position, colour))


def confirmed(layer=None):
    layer = layer or MatchIdentity('analysis-one')
    anchor(layer)
    anchor(layer, 'B', 2, position=(7., 4.), colour=BLUE)
    layer.update([person(1), person(2, (7., 4.), BLUE)], 0., 'rally-1')
    return layer


def rows(result):
    return {row['track_id']: row for row in result['observations']}


def test_initial_near_far_positions_never_assign_match_names():
    layer = MatchIdentity('new-upload')
    result = layer.update([person(1), person(2, (7., 4.), BLUE)], 0., 'rally-1')
    assert all(row['player_id'] is None and row['state'] == 'UNKNOWN' for row in result['observations'])
    assert all(not row['motion_continuity'] for row in result['observations'])


def test_reviewed_anchors_local_continuity_and_independent_streams():
    first = confirmed(MatchIdentity('first-upload'))
    second = confirmed(MatchIdentity('second-upload'))
    a = first.update([person(1, (3.1, 20.))], .1, 'rally-1')
    b = second.update([person(1, (3.1, 20.))], .1, 'rally-1')
    assert a['observations'][0]['player_id'] == b['observations'][0]['player_id'] == 'A'
    assert a['observations'][0]['motion_continuity']
    assert a['observations'][0]['event_id'] != b['observations'][0]['event_id']
    assert a['observations'][0]['support']['calibrated_probability'] is None
    anchor(first, 'B', 1, .1)
    assert second.update([person(1, (3.2, 20.))], .2, 'rally-1')['observations'][0]['player_id'] == 'A'


def test_end_change_recycled_track_ids_need_new_anchors():
    layer = confirmed()
    result = layer.update([person(1, (3., 20.), BLUE), person(2, (7., 4.), RED)], 10., 'rally-after-end-change')
    assert result['local_reset'] and result['epoch'] == 1
    observations = rows(result)
    assert observations[1]['player_id'] is None and observations[1]['state'] == 'PROVISIONAL'
    assert observations[1]['hypotheses'][0]['player_id'] == 'B'
    assert observations[2]['hypotheses'][0]['player_id'] == 'A'
    assert not any(row['motion_continuity'] for row in result['observations'])
    anchor(layer, 'B', 1, 10., 'rally-after-end-change', (3., 20.), BLUE)
    anchor(layer, 'A', 2, 10., 'rally-after-end-change', (7., 4.), RED)
    corrected = rows(layer.update([person(1, (3., 20.), BLUE), person(2, (7., 4.), RED)], 10., 'rally-after-end-change'))
    assert corrected[1]['player_id'] == 'B' and corrected[2]['player_id'] == 'A'
    assert not corrected[1]['motion_continuity'] and not corrected[2]['motion_continuity']


def test_cut_same_local_id_does_not_inherit_identity_even_same_colour():
    layer = confirmed()
    row = layer.update([person(1)], .1, 'new-camera')['observations'][0]
    assert row['track_key'] == '1:1' and row['state'] == 'PROVISIONAL' and row['player_id'] is None
    assert row['hypotheses'][0]['evidence'][0]['signal'] == 'colour'
    assert not row['motion_continuity']


def test_time_gap_disallows_bridge_and_reconnection_remains_provisional():
    layer = confirmed()
    result = layer.update([person(1)], 1.1, 'rally-1')
    assert result['reset_reason'] == 'source_timestamp_gap'
    assert result['observations'][0]['player_id'] is None
    assert not result['observations'][0]['motion_continuity']


def test_missing_ground_interval_cannot_bridge_motion():
    layer = confirmed()
    missing = layer.update([person(1, position=None, ground_point_usable=False)], .1, 'rally-1')
    observed = layer.update([person(1, (3.2, 20.))], .2, 'rally-1')
    assert missing['observations'][0]['player_id'] == observed['observations'][0]['player_id'] == 'A'
    assert not missing['observations'][0]['motion_continuity']
    assert not observed['observations'][0]['motion_continuity']


def test_unseen_frame_breaks_motion_but_can_retain_short_local_identity():
    layer = confirmed()
    layer.update([], .1, 'rally-1')
    observed = layer.update([person(1, (3.2, 20.))], .2, 'rally-1')
    assert observed['observations'][0]['player_id'] == 'A'
    assert not observed['observations'][0]['motion_continuity']


def test_old_epoch_court_geometry_cannot_support_reconnection_later():
    layer = confirmed()
    layer.update([person(9, colour=None)], .1, 'new-camera')
    result = layer.update([person(9, colour=None)], .2, 'new-camera')
    assert result['observations'][0]['state'] == 'UNKNOWN'
    assert result['observations'][0]['hypotheses'] == []


def test_overlap_revokes_binding_and_recovery_abstains_until_review():
    layer = confirmed()
    mixed = layer.update([person(1, overlap=True), person(2, (7., 4.), BLUE, overlap=True)], .1, 'rally-1')
    assert all(row['state'] == 'AMBIGUOUS' and row['player_id'] is None for row in mixed['observations'])
    recovered = layer.update([person(1), person(2, (7., 4.), BLUE)], .2, 'rally-1')
    assert all(row['player_id'] is None for row in recovered['observations'])


def test_identical_colour_ambiguous_candidates_and_officials_are_excluded():
    layer = MatchIdentity('same-clothes')
    anchor(layer)
    anchor(layer, 'B', 2, position=(7., 4.), colour=RED)
    layer.update([person(1), person(2, (7., 4.), RED)], 0., 'rally-1')
    result = layer.update([person(9), person(10, participant=False)], 3., 'new-camera')
    observations = rows(result)
    assert observations[9]['state'] == 'AMBIGUOUS' and observations[9]['player_id'] is None
    assert len(observations[9]['hypotheses']) == 2
    assert observations[10]['state'] == 'EXCLUDED' and observations[10]['hypotheses'] == []


def test_departure_recycled_id_while_other_player_continues_is_unknown():
    layer = confirmed()
    layer.update([person(1)], .6, 'rally-1')
    result = layer.update([person(1), person(2, (7., 4.), BLUE)], 1.2, 'rally-1')
    observations = rows(result)
    assert not result['local_reset']
    assert observations[1]['player_id'] == 'A'
    assert observations[2]['player_id'] is None and observations[2]['state'] == 'PROVISIONAL'


def test_implausible_ground_or_colour_jump_abstains_without_swapping():
    layer = confirmed()
    result = rows(layer.update([person(1, (3., 2.), BLUE), person(2, (7., 4.), BLUE)], .1, 'rally-1'))
    assert result[1]['state'] == 'AMBIGUOUS' and result[1]['player_id'] is None
    assert result[2]['player_id'] == 'B'
    assert any(not evidence['compatible'] for evidence in result[1]['support']['evidence'])


def test_correction_is_idempotent_revised_and_does_not_infer_other_swap():
    layer = confirmed()
    event = anchor(layer, 'A', 2, 0., position=(7., 4.), colour=BLUE, command='explicit-correction')
    assert event['replaced_player_id'] == 'B' and event['revision'] == 3
    assert anchor(layer, 'A', 2, 0., position=(7., 4.), colour=BLUE, command='explicit-correction') == event
    with pytest.raises(ValueError, match='different correction'):
        anchor(layer, 'B', 2, 0., position=(7., 4.), colour=BLUE, command='explicit-correction')
    result = rows(layer.update([person(1), person(2, (7., 4.), BLUE)], 0., 'rally-1'))
    assert result[1]['player_id'] is None and result[2]['player_id'] == 'A'


def test_checkpoint_resume_and_overlap_replay_preserve_complete_records():
    layer = confirmed()
    first = layer.update([person(1, (3.1, 20.)), person(2, (7., 4.), BLUE)], .1, 'rally-1')
    state = json.loads(json.dumps(layer.checkpoint()))
    resumed = MatchIdentity.restore(state, 'analysis-one')
    assert resumed.update([person(1, (3.1, 20.)), person(2, (7., 4.), BLUE)], .1, 'rally-1') == first
    observations = [person(1, (3.2, 20.)), person(2, (7., 4.1), BLUE)]
    assert resumed.update(observations, .2, 'rally-1') == layer.update(observations, .2, 'rally-1')
    assert resumed.correct(IdentityAnchor('anchor-A-1-0.0', 'A', 1, 0., 'rally-1', 'Reviewed video evidence', (3., 20.), RED))['revision'] == 1
    with pytest.raises(ValueError, match='another analysis'):
        MatchIdentity.restore(state, 'another-stream')


def test_event_ids_stable_under_correction_and_unknowns_are_unique():
    layer = MatchIdentity('event-identity')
    observations = [person(1), person(None), person(None)]
    before = layer.update(observations, 0., 'rally-1')
    anchor(layer)
    after = layer.update(observations, 0., 'rally-1')
    assert len({item['event_id'] for item in after['observations']}) == 3
    assert before['observations'][0]['event_id'] == after['observations'][0]['event_id']
    assert before['observations'][0]['player_id'] is None and after['observations'][0]['player_id'] == 'A'


def test_causal_ordering_duplicate_evidence_and_historical_correction_require_replay():
    layer = confirmed()
    layer.update([person(1)], .1, 'rally-1')
    with pytest.raises(ValueError, match='backwards'):
        layer.update([person(1)], .05, 'rally-1')
    with pytest.raises(ValueError, match='same source time'):
        layer.update([person(1, (4., 20.))], .1, 'rally-1')
    with pytest.raises(ValueError, match='current source timestamp'):
        anchor(layer, timestamp=.05)
    with pytest.raises(ValueError, match='new continuity segment'):
        anchor(layer, timestamp=.1, segment='unseen-camera')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., True])
def test_invalid_source_times_never_mutate_state(bad):
    layer = confirmed()
    state = layer.checkpoint()
    with pytest.raises(ValueError):
        layer.update([person(1)], bad, 'rally-1')
    assert layer.checkpoint() == state


def test_invalid_duplicate_tracks_and_bounded_inputs():
    layer = MatchIdentity('bounded', IdentitySettings(max_observations=2))
    with pytest.raises(ValueError, match='unique'):
        layer.update([person(1), person(1)], 0., 'rally-1')
    with pytest.raises(ValueError, match='bounded'):
        layer.update([person(1), person(2), person(3)], 0., 'rally-1')
    with pytest.raises(ValueError):
        IdentityObservation(1, colour_rgb=(255., 0., 0.))
    with pytest.raises(ValueError):
        IdentityObservation(1, participant=0)


def test_generator_consumption_and_checkpoint_size_do_not_grow_with_frames(tmp_path):
    layer = MatchIdentity('bounded-generators', IdentitySettings(max_observations=2))
    consumed = []
    def people():
        for index in range(10000):
            consumed.append(index)
            yield person(index)
    with pytest.raises(ValueError, match='bounded'):
        layer.update(people(), 0., 'rally-1')
    assert consumed == [0, 1, 2]
    layer = confirmed()
    for index in range(1, 1001):
        layer.update([person(1), person(2, (7., 4.), BLUE)], index / 60., 'rally-1')
    path = tmp_path / 'identity-checkpoint.json'
    path.write_text(json.dumps(layer.checkpoint()))
    restored = MatchIdentity.restore(json.loads(path.read_text()), 'analysis-one')
    assert len(restored.checkpoint()['bindings']) == 2
    assert path.stat().st_size < 10000
    assert restored.update([person(1)], 1001 / 60., 'rally-1') == layer.update([person(1)], 1001 / 60., 'rally-1')


@pytest.mark.parametrize('mutation', ['epoch_boolean', 'noncanonical_track', 'future_seen', 'old_binding_epoch',
                                      'conflicting_players', 'wrong_command_provenance', 'malformed'])
def test_invalid_checkpoint_is_rejected(mutation):
    layer = confirmed()
    layer.update([person(1), person(2, (7., 4.), BLUE)], .1, 'rally-1')
    state = layer.checkpoint()
    if mutation == 'epoch_boolean':
        state['epoch'] = True
    elif mutation == 'noncanonical_track':
        state['bindings']['01'] = state['bindings'].pop('1')
    elif mutation == 'future_seen':
        state['bindings']['1']['last_seen'] = 10.
    elif mutation == 'old_binding_epoch':
        state['epoch'] = 1
    elif mutation == 'conflicting_players':
        state['bindings']['2']['player_id'] = 'A'
    elif mutation == 'wrong_command_provenance':
        state['commands']['anchor-A-1-0.0']['event']['analysis_id'] = 'other'
    else:
        del state['settings']
    with pytest.raises(ValueError):
        MatchIdentity.restore(state, 'analysis-one')
