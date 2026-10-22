"""Exact raw-state replay prerequisites; no inference or accuracy claims."""
import ast
import builtins
import copy
from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import pytest

from src.scoring.match_state import (
    BallPlayState, CompletedSetScore, MatchFormat, MatchState, PointState,
    ServerCourtEnd, ServiceSide, SetFormat,
)
from src.scoring.match_state_codec import decode_match_state, encode_match_state


def test_importing_codec_does_not_import_state_geometry_or_optional_runtime(monkeypatch):
    source = Path(__file__).parents[1] / 'src/scoring/match_state_codec.py'
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'src' or name.startswith('src.'):
            pytest.fail('Codec import eagerly loaded project classes: ' + name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    namespace = {'__name__': 'lazy_codec_import_test'}
    exec(compile(source.read_text(encoding='utf-8'), str(source), 'exec'), namespace)
    assert callable(namespace['encode_match_state'])
    assert callable(namespace['decode_match_state'])


def test_fixed_lazy_registry_uses_current_actual_class_definitions(monkeypatch):
    # Some pure scoring tests install the actual AST definitions in this module
    # slot to omit unrelated package imports. Resolve that fixed slot at use.
    import src.scoring.match_state as original
    source = Path(original.__file__)
    module = ModuleType('src.scoring.match_state')
    module.__dict__.update(vars(original))
    tree = ast.parse(source.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    monkeypatch.setitem(sys.modules, 'src.scoring.match_state', module)
    exec(compile(tree, str(source), 'exec'), module.__dict__)
    assert module.MatchState is not original.MatchState
    state = module.MatchState(points_p1=12, points_p2=11, tie_break_points_p1=17)
    restored = decode_match_state(encode_match_state(state))
    assert type(restored) is module.MatchState
    assert asdict(restored) == asdict(state)
    with pytest.raises(ValueError):
        encode_match_state(original.MatchState())


def raw_state():
    return MatchState(
        match_id='private-match-\u0631\u0627\u062c\u0639', format=MatchFormat.BEST_OF_5,
        set_format=SetFormat.ADVANTAGE_SET, player_1_id=17, player_2_id=23,
        server_id=23, receiver_id=17, initial_server_id=17, current_set=5,
        current_game=29, sets_won_p1=2, sets_won_p2=2, games_p1=14, games_p2=14,
        points_p1=1001, points_p2=1000, serve_attempt=2, service_side=ServiceSide.AD,
        tie_break_active=True, tie_break_points_p1=31, tie_break_points_p2=30,
        point_state=PointState.POINT_REVIEW_PENDING, ball_state=BallPlayState.DEAD,
        completed_sets=[CompletedSetScore(1, 7, 6, 14, 12, 17),
                        CompletedSetScore(2, 4, 6, None, None, 23)],
        match_complete=False, winner_id=None, last_point_winner=23,
        last_point_reason='Review: camera gap \u2192 outcome unknown', state_version=987,
        tie_break_points_to_win=10, deciding_set_tie_break_points_to_win=7,
        deciding_set_format=SetFormat.TIE_BREAK_SET,
        player_1_court_end=ServerCourtEnd.FAR_COURT,
        initial_state_confirmed=False, score_confirmed=False,
        score_source='human_review:frame-777', match_status='SUSPENDED',
    )


def document():
    return encode_match_state(raw_state())


def test_json_roundtrip_preserves_all_raw_fields_and_dual_counters(monkeypatch):
    state = raw_state()
    monkeypatch.setattr(MatchState, 'to_dict', lambda _: pytest.fail('presentation serializer used'))
    saved = encode_match_state(state)
    assert set(saved['fields']) == {field.name for field in fields(MatchState)}
    restored = decode_match_state(json.loads(json.dumps(saved, allow_nan=False)))
    assert type(restored) is MatchState
    assert asdict(restored) == asdict(state)
    assert (restored.points_p1, restored.points_p2) == (1001, 1000)
    assert (restored.tie_break_points_p1, restored.tie_break_points_p2) == (31, 30)
    assert type(restored.completed_sets[0]) is CompletedSetScore
    assert encode_match_state(restored) == saved


def test_default_state_and_all_nullable_fields_are_retained():
    state = MatchState()
    restored = decode_match_state(encode_match_state(state))
    assert asdict(restored) == asdict(state)
    assert restored.winner_id is restored.last_point_winner is None
    assert restored.deciding_set_format is restored.deciding_set_tie_break_points_to_win is None


@pytest.mark.parametrize('name,kind', [
    ('format', MatchFormat), ('set_format', SetFormat), ('service_side', ServiceSide),
    ('point_state', PointState), ('ball_state', BallPlayState),
    ('deciding_set_format', SetFormat), ('player_1_court_end', ServerCourtEnd),
])
def test_every_declared_enum_value_has_typed_roundtrip(name, kind):
    for value in kind:
        state = raw_state()
        setattr(state, name, value)
        saved = encode_match_state(state)
        assert saved['fields'][name] == {'enum': kind.__name__, 'value': value.value}
        assert getattr(decode_match_state(saved), name) is value


@pytest.mark.parametrize('status', ['IN_PROGRESS', 'PARTIAL', 'COMPLETED', 'RETIRED', 'SUSPENDED'])
def test_status_and_confirmation_do_not_invent_score_consistency_rules(status):
    state = raw_state()
    state.match_status = status
    state.match_complete = True
    state.winner_id = None
    assert asdict(decode_match_state(encode_match_state(state))) == asdict(state)


def test_encoding_and_decoding_detach_mutable_aliases():
    state = raw_state()
    saved = encode_match_state(state)
    restored = decode_match_state(saved)
    state.completed_sets[0].games_p1 = 100
    state.completed_sets.append(CompletedSetScore(3, 6, 0, winner_id=17))
    assert saved['fields']['completed_sets'][0]['fields']['games_p1'] == 7
    saved['fields']['completed_sets'][0]['fields']['games_p2'] = 101
    saved['fields']['completed_sets'].clear()
    assert restored.completed_sets[0].games_p2 == 6
    assert len(restored.completed_sets) == 2
    restored.completed_sets[1].games_p1 = 102
    assert state.completed_sets[1].games_p1 == 4


@pytest.mark.parametrize('field', [field.name for field in fields(MatchState)])
def test_missing_or_additional_state_fields_are_rejected(field):
    saved = document()
    saved['fields'].pop(field)
    with pytest.raises(ValueError):
        decode_match_state(saved)
    saved = document()
    saved['fields']['unrecognized_field'] = 0
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('field', [field.name for field in fields(CompletedSetScore)])
def test_completed_set_exact_fields_are_required(field):
    saved = document()
    saved['fields']['completed_sets'][0]['fields'].pop(field)
    with pytest.raises(ValueError):
        decode_match_state(saved)
    saved = document()
    saved['fields']['completed_sets'][0]['fields']['unexpected'] = None
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('value', [None, [], {}, 'MatchState', True])
def test_invalid_root_documents(value):
    with pytest.raises(ValueError):
        decode_match_state(value)


@pytest.mark.parametrize('key,value', [
    ('schema_version', True), ('schema_version', 1.0), ('schema_version', '1'),
    ('schema_version', 2), ('type', 'OtherState'), ('type', None),
    ('fields', []), ('fields', None),
])
def test_bad_root_tags_and_types(key, value):
    saved = document()
    saved[key] = value
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('change', [
    {'enum': 'SetFormat', 'value': 'BEST_OF_5'},
    {'enum': 'MatchFormat', 'value': 'UNKNOWN'},
    {'enum': 'MatchFormat', 'value': 5},
    {'enum': 'MatchFormat', 'value': 'BEST_OF_5', 'module': 'malicious'},
    {'value': 'BEST_OF_5'}, 'BEST_OF_5', None,
])
def test_wrong_enum_tags_and_unknown_values(change):
    saved = document()
    saved['fields']['format'] = change
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('name,value', [
    ('points_p1', True), ('points_p1', -1), ('points_p1', 2.0),
    ('tie_break_points_p2', math.inf), ('games_p1', math.nan),
    ('sets_won_p1', -math.inf), ('state_version', '5'),
    ('player_1_id', 0), ('current_set', 0), ('current_game', False),
    ('serve_attempt', True), ('serve_attempt', 0), ('serve_attempt', 3),
    ('tie_break_points_to_win', None), ('tie_break_points_to_win', 0),
    ('tie_break_points_to_win', 7.0), ('deciding_set_tie_break_points_to_win', False),
    ('deciding_set_tie_break_points_to_win', -1), ('tie_break_active', 1),
    ('match_complete', None), ('score_confirmed', 'true'),
    ('initial_state_confirmed', 0), ('match_id', ''), ('match_id', '  '),
    ('score_source', []), ('score_source', ''), ('match_status', None),
    ('match_status', ''), ('last_point_reason', 0), ('winner_id', True),
    ('winner_id', 1), ('last_point_winner', 99), ('server_id', 1),
    ('receiver_id', 1), ('initial_server_id', 1), ('completed_sets', ()),
])
def test_structurally_invalid_fields_rejected_by_both_directions(name, value):
    state = raw_state()
    setattr(state, name, value)
    with pytest.raises(ValueError):
        encode_match_state(state)
    saved = document()
    saved['fields'][name] = value
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('name', ['player_2_id', 'receiver_id'])
def test_duplicate_participant_and_server_receiver_rejected(name):
    state = raw_state()
    setattr(state, name, state.player_1_id if name == 'player_2_id' else state.server_id)
    with pytest.raises(ValueError):
        encode_match_state(state)


@pytest.mark.parametrize('target', [1, 7, 10, 11, 101])
def test_codec_preserves_positive_targets_without_setting_rule_configuration_policy(target):
    state = raw_state()
    state.tie_break_points_to_win = target
    state.deciding_set_tie_break_points_to_win = target
    restored = decode_match_state(encode_match_state(state))
    assert restored.tie_break_points_to_win == target
    assert restored.deciding_set_tie_break_points_to_win == target


@pytest.mark.parametrize('name,value', [
    ('set_number', False), ('set_number', 0), ('games_p1', -1),
    ('games_p2', math.inf), ('tie_break_p1', True), ('tie_break_p2', -1),
    ('winner_id', True), ('winner_id', None), ('winner_id', 999),
])
def test_invalid_completed_set_values_rejected_both_directions(name, value):
    state = raw_state()
    setattr(state.completed_sets[0], name, value)
    with pytest.raises(ValueError):
        encode_match_state(state)
    saved = document()
    saved['fields']['completed_sets'][0]['fields'][name] = value
    with pytest.raises(ValueError):
        decode_match_state(saved)


@pytest.mark.parametrize('value', [None, [], {}, {'type': 'Set', 'fields': {}},
                                  {'type': 'CompletedSetScore', 'fields': {}, 'extra': 1}])
def test_bad_completed_set_envelopes(value):
    saved = document()
    saved['fields']['completed_sets'][0] = value
    with pytest.raises(ValueError):
        decode_match_state(saved)


def test_encode_rejects_untyped_state_rows_and_enum_strings():
    with pytest.raises(ValueError):
        encode_match_state(document())
    state = raw_state()
    state.completed_sets = [asdict(state.completed_sets[0])]
    with pytest.raises(ValueError):
        encode_match_state(state)
    state = raw_state()
    state.service_side = 'AD'
    with pytest.raises(ValueError):
        encode_match_state(state)


def test_plain_containers_required_and_inputs_not_mutated():
    class DictSubclass(dict):
        pass
    class ListSubclass(list):
        pass
    saved = document()
    before = copy.deepcopy(saved)
    decode_match_state(saved)
    assert saved == before
    with pytest.raises(ValueError):
        decode_match_state(DictSubclass(saved))
    saved['fields']['completed_sets'] = ListSubclass(saved['fields']['completed_sets'])
    with pytest.raises(ValueError):
        decode_match_state(saved)
