"""Lossless, fixed-schema JSON for raw scoring state used by replay.

The public ``MatchState.to_dict`` is a presentation contract, not a checkpoint:
it combines standard and tie-break counters. This codec retains every declared
field separately. It validates structure and supported options, not whether a
particular score could have resulted from a complete observed tennis match.
"""
from dataclasses import fields

_STATE_FIELDS = frozenset((
    'match_id', 'format', 'set_format', 'player_1_id', 'player_2_id', 'server_id',
    'receiver_id', 'initial_server_id', 'current_set', 'current_game',
    'sets_won_p1', 'sets_won_p2', 'games_p1', 'games_p2', 'points_p1', 'points_p2',
    'serve_attempt', 'service_side', 'tie_break_active', 'tie_break_points_p1',
    'tie_break_points_p2', 'point_state', 'ball_state', 'completed_sets',
    'match_complete', 'winner_id', 'last_point_winner', 'last_point_reason',
    'state_version', 'tie_break_points_to_win',
    'deciding_set_tie_break_points_to_win', 'deciding_set_format',
    'player_1_court_end', 'initial_state_confirmed', 'score_confirmed',
    'score_source', 'match_status',
))
_SET_FIELDS = frozenset((
    'set_number', 'games_p1', 'games_p2', 'tie_break_p1', 'tie_break_p2', 'winner_id',
))
_ENUM_FIELDS = {
    'format': 'MatchFormat',
    'set_format': 'SetFormat',
    'service_side': 'ServiceSide',
    'point_state': 'PointState',
    'ball_state': 'BallPlayState',
    'deciding_set_format': 'SetFormat',
    'player_1_court_end': 'ServerCourtEnd',
}
_BOOL_FIELDS = frozenset((
    'tie_break_active', 'match_complete', 'initial_state_confirmed', 'score_confirmed',
))
_COUNTER_FIELDS = frozenset((
    'sets_won_p1', 'sets_won_p2', 'games_p1', 'games_p2', 'points_p1', 'points_p2',
    'tie_break_points_p1', 'tie_break_points_p2', 'state_version',
))
_POSITIVE_FIELDS = frozenset((
    'player_1_id', 'player_2_id', 'server_id', 'receiver_id', 'initial_server_id',
    'current_set', 'current_game',
))
_TEXT_FIELDS = frozenset(('match_id', 'last_point_reason', 'score_source', 'match_status'))


def _require(condition, message):
    if not condition:
        raise ValueError('Invalid raw match state: ' + message)


def _schema():
    # Resolve only this fixed module's declared classes when the codec is used.
    # Importing this module alone never loads line geometry or inference code.
    from src.scoring.match_state import (
        BallPlayState, CompletedSetScore, MatchFormat, MatchState, PointState,
        ServerCourtEnd, ServiceSide, SetFormat,
    )
    enums = {
        'BallPlayState': BallPlayState, 'MatchFormat': MatchFormat,
        'PointState': PointState, 'ServerCourtEnd': ServerCourtEnd,
        'ServiceSide': ServiceSide, 'SetFormat': SetFormat,
    }
    # Changes to either dataclass require an intentional new codec contract.
    _require({field.name for field in fields(MatchState)} == _STATE_FIELDS,
             'MatchState fields differ from codec version 1')
    _require({field.name for field in fields(CompletedSetScore)} == _SET_FIELDS,
             'CompletedSetScore fields differ from codec version 1')
    return MatchState, CompletedSetScore, enums


def _object(value, names, label):
    _require(type(value) is dict and set(value) == names,
             label + ' requires its exact field set')
    return value


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum,
             label + ' requires an integer >= ' + str(minimum))


def _enum(value, kind, *, encode):
    if encode:
        _require(type(value) is kind, kind.__name__ + ' requires its declared enum type')
        return {'enum': kind.__name__, 'value': value.value}
    record = _object(value, {'enum', 'value'}, kind.__name__)
    _require(type(record['enum']) is str and record['enum'] == kind.__name__
             and type(record['value']) is str, 'unsupported enum tag or value type')
    try:
        return kind(record['value'])
    except ValueError as error:
        raise ValueError('Invalid raw match state: unknown ' + kind.__name__ + ' value') from error


def _validate(record):
    for name in _COUNTER_FIELDS:
        _integer(record[name], name)
    for name in _POSITIVE_FIELDS:
        _integer(record[name], name, 1)
    for name in _BOOL_FIELDS:
        _require(type(record[name]) is bool, name + ' requires a boolean')
    for name in _TEXT_FIELDS:
        _require(type(record[name]) is str, name + ' requires a string')
        if name != 'last_point_reason':
            _require(bool(record[name].strip()), name + ' requires nonblank text')
    _require(type(record['serve_attempt']) is int and record['serve_attempt'] in (1, 2),
             'serve_attempt must be 1 or 2')
    for name in ('tie_break_points_to_win', 'deciding_set_tie_break_points_to_win'):
        value = record[name]
        if value is None and name == 'deciding_set_tie_break_points_to_win':
            continue
        _integer(value, name, 1)
    participants = (record['player_1_id'], record['player_2_id'])
    _require(participants[0] != participants[1], 'participants must be distinct')
    for name in ('server_id', 'receiver_id', 'initial_server_id', 'winner_id', 'last_point_winner'):
        value = record[name]
        if value is None and name in ('winner_id', 'last_point_winner'):
            continue
        _require(type(value) is int and value in participants, name + ' must identify a participant')
    _require(record['server_id'] != record['receiver_id'], 'server and receiver must differ')
    _require(type(record['completed_sets']) is list, 'completed_sets requires a list')
    for completed in record['completed_sets']:
        _integer(completed['set_number'], 'completed set_number', 1)
        for name in ('games_p1', 'games_p2', 'tie_break_p1', 'tie_break_p2'):
            if completed[name] is None and name.startswith('tie_break_'):
                continue
            _integer(completed[name], 'completed ' + name)
        _require(type(completed['winner_id']) is int and completed['winner_id'] in participants,
                 'completed winner_id must identify a participant')


def encode_match_state(state):
    """Return detached plain JSON; never use a presentation state as input."""
    MatchState, CompletedSetScore, enums = _schema()
    _require(type(state) is MatchState, 'encoding requires MatchState')
    record = {field.name: getattr(state, field.name) for field in fields(MatchState)}
    _require(type(record['completed_sets']) is list, 'completed_sets requires a list')
    completed_sets = []
    for completed in record['completed_sets']:
        _require(type(completed) is CompletedSetScore, 'encoding requires CompletedSetScore rows')
        completed_sets.append({field.name: getattr(completed, field.name)
                               for field in fields(CompletedSetScore)})
    record['completed_sets'] = completed_sets
    _validate(record)
    for name, enum_name in _ENUM_FIELDS.items():
        if record[name] is None and name == 'deciding_set_format':
            continue
        record[name] = _enum(record[name], enums[enum_name], encode=True)
    record['completed_sets'] = [
        {'type': 'CompletedSetScore', 'fields': completed} for completed in completed_sets
    ]
    return {'schema_version': 1, 'type': 'MatchState', 'fields': record}


def decode_match_state(document):
    """Validate version 1 and construct detached, explicitly typed raw state."""
    MatchState, CompletedSetScore, enums = _schema()
    envelope = _object(document, {'schema_version', 'type', 'fields'}, 'envelope')
    _require(type(envelope['schema_version']) is int and envelope['schema_version'] == 1,
             'unsupported schema_version')
    _require(type(envelope['type']) is str and envelope['type'] == 'MatchState',
             'unsupported root type tag')
    record = dict(_object(envelope['fields'], _STATE_FIELDS, 'MatchState'))
    for name, enum_name in _ENUM_FIELDS.items():
        if record[name] is None and name == 'deciding_set_format':
            continue
        record[name] = _enum(record[name], enums[enum_name], encode=False)
    _require(type(record['completed_sets']) is list, 'completed_sets requires a list')
    completed_sets = []
    for item in record['completed_sets']:
        item = _object(item, {'type', 'fields'}, 'completed set envelope')
        _require(type(item['type']) is str and item['type'] == 'CompletedSetScore',
                 'unsupported completed set type tag')
        completed_sets.append(dict(_object(item['fields'], _SET_FIELDS, 'CompletedSetScore')))
    record['completed_sets'] = completed_sets
    _validate(record)
    record['completed_sets'] = [CompletedSetScore(**completed) for completed in completed_sets]
    return MatchState(**record)
