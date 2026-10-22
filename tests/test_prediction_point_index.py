"""Caller-owned identity indexes keep forecast arithmetic and retries intact."""
from __future__ import annotations

from collections.abc import MutableMapping
import sqlite3
from types import MappingProxyType

import pytest

from src.prediction.tennis import ServePointPrior, TennisWinProbability
from src.scoring.match_state import MatchState
from src.scoring.scoring_rules import TennisScoringRules
from src.utils.forecast_sequence import ForecastSequence


class _ObservedIndex(MutableMapping):
    """Tiny test mapping that exposes any engine iteration or fallback."""
    def __init__(self):
        self.rows = {}
        self.operations = []
        self.fail_at = None

    def _operation(self, name):
        self.operations.append(name)
        if self.fail_at == name:
            raise RuntimeError('Explicit index ' + name + ' failure')

    def __getitem__(self, key):
        self._operation('get')
        return self.rows[key]

    def __setitem__(self, key, value):
        self._operation('set')
        self.rows[key] = value

    def __delitem__(self, key):
        self._operation('delete')
        del self.rows[key]

    def __iter__(self):
        raise AssertionError('Forecast must not copy or iterate the duration index')

    def __len__(self):
        self._operation('len')
        return len(self.rows)

    def __contains__(self, key):
        self._operation('contains')
        return key in self.rows


def test_empty_external_dict_is_retained_and_updates_use_the_same_object():
    index = {}
    engine = TennisWinProbability(point_index=index)
    assert engine.point_ids is index
    engine.observe_point(point_id='point-1', server_id=1, winner_id=2, timestamp_seconds=1.25)
    assert index == {'point-1': (1, 2, 1.25)}
    assert engine.counts == {1: [0, 1], 2: [0, 0]}
    snapshot = engine.snapshot(MatchState(), timestamp_seconds=1.25)
    assert snapshot['sample_count'] == 1 and snapshot['status'] == 'uncalibrated'


def test_manual_default_stays_an_independent_dict_and_external_index_is_keyword_only():
    first, second = TennisWinProbability(), TennisWinProbability()
    assert type(first.point_ids) is type(second.point_ids) is dict
    assert first.point_ids is not second.point_ids
    with pytest.raises(TypeError):
        TennisWinProbability((1, 2), None, {})


@pytest.mark.parametrize('unsupported', [[], (), set(), object(), MappingProxyType({})])
def test_index_requires_a_mutable_mapping(unsupported):
    with pytest.raises(TypeError, match='mutable mapping'):
        TennisWinProbability(point_index=unsupported)


@pytest.mark.parametrize('index', [{'old': (1, 1, 1.)}, _ObservedIndex()])
def test_nonempty_index_cannot_start_with_fresh_posterior_counts(index):
    if isinstance(index, _ObservedIndex):
        index.rows['old'] = (1, 1, 1.)
    with pytest.raises(ValueError, match='empty point index'):
        TennisWinProbability(point_index=index)


def test_external_index_idempotency_conflict_and_order_match_manual_behavior():
    index = _ObservedIndex()
    indexed = TennisWinProbability(point_index=index)
    manual = TennisWinProbability()
    assert indexed.point_ids is index
    for engine in (indexed, manual):
        engine.observe_point(point_id='p-1', server_id=1, winner_id=1, timestamp_seconds=2.)
        engine.observe_point(point_id='p-2', server_id=2, winner_id=1, timestamp_seconds=3.)
        # An exact retry from an earlier timestamp remains idempotent.
        engine.observe_point(point_id='p-1', server_id=1, winner_id=1, timestamp_seconds=2.)
        with pytest.raises(ValueError, match='different outcome'):
            engine.observe_point(point_id='p-1', server_id=1, winner_id=2, timestamp_seconds=2.)
        with pytest.raises(ValueError, match='ordered'):
            engine.observe_point(point_id='p-3', server_id=1, winner_id=1, timestamp_seconds=1.)
    assert indexed.counts == manual.counts == {1: [1, 1], 2: [0, 1]}
    assert indexed.last_timestamp == manual.last_timestamp == 3.
    assert indexed.snapshot(MatchState(), revision=4, timestamp_seconds=3.) == \
        manual.snapshot(MatchState(), revision=4, timestamp_seconds=3.)
    assert set(vars(indexed)) == {'player_ids', 'prior', 'counts', 'point_ids', 'last_timestamp'}
    assert index.rows == manual.point_ids


@pytest.mark.parametrize('operation', ['contains', 'set'])
def test_index_update_failures_propagate_before_counters_and_time_change(operation):
    index = _ObservedIndex()
    engine = TennisWinProbability(point_index=index)
    index.fail_at = operation
    with pytest.raises(RuntimeError, match=operation + ' failure'):
        engine.observe_point(point_id='p-1', server_id=1, winner_id=1, timestamp_seconds=1.)
    assert engine.point_ids is index
    assert index.rows == {}
    assert engine.counts == {1: [0, 0], 2: [0, 0]}
    assert engine.last_timestamp is None
    index.fail_at = None
    engine.observe_point(point_id='p-1', server_id=1, winner_id=1, timestamp_seconds=1.)
    assert index.rows == {'p-1': (1, 1, 1.)}


def test_duplicate_lookup_failure_does_not_fall_back_or_double_count():
    index = _ObservedIndex()
    engine = TennisWinProbability(point_index=index)
    engine.observe_point(point_id='p-1', server_id=1, winner_id=2, timestamp_seconds=1.)
    index.fail_at = 'get'
    with pytest.raises(RuntimeError, match='get failure'):
        engine.observe_point(point_id='p-1', server_id=1, winner_id=2, timestamp_seconds=1.)
    assert engine.point_ids is index and len(index.rows) == 1
    assert engine.counts == {1: [0, 1], 2: [0, 0]}
    assert engine.last_timestamp == 1.


def test_closed_or_failed_index_refuses_constructor_and_snapshot_without_fallback():
    index = _ObservedIndex()
    index.fail_at = 'len'
    with pytest.raises(RuntimeError, match='len failure'):
        TennisWinProbability(point_index=index)
    index.fail_at = None
    engine = TennisWinProbability(point_index=index)
    index.fail_at = 'len'
    with pytest.raises(RuntimeError, match='len failure'):
        engine.snapshot(MatchState())
    assert engine.point_ids is index


def test_prior_and_score_only_forecasts_are_identical_with_external_storage():
    state = MatchState(points_p1=2, points_p2=1)
    prior = ServePointPrior(alpha=3., beta=4., source='synthetic-explicit-prior')
    indexed = TennisWinProbability(prior=prior, point_index=_ObservedIndex())
    manual = TennisWinProbability(prior=prior)
    assert indexed.snapshot(state, revision=2, timestamp_seconds=0.) == \
        manual.snapshot(state, revision=2, timestamp_seconds=0.)


def test_constructor_does_not_use_truthiness_to_replace_a_supplied_empty_index():
    class NoTruthinessIndex(_ObservedIndex):
        def __bool__(self):
            raise AssertionError('Constructor must select the supplied index by None only')

    index = NoTruthinessIndex()
    engine = TennisWinProbability(point_index=index)
    assert engine.point_ids is index
    engine.observe_point(point_id='p-1', server_id=1, winner_id=1, timestamp_seconds=1.)
    assert index.rows == {'p-1': (1, 1, 1.)}


def test_duration_point_ids_stay_in_the_supplied_index_without_copy_or_hidden_cache():
    index = _ObservedIndex()
    engine = TennisWinProbability(point_index=index)
    initial_fields = set(vars(engine))
    for number in range(2049):
        engine.observe_point(point_id='native-point-' + str(number), server_id=1,
                             winner_id=1 if number % 2 else 2, timestamp_seconds=float(number))
    assert engine.point_ids is index and len(index) == 2049
    assert set(vars(engine)) == initial_fields
    assert engine.counts == {1: [1024, 2049], 2: [0, 0]}
    assert engine.snapshot(MatchState(), timestamp_seconds=2048.)['sample_count'] == 2049
    # Iteration deliberately raises in _ObservedIndex; constructor, updates,
    # retries and snapshot must use indexed membership/lookup/count operations.
    assert set(index.operations) == {'len', 'contains', 'set'}


def test_actual_sql_index_retains_exact_object_and_matches_every_causal_prefix(tmp_path):
    """Compare complete actual snapshots, not only the final probability."""
    state = MatchState(match_id='sql-prefix-regression')
    prior = ServePointPrior(alpha=3., beta=4., source='explicit-synthetic-regression-prior')
    manual = TennisWinProbability(prior=prior)
    with ForecastSequence(tmp_path, transaction_size=3) as owner:
        supplied = owner.point_index
        indexed = TennisWinProbability(prior=prior, point_index=supplied)
        assert indexed.point_ids is owner.point_index is supplied
        for number, winner in enumerate((1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2)):
            timestamp = (number + 1) / 30.
            assert indexed.snapshot(state, revision=7, timestamp_seconds=timestamp) == \
                manual.snapshot(state, revision=7, timestamp_seconds=timestamp)
            data = dict(point_id='actual-point-' + str(number), server_id=state.server_id,
                        winner_id=winner, timestamp_seconds=timestamp)
            indexed.observe_point(**data)
            manual.observe_point(**data)
            state = TennisScoringRules.award_point(state, winner, 'Synthetic confirmed prefix')
            actual = indexed.snapshot(state, revision=7, timestamp_seconds=timestamp)
            assert actual == manual.snapshot(state, revision=7, timestamp_seconds=timestamp)
            assert actual['sample_count'] == number + 1
            assert actual['calibration_status'] == 'not_calibrated'
            assert indexed.counts == manual.counts
            assert indexed.last_timestamp == manual.last_timestamp == timestamp
            assert owner.point_index[data['point_id']] == (data['server_id'], winner, timestamp)
            owner.history.append(actual)
            assert owner.pending < owner.transaction_size
        owner.seal()
        assert len(owner.history) == len(owner.point_index) == 12
        assert owner.history[-1] == actual
        assert dict(owner.point_index) == manual.point_ids
    assert not list(tmp_path.iterdir())


def test_actual_sql_index_retry_conflict_and_source_order_match_dictionary(tmp_path):
    with ForecastSequence(tmp_path) as owner:
        indexed = TennisWinProbability(point_index=owner.point_index)
        manual = TennisWinProbability()
        for engine in (indexed, manual):
            engine.observe_point(point_id='p1', server_id=1, winner_id=2, timestamp_seconds=1.25)
            engine.observe_point(point_id='p2', server_id=2, winner_id=2, timestamp_seconds=2.25)
        before_pending = owner.pending
        for engine in (indexed, manual):
            engine.observe_point(point_id='p1', server_id=1, winner_id=2, timestamp_seconds=1.25)
            for changed in ({'server_id': 2}, {'winner_id': 1}, {'timestamp_seconds': 1.5}):
                data = dict(point_id='p1', server_id=1, winner_id=2, timestamp_seconds=1.25)
                data.update(changed)
                with pytest.raises(ValueError, match='different outcome'):
                    engine.observe_point(**data)
            with pytest.raises(ValueError, match='ordered'):
                engine.observe_point(point_id='older', server_id=1, winner_id=1, timestamp_seconds=.5)
        assert owner.pending == before_pending
        assert indexed.point_ids is owner.point_index
        assert dict(owner.point_index) == manual.point_ids
        assert indexed.counts == manual.counts == {1: [0, 1], 2: [1, 1]}
        assert indexed.last_timestamp == manual.last_timestamp == 2.25
        assert indexed.snapshot(MatchState(), timestamp_seconds=2.25) == \
            manual.snapshot(MatchState(), timestamp_seconds=2.25)


def test_actual_sql_index_constructor_rejects_existing_rows_and_closed_owner(tmp_path):
    with ForecastSequence(tmp_path) as owner:
        index = owner.point_index
        index['already-observed'] = (1, 2, .25)
        with pytest.raises(ValueError, match='empty point index'):
            TennisWinProbability(point_index=index)
        assert index['already-observed'] == (1, 2, .25)
    with pytest.raises(RuntimeError, match='not open'):
        TennisWinProbability(point_index=index)
    assert not list(tmp_path.iterdir())


def test_actual_sql_index_closed_owner_refuses_update_and_snapshot_without_fallback(tmp_path):
    with ForecastSequence(tmp_path) as owner:
        index = owner.point_index
        indexed = TennisWinProbability(point_index=index)
        indexed.observe_point(point_id='committed', server_id=1, winner_id=2, timestamp_seconds=1.)
        owner.seal()
        assert indexed.snapshot(MatchState(), timestamp_seconds=1.)['sample_count'] == 1
    for action in (
        lambda: indexed.observe_point(point_id='committed', server_id=1, winner_id=2, timestamp_seconds=1.),
        lambda: indexed.observe_point(point_id='next', server_id=2, winner_id=2, timestamp_seconds=2.),
        lambda: indexed.snapshot(MatchState(), timestamp_seconds=2.),
    ):
        with pytest.raises(RuntimeError, match='not open'):
            action()
        assert indexed.point_ids is index
        assert indexed.counts == {1: [0, 1], 2: [0, 0]}
        assert indexed.last_timestamp == 1.


def test_actual_sql_index_predictor_does_not_iterate_copy_or_keep_another_identity_cache(tmp_path, monkeypatch):
    with ForecastSequence(tmp_path, transaction_size=7) as owner:
        def forbidden_iteration(_self):
            raise AssertionError('Predictor must use the actual SQL identity index directly')
        monkeypatch.setattr(type(owner.point_index), '__iter__', forbidden_iteration)
        indexed = TennisWinProbability(point_index=owner.point_index)
        original_fields = set(vars(indexed))
        for number in range(257):
            indexed.observe_point(point_id='sql-point-' + str(number), server_id=1,
                winner_id=1 if number % 2 else 2, timestamp_seconds=float(number))
        indexed.observe_point(point_id='sql-point-0', server_id=1, winner_id=2, timestamp_seconds=0.)
        assert indexed.snapshot(MatchState(), timestamp_seconds=256.)['sample_count'] == 257
        assert set(vars(indexed)) == original_fields == {
            'player_ids', 'prior', 'counts', 'point_ids', 'last_timestamp'}
        assert indexed.point_ids is owner.point_index
        assert indexed.counts == {1: [128, 257], 2: [0, 0]}
        assert len(owner.point_index) == 257


def test_actual_sql_commit_failure_rolls_back_before_posterior_or_clock_changes(tmp_path):
    with ForecastSequence(tmp_path, transaction_size=1) as owner:
        indexed = TennisWinProbability(point_index=owner.point_index)
        indexed.observe_point(point_id='previous', server_id=1, winner_id=2, timestamp_seconds=1.)
        connection = owner.connection
        transactions = []
        def deny_commit(operation, first, _second, _database, _trigger):
            if operation == sqlite3.SQLITE_TRANSACTION:
                transactions.append(first)
                if first == 'COMMIT':
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(deny_commit)
        with pytest.raises(sqlite3.DatabaseError, match='not authorized'):
            indexed.observe_point(point_id='failed', server_id=2, winner_id=2, timestamp_seconds=2.)
        assert transactions == ['BEGIN', 'COMMIT', 'ROLLBACK']
        assert owner.pending == 0 and 'DatabaseError' in owner._failure
        assert not connection.in_transaction
        # Bypass the now-invalid owner only to inspect the real committed DB.
        assert connection.execute('SELECT point_id FROM point_index ORDER BY rowid').fetchall() == [('previous',)]
        assert indexed.point_ids is owner.point_index
        assert indexed.counts == {1: [0, 1], 2: [0, 0]}
        assert indexed.last_timestamp == 1.
        for action in (
            lambda: indexed.snapshot(MatchState(), timestamp_seconds=2.),
            lambda: indexed.observe_point(point_id='failed', server_id=2, winner_id=2, timestamp_seconds=2.),
            lambda: indexed.observe_point(point_id='previous', server_id=1, winner_id=2, timestamp_seconds=1.),
        ):
            with pytest.raises(RuntimeError, match='transaction failure'):
                action()
            assert indexed.point_ids is owner.point_index
            assert indexed.counts == {1: [0, 1], 2: [0, 0]}
            assert indexed.last_timestamp == 1.
    assert not list(tmp_path.iterdir())


def test_actual_sql_index_preserves_nondefault_participants_at_each_prefix(tmp_path):
    state = MatchState(match_id='explicit-synthetic-ids', player_1_id=17,
                       player_2_id=43, server_id=43, initial_server_id=43)
    with ForecastSequence(tmp_path, player_ids=(17, 43)) as owner:
        indexed = TennisWinProbability((17, 43), point_index=owner.point_index)
        manual = TennisWinProbability((17, 43))
        for number, winner in enumerate((17, 43, 17, 17, 43)):
            data = dict(point_id='nondefault-' + str(number), server_id=state.server_id,
                        winner_id=winner, timestamp_seconds=number + .125)
            indexed.observe_point(**data)
            manual.observe_point(**data)
            state = TennisScoringRules.award_point(state, winner, 'Synthetic nondefault participant prefix')
            assert indexed.snapshot(state, revision=2, timestamp_seconds=data['timestamp_seconds']) == \
                manual.snapshot(state, revision=2, timestamp_seconds=data['timestamp_seconds'])
            assert indexed.point_ids is owner.point_index
            assert indexed.counts == manual.counts
