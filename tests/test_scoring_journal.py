"""Pure temporary-journal contract and failure tests; no match accuracy claim."""
from collections.abc import Sequence, Set
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from src.scoring.event_history import ScoringEventLogEntry
from src.utils import scoring_journal as module
from src.utils.scoring_journal import ScoringJournal


def records(event_id, *, marker='original'):
    history = dict(event_id=event_id, timestamp_seconds=event_id / 50.,
        event_type='SYNTHETIC_UNKNOWN', point_outcome='UNKNOWN_EVENT',
        previous_state=dict(marker=marker), new_state=dict(marker=marker),
        reason='Explicit synthetic owner test', confidence=.9)
    source = dict(event_id=event_id, event_type='SYNTHETIC_UNKNOWN',
        timestamp_seconds=event_id / 50., evidence=dict(marker=marker))
    return history, source


def append(view, event_id, *, marker='original', completed=None):
    return view.append_transition(*records(event_id, marker=marker),
                                  completed_point_id=completed)


def replay(view, *, marker='corrected'):
    with view.staged_correction(1, 'Explicit synthetic correction') as staged:
        for record in view.inputs:
            append(staged, record['event_id'], marker=marker,
                   completed='corrected-point' if record['event_id'] == 1 else None)
        staged.publish()
    return staged


def test_typed_uncached_rows_slices_and_explicit_compatibility(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        view = owner.revision(0)
        for event_id in range(1, 5):
            append(view, event_id)
        assert isinstance(view.history.entries, Sequence)
        assert isinstance(view.history.processed_event_ids, Set)
        assert not isinstance(view.history.entries, list)
        assert not isinstance(view.inputs, list)
        assert all(isinstance(entry, ScoringEventLogEntry) for entry in view.history.entries)
        assert view.history.entries[0].sequence_number == 1
        assert view.history.entries[-1].event_id == 4
        assert [entry.event_id for entry in view.history.entries[::-2]] == [4, 2]
        assert [entry.event_id for entry in view.history.entries[1:][::2]] == [2, 4]
        assert view.inputs[-1]['evidence'] == {'marker': 'original'}
        assert view.history.to_list() == [entry.to_dict() for entry in view.history.entries]
        view.history.entries[0].new_state['marker'] = 'caller mutation'
        view.inputs[0]['evidence']['marker'] = 'caller mutation'
        assert view.history.entries[0].new_state['marker'] == 'original'
        assert view.inputs[0]['evidence']['marker'] == 'original'
        assert view.history.is_processed(1) and not view.history.is_processed(9)
        assert set(view.history.processed_event_ids) == {1, 2, 3, 4}
        with pytest.raises(IndexError):
            view.inputs[4]
        with pytest.raises(IndexError):
            view.history.entries[-5]


def test_completed_point_keys_preserve_scalar_identity(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        view = owner.revision()
        append(view, 1, completed=1)
        append(view, 2, completed='1')
        append(view, 3, completed=1)
        assert set(view.completed_point_ids) == {1, '1'}
        assert len(view.completed_point_ids) == 2
        with pytest.raises(TypeError):
            [] in view.completed_point_ids


def test_cache_no_mmap_file_temporaries_and_bounded_ordinary_commits(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        connection = owner.connection
        assert owner._execute('PRAGMA cache_size', one=True)[0] == -2048
        assert owner._execute('PRAGMA mmap_size', one=True)[0] == 0
        assert owner._execute('PRAGMA temp_store', one=True)[0] == 1
        assert owner._execute('PRAGMA journal_mode', one=True)[0] == 'delete'
        view = owner.revision()
        for event_id in range(1, 257):
            append(view, event_id)
            assert owner.pending <= 256
        assert owner.pending == 0
        with closing(sqlite3.connect(owner.path)) as external:
            with closing(external.execute('SELECT COUNT(*) FROM transitions')) as cursor:
                assert cursor.fetchone()[0] == 256
        for event_id in range(257, 384):
            append(view, event_id, completed=event_id)
            assert owner.pending <= 256
        assert owner.pending == 254
        append(view, 384, completed=384)
        assert owner.pending == 0


@pytest.mark.parametrize('invalid', ['mismatch', 'nonfinite', 'duplicate', 'unserializable', 'invalid-point'])
def test_failed_event_does_not_lose_previously_accepted_pending_rows(tmp_path, invalid):
    with ScoringJournal(tmp_path) as owner:
        view = owner.revision()
        append(view, 1, completed='accepted')
        history, source = records(2)
        completed = None
        if invalid == 'mismatch':
            source['event_id'] = 3
        elif invalid == 'nonfinite':
            history['confidence'] = float('nan')
        elif invalid == 'duplicate':
            history, source = records(1)
        elif invalid == 'unserializable':
            source['evidence'] = object()
        else:
            completed = ('hashable-but-not-a-sqlite-scalar',)
        with pytest.raises((ValueError, TypeError, sqlite3.Error)):
            view.append_transition(history, source, completed_point_id=completed)
        assert len(view.history.entries) == len(view.inputs) == 1
        assert set(view.completed_point_ids) == {'accepted'}
        assert owner.pending == 2
        append(view, 2)
        assert [entry.event_id for entry in view.history.entries] == [1, 2]


def test_published_revisions_and_nested_prior_histories_are_pinned(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        original = owner.revision()
        append(original, 1, completed='original-point')
        append(original, 2)
        first = replay(original, marker='first')
        second = replay(first, marker='second')
        assert second.number == 2 and len(second.corrections) == 2
        assert original.history.entries[0].new_state['marker'] == 'original'
        assert first.history.entries[0].new_state['marker'] == 'first'
        assert second.history.entries[0].new_state['marker'] == 'second'
        assert second.corrections[0]['previous_history'][0]['new_state']['marker'] == 'original'
        assert second.corrections[-1]['previous_history'][0]['new_state']['marker'] == 'first'
        assert [row['revision'] for row in second.corrections[::-1]] == [2, 1]
        assert set(original.completed_point_ids) == {'original-point'}
        assert set(second.completed_point_ids) == {'corrected-point'}
        for historical in (original, first):
            with pytest.raises(ValueError, match='historical'):
                append(historical, 3)
        assert owner.revision(1).history.entries[0].new_state['marker'] == 'first'


@pytest.mark.parametrize('failure', ['missing-publish', 'after-publish', 'missing-event', 'reordered-events'])
def test_failed_correction_discards_staged_rows_even_after_publish_request(tmp_path, failure):
    with ScoringJournal(tmp_path) as owner:
        original = owner.revision()
        append(original, 1)
        append(original, 2)
        with pytest.raises((ValueError, LookupError)):
            with original.staged_correction(1, 'Synthetic rollback') as staged:
                ids = [2, 1] if failure == 'reordered-events' else [1, 2]
                if failure == 'missing-event':
                    ids = [1]
                for event_id in ids:
                    append(staged, event_id, marker='discarded')
                if failure != 'missing-publish':
                    staged.publish()
                if failure == 'after-publish':
                    raise LookupError('Synthetic failure after publication request')
        assert original.history.entries[0].new_state['marker'] == 'original'
        assert len(original.corrections) == 0
        with pytest.raises(ValueError, match='discarded'):
            len(staged.inputs)
        with pytest.raises(ValueError, match='Unknown published'):
            owner.revision(1)
        append(original, 3)
        assert len(original.inputs) == 3
        assert replay(original).number == 1


def test_correction_disallows_flush_nested_stage_old_writes_and_post_publish_writes(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        view = owner.revision()
        append(view, 1)
        with view.staged_correction(1, 'Synthetic guarded replay') as stage:
            with pytest.raises(ValueError):
                owner.flush()
            with pytest.raises(ValueError):
                append(view, 2)
            with pytest.raises(ValueError):
                with stage.staged_correction(1, 'Nested correction'):
                    pass
            append(stage, 1)
            stage.publish()
            with pytest.raises(ValueError, match='already requested'):
                append(stage, 2)
        assert len(owner.revision(1).inputs) == 1


class ConnectionProxy:
    def __init__(self, connection, *, commit_failure=False, rollback_failure=False):
        self.connection = connection
        self.commit_failure = commit_failure
        self.rollback_failure = rollback_failure

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def commit(self):
        if self.commit_failure:
            raise OSError('Synthetic commit failure at actual batch boundary')
        return self.connection.commit()

    def rollback(self):
        if self.rollback_failure:
            raise OSError('Synthetic rollback failure')
        return self.connection.rollback()


@pytest.mark.parametrize('boundary', ['automatic', 'explicit'])
def test_failed_ordinary_commit_invalidates_every_view_and_export(tmp_path, boundary):
    with ScoringJournal(tmp_path) as owner:
        view = owner.revision()
        for event_id in range(1, 256):
            append(view, event_id)
        assert len(view.inputs) == 255 and owner.pending == 255
        owner.connection = ConnectionProxy(owner.connection, commit_failure=True)
        target = tmp_path / 'prior.json'
        target.write_text('prior output', encoding='utf-8')
        with pytest.raises(OSError, match='commit failure'):
            if boundary == 'automatic':
                append(view, 256)
            else:
                owner.flush()
        for read_or_write in (lambda: len(view.inputs),
                lambda: view.history.is_processed(1), lambda: list(view.completed_point_ids),
                lambda: list(view.corrections), lambda: owner.revision(),
                lambda: append(view, 257), lambda: view.save_history_json(target)):
            with pytest.raises(RuntimeError, match='invalid after a transaction failure'):
                read_or_write()
        assert target.read_text(encoding='utf-8') == 'prior output'
    assert not list(tmp_path.glob('*.sqlite3*'))


def test_commit_primary_exception_survives_failed_best_effort_rollback(tmp_path):
    with ScoringJournal(tmp_path) as owner:
        append(owner.revision(), 1)
        owner.connection = ConnectionProxy(owner.connection,
            commit_failure=True, rollback_failure=True)
        with pytest.raises(OSError, match='commit failure') as failure:
            owner.flush()
        assert any('rollback also failed' in note for note in failure.value.__notes__)
        with pytest.raises(RuntimeError, match='invalid'):
            len(owner.revision().inputs)


def test_streamed_export_includes_current_and_nested_revision_records(tmp_path, monkeypatch):
    with ScoringJournal(tmp_path) as owner:
        original = owner.revision()
        append(original, 1)
        append(original, 2)
        current = replay(replay(original, marker='first'), marker='second')
        def materialization_forbidden(*args):
            raise AssertionError('Streamed export must not call history.to_list')
        monkeypatch.setattr(module.JournalHistory, 'to_list', materialization_forbidden)
        target = tmp_path / 'history.json'
        current.save_history_json(target)
        exported = json.loads(target.read_text(encoding='utf-8'))
        assert exported['schema_version'] == '2.0' and exported['revision'] == 2
        assert [entry['event_id'] for entry in exported['score_history']] == [1, 2]
        assert exported['score_history'][0]['new_state']['marker'] == 'second'
        assert exported['corrections'][0]['previous_history'][0]['new_state']['marker'] == 'original'
        assert exported['corrections'][1]['previous_history'][0]['new_state']['marker'] == 'first'
        assert not owner.readers
        assert not list(tmp_path.glob('history.json.*.tmp'))


@pytest.mark.parametrize('failure', ['nested-write', 'replace'])
def test_failed_export_closes_every_reader_and_preserves_destination(tmp_path, monkeypatch, failure):
    with ScoringJournal(tmp_path) as owner:
        original = owner.revision()
        append(original, 1)
        current = replay(original)
        target = tmp_path / 'history.json'
        target.write_text('prior output', encoding='utf-8')
        if failure == 'replace':
            def fail_replace(*args):
                raise OSError('Synthetic atomic replacement failure')
            monkeypatch.setattr(module.os, 'replace', fail_replace)
        else:
            original_json = module._json
            def fail_nested(value):
                if isinstance(value, dict) and value.get('new_state', {}).get('marker') == 'original':
                    raise OSError('Synthetic nested prior-history write failure')
                return original_json(value)
            monkeypatch.setattr(module, '_json', fail_nested)
        with pytest.raises(OSError):
            current.save_history_json(target)
        assert target.read_text(encoding='utf-8') == 'prior output'
        assert not owner.readers
        assert not list(tmp_path.glob('history.json.*.tmp'))
        assert current.history.entries[0].new_state['marker'] == 'corrected'


def test_export_primary_exception_survives_temporary_unlink_failure(tmp_path, monkeypatch):
    with ScoringJournal(tmp_path) as owner:
        current = owner.revision()
        append(current, 1)
        target = tmp_path / 'history.json'
        target.write_text('prior output', encoding='utf-8')
        original_unlink = Path.unlink
        def fail_unlink(path, *args, **kwargs):
            if path.name.startswith('history.json.') and path.suffix == '.tmp':
                raise OSError('Synthetic temporary unlink failure')
            return original_unlink(path, *args, **kwargs)
        def fail_replace(*args):
            raise LookupError('Synthetic primary replacement failure')
        monkeypatch.setattr(module.os, 'replace', fail_replace)
        monkeypatch.setattr(Path, 'unlink', fail_unlink)
        with pytest.raises(LookupError, match='primary replacement failure') as failure:
            current.save_history_json(target)
        assert any('temporary cleanup also failed' in note for note in failure.value.__notes__)
        assert target.read_text(encoding='utf-8') == 'prior output'
        assert not owner.readers


def test_owner_closes_suspended_readers_and_views_fail_after_exit(tmp_path):
    owner = ScoringJournal(tmp_path)
    with owner:
        view = owner.revision()
        append(view, 1)
        append(view, 2)
        readers = [iter(view.history.entries), iter(view.inputs), iter(view.history.processed_event_ids)]
        for iterator in readers:
            next(iterator)
        assert len(owner.readers) == 3
        path = owner.path
    assert not path.exists() and not owner.readers
    for iterator in readers:
        with pytest.raises(RuntimeError, match='not open'):
            next(iterator)
    with pytest.raises(RuntimeError, match='not open'):
        len(view.inputs)
    with pytest.raises(RuntimeError, match='cannot be reused'):
        owner.__enter__()


def test_owner_cleanup_does_not_mask_callers_primary_failure(tmp_path, monkeypatch):
    owner = ScoringJournal(tmp_path)
    original_unlink = Path.unlink
    def fail_owned_unlink(path, *args, **kwargs):
        if owner.path is not None and path == owner.path:
            raise OSError('Synthetic owned-path cleanup failure')
        return original_unlink(path, *args, **kwargs)
    with pytest.raises(LookupError, match='caller primary') as failure:
        with owner:
            append(owner.revision(), 1)
            monkeypatch.setattr(Path, 'unlink', fail_owned_unlink)
            raise LookupError('caller primary')
    assert any('cleanup also failed' in note for note in failure.value.__notes__)
    assert owner.connection is None
