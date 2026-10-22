"""Bounded original-source decoding for unchanged and corrected rule consumers.

These use real rule definitions with synthetic evidence, without model or image
runtimes. A source decoder deliberately does not replay recorded line decisions.
"""
from collections.abc import Sequence
from contextlib import contextmanager
import copy

import pytest

from src.api.result_storage import DiskSeries, ResultStorage
from src.scoring.match_state_codec import encode_match_state
from src.scoring.point_timeline import context_record
from src.scoring.rule_replay_source import open_rule_replay_source
from tests.test_rule_reproduction import (
    api, build_replay, coherent_actual_scoring_definitions, file_sha,
    no_model_or_image_imports, plain, rewrite,
)


@contextmanager
def open_source(fixture, directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    kwargs = {'artifact_sha256': fixture.digest, 'source_sha256': fixture.source_sha,
              'analysis_id': fixture.analysis_id, 'analysis_revision': fixture.revision}
    kwargs.update(overrides)
    with open_rule_replay_source(fixture.path, directory, **kwargs) as value:
        yield value


def test_source_decoder_retains_exact_raw_states_and_disk_backed_rows(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original', scenario='sparse_rally')
    directory = tmp_path / 'decode'
    with open_source(fixture, directory) as source:
        assert isinstance(source.source_frames, Sequence)
        assert isinstance(source.source_frames.rows, DiskSeries)
        # The parser keeps small arrays within its fixed item bound. Only the
        # duration-dependent source audit must spill for this sparse fixture.
        assert isinstance(source.document['inputs'], (list, DiskSeries))
        assert isinstance(source.document['shots'], (list, DiskSeries))
        assert len(source.source_frames) == len(fixture.source)
        assert encode_match_state(source.initial_state) == fixture.initial_raw
        assert encode_match_state(source.final_state) == fixture.final_raw
        assert source.authoritative_events_enabled is True
        assert source.storage.path.is_file()
        assert source.source_frames.analysis_id == fixture.analysis_id
        assert source.source_frames.analysis_revision == fixture.revision
        assert source.source_frames.source_sha256 == fixture.source_sha
        assert source.source_frames.artifact_sha256 == fixture.digest
        for index, original in enumerate(fixture.source):
            assert source.source_frames[index] == original
        for row in source.document['inputs']:
            original = plain(row['source_input'])
            event, context = source.decode_source_event(original)
            assert event.event_id == original['event_id']
            assert context_record(context) == original['context']
        rows = source.source_frames.rows
    assert not list(directory.glob('.result_storage_*'))
    with pytest.raises(RuntimeError, match='not open'):
        len(rows)
    with pytest.raises(RuntimeError, match='not open'):
        len(source.source_frames)


def test_source_decoder_never_runs_original_transitions_or_line_decisions(api, tmp_path, monkeypatch):
    fixture = build_replay(api, tmp_path / 'original')

    def forbid(*args, **kwargs):
        raise AssertionError('Source decoding must not apply recorded rule decisions')

    monkeypatch.setattr(api.TennisScoringEngine, '_decode_input', forbid)
    monkeypatch.setattr(api.TennisScoringEngine, 'process_match_event', forbid)
    with open_source(fixture, tmp_path / 'decode') as source:
        for row in source.document['inputs']:
            source.decode_source_event(row['source_input'])
        assert not hasattr(source, 'evaluate_bounce')
        assert not hasattr(source, 'engine')
        assert not hasattr(source, 'timeline')


def test_detached_replacement_is_rebound_without_changing_sealed_original(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        original = plain(source.document['inputs'][0]['source_input'])
        replacement = copy.deepcopy(original)
        replacement['confidence'] = .876543210123456
        replacement['evidence'] = {'review_ref': 'SYNTHETIC_SOURCE_REPLACEMENT'}
        event, context = source.decode_source_event(replacement)
        assert event.confidence == replacement['confidence']
        assert event.evidence == replacement['evidence']
        assert context_record(context) == original['context']
        assert plain(source.document['inputs'][0]['source_input']) == original
    assert file_sha(fixture.path) == fixture.digest


@pytest.mark.parametrize('mutation', [
    lambda record: record['context']['binding'].__setitem__('source_fingerprint', 'b' * 64),
    lambda record: record['context']['binding'].__setitem__('analysis_id', 'another'),
    lambda record: record['context']['binding'].__setitem__('source_pts', '42'),
    lambda record: record['context'].__setitem__('anchor_frame_index', 10),
    lambda record: record.__setitem__('frame_index', True),
    lambda record: record.__setitem__('event_id', False),
    lambda record: record.__setitem__('confidence', float('nan')),
])
def test_replacement_rejects_invalid_current_source_binding(api, tmp_path, mutation):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        mutation(replacement)
        with pytest.raises(ValueError):
            source.decode_source_event(replacement)


@pytest.mark.parametrize('index', [True, 1.0, '1', slice(None)])
def test_source_frames_require_a_single_strict_integer_index(api, tmp_path, index):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        with pytest.raises(TypeError):
            source.source_frames[index]
        assert source.source_frames[-1] == fixture.source[-1]
        with pytest.raises(IndexError):
            source.source_frames[len(source.source_frames)]


def test_source_provenance_properties_cannot_be_reassigned(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        for name in ('analysis_revision', 'source_sha256', 'artifact_sha256'):
            with pytest.raises(AttributeError):
                setattr(source.source_frames, name, 'different')


@pytest.mark.parametrize('mutation', [
    lambda document: document.__setitem__('source_frame_count', True),
    lambda document: document.__setitem__('input_count', True),
    lambda document: document.__setitem__('shot_count', -1),
    lambda document: document.__setitem__('scoring_revision', 1),
    lambda document: document.__setitem__('analysis_revision', True),
    lambda document: document.__setitem__('authoritative_events_enabled', 1),
    lambda document: document.__setitem__('independent_accuracy_validated', True),
    lambda document: document.__setitem__('authoritative_events_enabled', False),
    lambda document: document['source_frames'][100]['interval'].__setitem__('analysis_id', 'another'),
    lambda document: document['source_frames'][100].__setitem__('frame_index', 99),
])
def test_shared_entry_keeps_original_envelope_count_and_all_frame_guards(api, tmp_path, mutation):
    fixture = build_replay(api, tmp_path / 'original')
    rewrite(fixture, mutation)
    fixture.digest = file_sha(fixture.path)
    with pytest.raises(ValueError):
        with open_source(fixture, tmp_path / 'decode'):
            pytest.fail('Invalid externally sealed source envelope was admitted')


def test_authority_off_source_decodes_without_inventing_inputs(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original', authority=False)
    with open_source(fixture, tmp_path / 'decode') as source:
        assert source.authoritative_events_enabled is False
        assert len(source.document['inputs']) == len(source.document['shots']) == 0
        assert encode_match_state(source.initial_state) == encode_match_state(source.final_state)


def test_hash_is_rechecked_after_disk_backed_parsing(api, tmp_path, monkeypatch):
    fixture = build_replay(api, tmp_path / 'original')
    read_json = ResultStorage.read_json

    def mutate_after_read(storage, path, **kwargs):
        result = read_json(storage, path, **kwargs)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(' ')
        return result

    monkeypatch.setattr(ResultStorage, 'read_json', mutate_after_read)
    with pytest.raises(ValueError, match='changed during reading'):
        with open_source(fixture, tmp_path / 'decode'):
            pytest.fail('Artifact changed during parsing was admitted')


@pytest.mark.parametrize('value', [True, False, 1.0, 2.0, 0, 3, '1'])
def test_physical_player_role_rejects_boolean_float_and_unsupported_aliases(api, tmp_path, value):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        replacement['player_id'] = value
        with pytest.raises(ValueError, match='physical player role'):
            source.decode_source_event(replacement)


@pytest.mark.parametrize('field,value', [
    ('frame_index', False), ('frame_index', 0.0), ('frame_index', -1),
    ('source_frame_index', False), ('source_frame_index', 0.0), ('source_frame_index', -1),
    ('analysis_allowed', 1), ('analysis_allowed', 1.0),
    ('binding_audit_matches', 1), ('binding_audit_matches', 0.0),
    ('source_pts', 1), ('source_time_base', True), ('segment_id', 1),
    ('review_evidence_ref', True), ('binding', []),
    ('anchor_frame_index', False), ('anchor_frame_index', 0.0), ('anchor_frame_index', -1),
])
def test_context_primitives_reject_python_equality_aliases(api, tmp_path, field, value):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        replacement['context'][field] = value
        with pytest.raises(ValueError):
            source.decode_source_event(replacement)


@pytest.mark.parametrize('field,value', [
    ('frame_index', False), ('frame_index', 0.0), ('frame_index', -1),
    ('role_id', True), ('role_id', 1.0), ('participant_id', True), ('participant_id', 1.0),
    ('identity_revision', False), ('identity_revision', 0.0), ('identity_revision', -1),
    ('anchor_frame_index', False), ('anchor_frame_index', 0.0), ('anchor_frame_index', -1),
    ('source_frame_index', False), ('source_frame_index', 0.0), ('source_frame_index', -1),
    ('identity_epoch', False), ('identity_epoch', 0.0), ('identity_epoch', -1),
    ('source_pts', 1), ('source_time_base', False),
    ('segment_id', True), ('match_identity', True), ('track_key', True), ('court_end', True),
    ('calibration_revision', True), ('evidence_ref', True), ('analysis_id', True),
    ('source_fingerprint', True), ('participant_mapping_ref', True),
    ('geometry_qualification_ref', True), ('participant_mapping_ref', None),
])
def test_binding_primitives_reject_aliases_and_nontext_provenance(api, tmp_path, field, value):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        replacement['context']['binding'][field] = value
        with pytest.raises(ValueError, match='binding primitive types'):
            source.decode_source_event(replacement)


def test_unavailable_binding_retains_none_optional_fields_and_empty_provenance(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        replacement['context']['binding'].update(source_frame_index=None, source_pts=None,
            source_time_base=None, identity_epoch=None, analysis_id='', source_fingerprint='',
            participant_mapping_ref='', geometry_qualification_ref='')
        replacement['context']['binding_audit_matches'] = False
        event, context = source.decode_source_event(replacement)
        assert event.player_id == 1
        assert context.binding is not None and not context.binding_current
        assert context.binding.source_frame_index is context.binding.identity_epoch is None
        assert context.binding.source_pts is context.binding.source_time_base is None
        assert context.binding.participant_mapping_ref == context.binding.geometry_qualification_ref == ''


def test_absent_binding_and_nullable_physical_role_remain_unavailable(api, tmp_path):
    fixture = build_replay(api, tmp_path / 'original')
    with open_source(fixture, tmp_path / 'decode') as source:
        replacement = plain(source.document['inputs'][0]['source_input'])
        replacement['player_id'] = None
        replacement['context'].update(binding=None, anchor_frame_index=None, binding_audit_matches=False)
        event, context = source.decode_source_event(replacement)
        assert event.player_id is None and context.binding is None and not context.binding_current


@pytest.mark.parametrize('field,value', [
    ('epoch', False), ('epoch', 0.0), ('epoch', -1),
    ('revision', False), ('revision', 0.0), ('revision', -1),
    ('frame_index', False), ('frame_index', 0.0), ('frame_index', -1),
])
def test_raw_source_identity_rejects_boolean_float_and_negative_counters(api, tmp_path, field, value):
    fixture = build_replay(api, tmp_path / 'original')
    rewrite(fixture, lambda document: document['source_frames'][100]['identity'].__setitem__(field, value))
    fixture.digest = file_sha(fixture.path)
    with pytest.raises(ValueError, match='source identity counters'):
        with open_source(fixture, tmp_path / 'decode'):
            pytest.fail('Malformed identity counter was admitted')


@pytest.mark.parametrize('field,value', [
    ('player_id', True), ('player_id', 1), ('player_id', 1.0),
    ('player_id', None), ('player_id', 'C'),
    ('track_key', 1), ('track_key', False), ('state', True),
])
def test_raw_source_observation_primitives_and_confirmed_identity_are_strict(api, tmp_path, field, value):
    fixture = build_replay(api, tmp_path / 'original')
    rewrite(fixture, lambda document: document['source_frames'][100]['identity']['observations'][0].__setitem__(field, value))
    fixture.digest = file_sha(fixture.path)
    with pytest.raises(ValueError):
        with open_source(fixture, tmp_path / 'decode'):
            pytest.fail('Malformed identity observation was admitted')


@pytest.mark.parametrize('identity', [{}, {'observations': [
    {'player_id': None, 'track_key': None, 'state': 'UNKNOWN'},
]}])
def test_empty_and_unknown_raw_identity_stays_unavailable(api, tmp_path, identity):
    fixture = build_replay(api, tmp_path / 'original')
    rewrite(fixture, lambda document: document['source_frames'][100].__setitem__('identity', identity))
    fixture.digest = file_sha(fixture.path)
    with open_source(fixture, tmp_path / 'decode') as source:
        assert source.source_frames[100][1] == identity
