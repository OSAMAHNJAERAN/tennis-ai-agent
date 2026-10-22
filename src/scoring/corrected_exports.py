"""Complete private semantic exports from one live native correction.

The final seal identifies this scratch bundle only. Rendering, authenticated
commands, normalized reports and complete-generation publication follow later.
No old point, shot, forecast, line-call or landing output is copied here.
"""
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType

from src.scoring.corrected_forecasts import (
    _admit, _ForecastResult, _model_spec, write_corrected_forecasts,
)
from src.scoring.corrected_rule_rebuild import (
    _canonical, _CorrectedRuleResult, _digest, _request, _scratch,
)
from src.scoring.corrected_shot_rebuild import _current
from src.scoring.corrected_landing_exports import write_corrected_landing_zones
from src.scoring.match_state_codec import encode_match_state
from src.scoring.point_timeline import native_match_analytics
from src.scoring.rule_replay import write_rule_replay
from src.utils.analytics_json_artifact import (
    _destination, write_match_analytics, write_membership_analytics,
)
from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.point_timeline_artifact import (
    artifact_sha256, publish_point_timeline_manifest,
)


MANIFEST_NAME = 'corrected_export_manifest.json'
EXPORT_FILES = frozenset((
    'point_analytics.json', 'rallies.json', 'match_analytics.json',
    'serve_executions.json', 'shot_play_scope.json', 'scoring_events.json',
    'score_history.json', 'match_state.json', 'shot_events.json',
    'line_calls.json', 'match_events.json', 'metrics.json',
    'native_rule_replay.json', 'point_timeline_manifest.json',
    'corrected_forecasts.json', 'correction_request.json',
    'correction_audit.json', 'source.mp4', 'landing_zones.json',
))


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected exports: ' + message)


def _same(actual, expected, message):
    _require(_canonical(actual) == _canonical(expected), message)


def _object(destination, value):
    with _destination(destination) as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def _copy(source, destination, expected):
    """Independent bounded copy; a mutable media alias must never share an inode."""
    _require(source.is_file() and not source.is_symlink(), 'source artifact is missing')
    size = source.stat().st_size
    digest = hashlib.sha256()
    with source.open('rb') as reader, destination.open('xb') as writer:
        while chunk := reader.read(1024*1024):
            digest.update(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    _require(digest.hexdigest() == expected == artifact_sha256(source)
             and source.stat().st_size == destination.stat().st_size == size
             and destination.stat().st_nlink == 1, 'source artifact bytes/hash changed during copy')


def _admission(rebuilt, shots, forecast, scratch):
    from src.utils.event_analysis_sequence import EventAnalysisSequence
    from src.utils.forecast_sequence import ForecastSequence
    _require(isinstance(rebuilt, _CorrectedRuleResult), 'native corrected rule owner is required')
    _admit(rebuilt, shots)
    _require(isinstance(shots.event_owner, EventAnalysisSequence)
             and shots.events is shots.event_owner.events,
             'fresh physical event owner and exact event alias are required')
    shots.event_owner._open()
    _require(isinstance(forecast, _ForecastResult)
             and isinstance(forecast.owner, ForecastSequence)
             and forecast._rebuilt is rebuilt and forecast._shot_analysis is shots
             and forecast.snapshot == rebuilt.snapshot
             and forecast.base_generation == rebuilt.base_generation
             and forecast.history is forecast.owner.history,
             'native bound forecast owners/source/generation differ')
    forecast.owner._open()
    _require(forecast.owner.sealed is True, 'forecast owner must be sealed')
    directory = _scratch(scratch, rebuilt.snapshot.files_directory.parent.parent.resolve())
    _require(directory == forecast.scratch, 'destination must use the guarded forecast scratch')
    owner, snapshot = rebuilt.owner, rebuilt.snapshot
    command = _request(rebuilt.request)
    binding = {'analysis_id': owner.analysis_id, 'analysis_revision': owner.revision,
        'source_sha256': owner.source_fingerprint, 'base_generation': snapshot.generation}
    _same({key: command[key] for key in ('analysis_id', 'source_sha256', 'base_generation')},
          {key: binding[key] for key in ('analysis_id', 'source_sha256', 'base_generation')},
          'admitted command belongs to a different source/generation')
    _require(command['base_analysis_revision'] == snapshot.analysis_revision
             and command['base_rule_replay_sha256'] == snapshot.file_sha256('native_rule_replay.json'),
             'admitted command base revision/source differs')
    _same(forecast.current, forecast.history[-1]['forecast'], 'current forecast differs from owned history')
    for row, state in ((forecast.history[0], rebuilt.engine.initial_state),
                       (forecast.history[-1], rebuilt.engine.get_current_state())):
        _same({key: row[key] for key in binding}, binding, 'forecast history source binding differs')
        _require(row['raw_state_sha256'] == _digest(encode_match_state(state)),
                 'forecast history raw state differs')
    spec = _model_spec()
    expected_audit = {**binding, 'schema_version': 1, 'status': 'PRIVATE_PENDING_EXPORTS_AND_RENDER',
        'accepted_input_count': len(owner.rows('transitions')), 'history_count': len(forecast.history),
        'posterior_point_count': len(forecast.owner.point_index),
        'withheld_reason_code': forecast.current.get('withheld_reason_code'),
        'model_signature': _digest(spec), 'model_spec': spec,
        'calibration_status': 'not_calibrated', 'old_forecasts_reused': False,
        'history_policy': 'INITIAL_AND_BEFORE_AFTER_EVERY_ACCEPTED_EVENT',
        'point_update_policy': 'NEW_ANCHORED_COMPLETED_POINT_AT_TERMINAL_EVENT_PRE_AWARD_SERVER',
        'human_review_dependent': True, 'independent_accuracy_validated': False, 'published': False}
    _same(forecast.audit, expected_audit, 'forecast audit differs from its bound native owners')
    _require(len(forecast.history) == 1+2*len(owner.rows('transitions')),
             'forecast history count differs from accepted inputs')
    return directory, command, binding


def _physical_events(events):
    # Match the native physical-event wire without importing the model pipeline.
    with closing(iter(events)) as rows:
        for event in rows:
            yield {'event_id': event.event_id, 'event_type': event.event_type.value,
                'frame': event.frame_index, 'timestamp_s': event.timestamp_s,
                'player_id': event.player_id,
                'ball_position_px': [round(value, 2) for value in event.ball_position_px]
                    if event.ball_position_px is not None else None,
                'court_position_m': list(event.court_position_m) if event.court_position_m is not None else None,
                'confidence': round(event.confidence, 3),
                'trajectory_state': event.trajectory_state, 'evidence': event.evidence}


class _CameraRecords(Sequence):
    """Independent frozen source frame registrations for the display validator."""
    def __init__(self, sources):
        self.sources = sources

    def __len__(self):
        return self.sources.frame_count

    def __getitem__(self, index):
        self.sources._index(index)
        row = self.sources._camera['frames'][index] if self.sources._camera['enabled'] else {}
        return {'frame_index': index, 'isValid': row.get('is_valid') is True,
                'imageToCourt': row.get('image_to_court')}


@dataclass(frozen=True)
class CorrectedExportResult:
    directory: Path
    files: object
    manifest_sha256: str


def write_corrected_exports(rebuilt, shot_analysis, forecast, scratch):
    """Regenerate and strictly validate a fresh uniquely named private bundle.

    All native owners must remain live. Successful earlier bundles are never
    overwritten. Failed candidates retain diagnostics without a final seal;
    the immutable analysis and its authoritative current pointer are untouched.
    """
    from src.api.landing_zone_contract import validate_landing_zones
    from src.api.point_timeline_contract import load_point_analysis
    from src.api.result_storage import ResultStorage
    from src.scoring.rule_reproduction import reproduce_rule_timeline
    directory, command, binding = _admission(rebuilt, shot_analysis, forecast, scratch)
    owner, engine = rebuilt.owner, rebuilt.engine
    authoritative = rebuilt.authoritative_events_enabled
    stage = Path(tempfile.mkdtemp(prefix='corrected-exports-', dir=directory))
    seal_path = stage / MANIFEST_NAME
    try:
        source = rebuilt.snapshot.resolve_file('source.mp4')
        _copy(source, stage/'source.mp4', owner.source_fingerprint)
        landing = write_corrected_landing_zones(rebuilt, shot_analysis, stage)
        # Recompute aggregates from sealed membership and fresh shots. Mutable
        # caller presentation dictionaries are never export authority.
        analytics = native_match_analytics(owner, shot_analysis.shots, engine.get_current_state(),
                                           authoritative=authoritative)
        for name, kind in (('point_analytics.json', 'points'), ('rallies.json', 'rallies')):
            write_membership_analytics(stage/name, kind, owner.rows(kind), lambda value: value,
                                       schema_version='2.0')
        write_match_analytics(stage/'match_analytics.json', analytics, lambda value: value)
        for name, key, rows, metadata in (
            ('serve_executions.json', 'serve_executions', owner.rows('attempts'),
                {'schema_version': '2.0', 'analysis_revision': owner.revision}),
            ('shot_play_scope.json', 'shots', owner.member_rows(),
                {'schema_version': '2.0', 'analysis_revision': owner.revision}),
            ('scoring_events.json', 'scoring_events', owner.rows('transitions'), {'schema_version': '2.0'}),
            ('shot_events.json', 'shot_events', (shot.to_dict() for shot in shot_analysis.shots),
                {'schema_version': '1.0'}),
            ('line_calls.json', 'line_calls', (call.to_dict() for call in engine.native_line_calls()),
                {'schema_version': '1.0'}),
            ('match_events.json', 'events', _physical_events(shot_analysis.events),
                {'schema_version': '1.0', 'analysis_id': owner.analysis_id,
                 'analysis_revision': owner.revision, 'events_count': len(shot_analysis.events)})):
            write_json_array_artifact(stage/name, metadata, key, rows, allow_nan=False)
        engine.save_history_json(stage/'score_history.json')
        _object(stage/'match_state.json', {'schema_version': '1.0', **engine.state.to_dict(),
            'authoritative_events_enabled': authoritative,
            'semantic_role': 'INFERRED_SCORING_STATE' if authoritative else
                'CONFIGURED_INITIAL_STATE_NOT_OBSERVED_MATCH_SCORE'})
        _object(stage/'metrics.json', {'schema_version': '1.0', **binding,
            'total_frames': rebuilt.snapshot.frames, 'shots_linked': len(shot_analysis.shots),
            'rallies': len(owner.rows('rallies')), 'authoritative_events_enabled': authoritative,
            'production_qualified': False, 'processing_fps': None, 'total_time_s': None,
            'timing_scope': 'SEMANTIC_CORRECTION_REUSES_SEALED_OBSERVATIONS; NO_NEW_INFERENCE_THROUGHPUT',
            'original_inference_generation': rebuilt.snapshot.generation,
            'original_inference_metrics_sha256': rebuilt.snapshot.file_sha256('metrics.json')})
        write_rule_replay(stage/'native_rule_replay.json', engine, owner,
            rebuilt.sources.source_frames, authoritative_events_enabled=authoritative)
        publish_point_timeline_manifest(stage, analysis_id=owner.analysis_id,
            analysis_revision=owner.revision, source_sha256=owner.source_fingerprint,
            scoring_revision=engine.revision, authoritative_events_enabled=authoritative)
        private_forecast = write_corrected_forecasts(directory/'corrected_forecasts.json', forecast)
        _copy(private_forecast, stage/'corrected_forecasts.json', artifact_sha256(private_forecast))
        _object(stage/'correction_request.json', command)
        landing.pop('path')
        _object(stage/'correction_audit.json', {**binding, 'schema_version': 1,
            'status': 'PRIVATE_PENDING_RENDER_AND_PUBLICATION', 'request_sha256': _digest(command),
            'scoring_revision': engine.revision,
            'initial_raw_state': encode_match_state(engine.initial_state),
            'final_raw_state': encode_match_state(engine.state),
            'accepted_input_count': len(owner.rows('transitions')),
            'physical_event_count': len(shot_analysis.events), 'shot_count': len(shot_analysis.shots),
            'completed_point_count': len(owner.rows('points', status='COMPLETED')),
            'landing': landing, 'forecast': forecast.audit,
            'old_semantic_exports_reused': False, 'human_review_dependent': True,
            'selected_checkpoint_execution_verified': False,
            'independent_accuracy_validated': False, 'published': False})
        with ResultStorage(directory) as storage:
            proof = load_point_analysis(stage, owner.analysis_id, frame_count=rebuilt.snapshot.frames,
                authoritative=authoritative, storage=storage)
            _require(proof is not None and proof['analysisRevision'] == owner.revision,
                     'strict native point validation did not establish the corrected revision')
            raw_landing = storage.read_json(stage/'landing_zones.json', indexed_root_objects=('calibrations',))
            events = storage.read_json(stage/'match_events.json')['events']
            validate_landing_zones(raw_landing, owner.analysis_id, events,
                frame_calibrations=_CameraRecords(rebuilt.sources), expected_revision=owner.revision)
        # This is unchanged reproduction of the NEW replay, never use of an old
        # generation's recorded decisions to calculate corrected semantics.
        with reproduce_rule_timeline(stage/'native_rule_replay.json', directory,
                artifact_sha256=artifact_sha256(stage/'native_rule_replay.json'),
                source_sha256=owner.source_fingerprint, analysis_id=owner.analysis_id,
                analysis_revision=owner.revision):
            pass
        _admission(rebuilt, shot_analysis, forecast, scratch)
        _require({path.name for path in stage.iterdir()} == EXPORT_FILES,
                 'private export inventory differs')
        artifacts = {name: {'sha256': artifact_sha256(stage/name), 'bytes': (stage/name).stat().st_size}
                     for name in sorted(EXPORT_FILES)}
        seal = {**binding, 'schema_version': '1.0', 'scoring_revision': engine.revision,
            'status': 'PRIVATE_PENDING_RENDER_AND_PUBLICATION', 'request_sha256': _digest(command),
            'artifacts': artifacts, 'human_review_dependent': True,
            'independent_accuracy_validated': False, 'published': False}
        _object(seal_path, seal)
        _current(rebuilt)
        _require(all(artifact_sha256(stage/name) == metadata['sha256']
            and (stage/name).stat().st_size == metadata['bytes'] for name, metadata in artifacts.items()),
            'private export artifact changed before return')
        return CorrectedExportResult(stage, MappingProxyType({name: stage/name
            for name in sorted(EXPORT_FILES | {MANIFEST_NAME})}), artifact_sha256(seal_path))
    except BaseException as error:
        try:
            seal_path.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            error.add_note('Private export seal cleanup also failed: '+str(cleanup_error))
        raise
