"""Owner-authorized complete correction delivery after private owners close.

The detached descriptor is integrity evidence, not authentication. The caller
must supply its durable owner's authorization receipt and revalidate it, plus
cancellation, in ``before_commit``. Existing complete-generation CAS commits
the new database, result, media and reports together.
"""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
import tempfile

from src.api.analysis_generation import _file_record, _safe_path, resolve_generation
from src.api.analysis_resume import REQUIRED_PIPELINE_FILES, OPTIONAL_PIPELINE_FILES
from src.api.landing_zone_contract import validate_landing_zones
from src.api.point_timeline_contract import load_point_analysis
from src.api.result_document import ResultDocument, publish_result
from src.api.result_storage import ResultStorage
from src.api.reporting import readable_report_chunks, telemetry_csv_chunks
from src.scoring.corrected_exports import EXPORT_FILES, MANIFEST_NAME as EXPORT_MANIFEST, _copy, _object
from src.scoring.corrected_preparation import (
    PreparedCorrection, MANIFEST_NAME as PREPARATION_MANIFEST, REPORT_FILES,
    _json_digest, _same_rows, _write_report,
)
from src.scoring.corrected_rendering import MANIFEST_NAME as RENDER_MANIFEST, VIDEO_FILES, _reader
from src.scoring.corrected_rule_rebuild import _canonical, _digest, _request
from src.scoring.rule_reproduction import reproduce_rule_timeline
from src.utils.point_timeline_artifact import artifact_sha256


MANIFEST_NAME = 'corrected_delivery_manifest.json'
AUTHORIZATION_NAME = 'correction_authorization.json'
EXECUTION_NAME = 'correction_execution.json'
_SHA = re.compile(r'[0-9a-f]{64}\Z')
_AUTH_FIELDS = frozenset(('schema_version', 'analysis_id', 'owner_id', 'reviewer_id',
    'correction_id', 'request_sha256', 'authorized_at', 'authorization_method'))


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected publication: '+message)


def _small(path, maximum=64*1024):
    _safe_path(path)
    _require(path.is_file() and path.stat().st_size <= maximum, 'bounded metadata file is missing or oversized')
    payload = path.read_bytes()
    _require(len(payload) <= maximum, 'metadata grew beyond its bounded size')
    value = json.loads(payload)
    _require(type(value) is dict, 'metadata must be an object')
    return value


def _same(actual, expected, message):
    _require(_canonical(actual) == _canonical(expected), message)


@dataclass(frozen=True)
class CorrectionPublicationDescriptor:
    directory: Path
    manifest_sha256: str
    analysis_id: str
    base_generation: str
    analysis_revision: int
    source_sha256: str
    request_sha256: str
    correction_id: str
    reviewer_id: str


def _binding(descriptor):
    _require(isinstance(descriptor, CorrectionPublicationDescriptor), 'native detached descriptor is required')
    for value in (descriptor.manifest_sha256, descriptor.base_generation,
                  descriptor.source_sha256, descriptor.request_sha256):
        _require(type(value) is str and _SHA.fullmatch(value), 'descriptor hash binding is invalid')
    _require(type(descriptor.analysis_revision) is int and descriptor.analysis_revision > 0
             and type(descriptor.analysis_id) is str and bool(descriptor.analysis_id),
             'descriptor identity or revision is invalid')
    return {'analysis_id': descriptor.analysis_id, 'analysis_revision': descriptor.analysis_revision,
        'source_sha256': descriptor.source_sha256, 'base_generation': descriptor.base_generation,
        'request_sha256': descriptor.request_sha256}


def _prepared(descriptor):
    binding = _binding(descriptor)
    directory = _safe_path(descriptor.directory)
    _require(directory.is_dir(), 'private prepared directory is missing')
    seal = directory/PREPARATION_MANIFEST
    _require(artifact_sha256(seal) == descriptor.manifest_sha256, 'private preparation manifest changed')
    manifest = _small(seal)
    _require(all(manifest.get(key) == value for key, value in binding.items())
             and manifest.get('schema_version') == '1.0'
             and manifest.get('status') == 'PRIVATE_PENDING_PUBLICATION'
             and manifest.get('human_review_dependent') is True
             and manifest.get('independent_accuracy_validated') is False
             and manifest.get('published') is False, 'private preparation binding differs')
    records = manifest.get('artifacts')
    _require(type(records) is dict and len(records) <= 128, 'prepared artifact inventory is invalid')
    originals = {name for name in records if name.startswith('original.')}
    required = set(REQUIRED_PIPELINE_FILES) | set(EXPORT_FILES) | set(VIDEO_FILES) | set(REPORT_FILES)
    required |= {EXPORT_MANIFEST, RENDER_MANIFEST}
    allowed = required | set(OPTIONAL_PIPELINE_FILES) | originals | {'annotated_web_credit.json'}
    _require(len(originals) == 1 and required <= set(records) <= allowed
             and set(path.name for path in directory.iterdir()) == set(records) | {PREPARATION_MANIFEST},
             'private complete inventory is missing, foreign or mixed')
    for name, record in records.items():
        _require(type(name) is str and Path(name).name == name and name not in {'.', '..'}
                 and type(record) is dict and set(record) == {'bytes', 'sha256'}
                 and type(record['bytes']) is int and record['bytes'] >= 0
                 and type(record['sha256']) is str and _SHA.fullmatch(record['sha256']),
                 'prepared artifact name or metadata is invalid')
        _require(_file_record(directory/name) == record, 'prepared artifact bytes changed: '+name)
    _require(records['source.mp4']['sha256'] == descriptor.source_sha256,
             'prepared source does not match the correction')
    command = _request(_small(directory/'correction_request.json', maximum=1024*1024))
    _require(_digest(command) == descriptor.request_sha256
             and command['analysis_id'] == descriptor.analysis_id
             and command['base_generation'] == descriptor.base_generation
             and command['base_analysis_revision']+1 == descriptor.analysis_revision
             and command['source_sha256'] == descriptor.source_sha256
             and command['correction_id'] == descriptor.correction_id
             and command['reviewer_id'] == descriptor.reviewer_id, 'prepared command binding differs')
    for name, digest in ((EXPORT_MANIFEST, manifest.get('export_manifest_sha256')),
                         (RENDER_MANIFEST, manifest.get('render_manifest_sha256'))):
        _require(records[name]['sha256'] == digest, 'prepared producer seal binding differs')
    exported = _small(directory/EXPORT_MANIFEST)
    rendered = _small(directory/RENDER_MANIFEST)
    _require(set(exported.get('artifacts', {})) == set(EXPORT_FILES)
             and set(rendered.get('artifacts', {})) == set(VIDEO_FILES), 'producer artifact inventory differs')
    for producer in (exported, rendered):
        for key in ('analysis_id', 'analysis_revision', 'base_generation', 'source_sha256'):
            _require(producer.get(key) == binding[key], 'producer source/revision binding differs')
        for name, record in producer['artifacts'].items():
            _require(records.get(name) == record, 'prepared file conflicts with its producer seal: '+name)
    _require(rendered.get('export_manifest_sha256') == records[EXPORT_MANIFEST]['sha256'],
             'render belongs to another semantic export')
    return directory, manifest, command


def describe_prepared_correction(prepared):
    """Capture a detached integrity descriptor while the prepared owner is open."""
    _require(isinstance(prepared, PreparedCorrection), 'native prepared result is required')
    prepared.storage._require_open()
    manifest = _small(prepared.directory/PREPARATION_MANIFEST)
    command = _request(_small(prepared.directory/'correction_request.json', maximum=1024*1024))
    descriptor = CorrectionPublicationDescriptor(prepared.directory, prepared.manifest_sha256,
        manifest['analysis_id'], prepared.base_generation, prepared.analysis_revision,
        manifest['source_sha256'], manifest['request_sha256'], command['correction_id'], command['reviewer_id'])
    _, saved, _ = _prepared(descriptor)
    _require(set(prepared.files) == set(saved['artifacts']) | {PREPARATION_MANIFEST}
             and all(path == prepared.directory/name for name, path in prepared.files.items()),
             'prepared native file handles differ')
    expected = saved['artifacts']['report.json']['sha256']
    _require(_json_digest(prepared.storage, prepared.result) == expected
             == _json_digest(prepared.storage, prepared.storage.root()), 'prepared result differs from its report')
    return descriptor


def _authorization(descriptor, value):
    _require(isinstance(value, dict) and set(value) == _AUTH_FIELDS,
             'complete durable owner authorization receipt is required')
    result = json.loads(json.dumps(value, allow_nan=False, separators=(',', ':')))
    _require(len(_canonical(result).encode('utf-8')) <= 8192
             and type(result['schema_version']) is int and result['schema_version'] == 1,
             'authorization schema or bounded size differs')
    for key in _AUTH_FIELDS-{'schema_version'}:
        _require(type(result[key]) is str and result[key].strip() and len(result[key]) <= 512,
                 'authorization fields must be bounded nonblank strings')
    _require(result['analysis_id'] == descriptor.analysis_id
             and result['request_sha256'] == descriptor.request_sha256
             and result['correction_id'] == descriptor.correction_id
             and result['reviewer_id'] == result['owner_id'] == descriptor.reviewer_id
             and result['authorization_method'] in ('AUTHENTICATED_OWNER', 'EXPLICIT_LOCAL_OWNER'),
             'owner authorization does not match the reviewed command')
    stamp = datetime.fromisoformat(result['authorized_at'].replace('Z', '+00:00'))
    _require(stamp.tzinfo is not None, 'authorization timestamp must include its time zone')
    return result


def _execution(descriptor, manifest, value):
    default = {'schema_version': 1, 'new_vision_inference': False, 'pose_provider_calls': 0,
        'pose_prediction_calls': 0,
        'selected_player_checkpoint_sha256': None, 'selected_player_checkpoint_filename': None,
        'source_sha256': descriptor.source_sha256, 'base_generation': descriptor.base_generation,
        'analysis_revision': descriptor.analysis_revision,
        'run_config_sha256': manifest['artifacts']['run_config.yaml']['sha256'],
        'runtime_versions': None, 'input_spec': None, 'architecture': None,
        'independent_accuracy_validated': False}
    if value is None:
        return default
    _require(type(value) is dict and set(value) == set(default), 'execution receipt fields differ')
    encoded = json.dumps(value, allow_nan=False, separators=(',', ':'))
    _require(len(encoded.encode('utf-8')) <= 64*1024, 'execution receipt exceeds bounded size')
    result = json.loads(encoded)
    for key in ('schema_version', 'source_sha256', 'base_generation', 'analysis_revision',
                'run_config_sha256', 'independent_accuracy_validated'):
        _require(type(result[key]) is type(default[key]) and result[key] == default[key],
                 'execution source/revision/configuration binding differs')
    _require(type(result['new_vision_inference']) is bool
             and type(result['pose_provider_calls']) is int and result['pose_provider_calls'] >= 0
             and type(result['pose_prediction_calls']) is int and result['pose_prediction_calls'] >= 0
             and result['new_vision_inference'] is (result['pose_prediction_calls'] > 0)
             and (result['pose_prediction_calls'] == 0 or result['pose_provider_calls'] > 0),
             'execution inference counters are invalid')
    if result['pose_provider_calls'] > 0:
        _require(type(result['selected_player_checkpoint_sha256']) is str
                 and _SHA.fullmatch(result['selected_player_checkpoint_sha256'])
                 and type(result['selected_player_checkpoint_filename']) is str
                 and Path(result['selected_player_checkpoint_filename']).name == result['selected_player_checkpoint_filename']
                 and all(result[key] is not None for key in ('runtime_versions', 'input_spec', 'architecture')),
                 'invoked pose provider lacks actual selected model/runtime evidence')
    else:
        _require(all(result[key] is None for key in (
            'selected_player_checkpoint_sha256', 'selected_player_checkpoint_filename',
            'runtime_versions', 'input_spec', 'architecture')), 'unused pose branch cannot claim model execution')
    return result


def _committed(descriptor, directory, authorization):
    snapshot = resolve_generation(directory)
    if snapshot is None or snapshot.analysis_revision != descriptor.analysis_revision:
        return None
    with ResultDocument(directory, descriptor.analysis_id, generation=snapshot.generation) as document:
        correction = document.value.get('analysisCorrection')
        if (type(correction) is dict and correction.get('requestSha256') == descriptor.request_sha256
                and correction.get('baseGeneration') == descriptor.base_generation
                and correction.get('sourceSha256') == descriptor.source_sha256
                and correction.get('privatePreparationManifestSha256') == descriptor.manifest_sha256
                and correction.get('authorizationReceipt') == authorization
                and snapshot.file_sha256('source.mp4') == descriptor.source_sha256):
            return snapshot.generation
    return None


def _base(descriptor, directory):
    snapshot = resolve_generation(directory)
    _require(snapshot is not None and snapshot.generation == descriptor.base_generation
             and snapshot.analysis_id == descriptor.analysis_id
             and snapshot.analysis_revision+1 == descriptor.analysis_revision
             and snapshot.file_sha256('source.mp4') == descriptor.source_sha256,
             'expected current base generation/source/revision differs')
    return snapshot


def _forecast(result, native, replay, descriptor):
    evidence = result.get('nativeForecast')
    _require(type(evidence) is dict and evidence.get('status') == 'UNQUALIFIED'
             and evidence.get('calibrationStatus') == 'not_calibrated'
             and evidence.get('independentAccuracyValidated') is False
             and evidence.get('humanReviewDependent') is True, 'native forecast qualification fields differ')
    for key, value in (('analysisId', descriptor.analysis_id), ('analysisRevision', descriptor.analysis_revision),
                       ('baseGeneration', descriptor.base_generation), ('sourceSha256', descriptor.source_sha256)):
        _require(evidence.get(key) == value, 'native forecast source/revision binding differs')
    _same(evidence['current'], native['current'], 'native current forecast differs from sealed producer')
    _require(evidence.get('modelSignature') == native['audit']['model_signature'],
             'native forecast model signature differs')
    _same(evidence.get('modelSpec'), native['audit']['model_spec'], 'native forecast model specification differs')
    sentinel = object()
    with _reader(evidence['history']) as actual, _reader(native['history']) as expected:
        for left, right in zip_longest(actual, expected, fillvalue=sentinel):
            _require(left is not sentinel and right is not sentinel, 'native forecast history count differs')
            _same(left, right, 'native forecast history differs from sealed producer')
    history = evidence['history']
    _require(len(history) == 1+2*len(replay.owner.rows('replay_inputs'))
             and history[0]['phase'] == 'INITIAL' and history[0]['event_id'] is None,
             'native causal history cadence differs')
    for index, row in enumerate(replay.owner.rows('replay_inputs')):
        transition = row['accepted_transition']
        for phase, position in (('BEFORE_EVENT', 1+index*2), ('AFTER_EVENT', 2+index*2)):
            record = history[position]
            context = transition['context']
            _require(record['phase'] == phase and record['event_id'] == transition['event_id']
                     and record['event_uid'] == transition['uid'] and record['point_uid'] == transition['point_uid']
                     and record['frame_index'] == transition['frame']
                     and record['timestamp_seconds'] == transition['timestamp_s']
                     and record['source_pts'] == context['source_pts']
                     and record['source_time_base'] == context['source_time_base']
                     and record['source_frame_index'] == context['source_frame_index'],
                     'native forecast causal source/event join differs')
    _same(history[-1]['forecast'], evidence['current'], 'native final forecast differs from history')


def _validate_result(storage, directory, result, descriptor):
    rendered = _small(directory/RENDER_MANIFEST)
    _require(type(result) is dict and result.get('analysisRevision') == descriptor.analysis_revision
             and result['summary']['id'] == descriptor.analysis_id
             and type(result['summary']['frames']) is int and result['summary']['frames'] == rendered['frames']
             and result['summary']['fps'] == rendered['fps']
             and len(result['detections']['frames']) == rendered['frames']
             and len(result['trajectories']['ballTrajectory']) == rendered['frames'],
             'normalized result identity or frame binding differs')
    metrics = storage.read_json(directory/'metrics.json')
    authority = metrics.get('authoritative_events_enabled')
    _require(type(authority) is bool, 'event authority must be explicit')
    points = load_point_analysis(directory, descriptor.analysis_id,
        frame_count=result['summary']['frames'], authoritative=authority, storage=storage)
    _require(_json_digest(storage, points) == _json_digest(storage, result.get('pointAnalysis')),
             'normalized point evidence differs from strict native evidence')
    landings = validate_landing_zones(storage.read_json(directory/'landing_zones.json',
        indexed_root_objects=('calibrations',)), descriptor.analysis_id, result['events'],
        frame_calibrations=result['cameraTracking']['frames'], expected_revision=descriptor.analysis_revision)
    _same_rows(landings, result.get('landingZones', ()))
    with reproduce_rule_timeline(directory/'native_rule_replay.json', directory.parent,
            artifact_sha256=artifact_sha256(directory/'native_rule_replay.json'),
            source_sha256=descriptor.source_sha256, analysis_id=descriptor.analysis_id,
            analysis_revision=descriptor.analysis_revision) as replay:
        _require(replay.authoritative_events_enabled is authority, 'native replay authority differs')
        _forecast(result, storage.read_json(directory/'corrected_forecasts.json'), replay, descriptor)


def publish_corrected_result(descriptor, analysis_directory, *, authorization, before_commit=None, execution=None):
    """Publish one authorized complete revision, or return its committed retry.

    The caller finishes all private contexts before this phase and supplies
    ``before_commit`` from its serialized durable authorization/cancel guard.
    A lost publisher acknowledgement is reconciled against the actual current
    generation, never by trusting an incomplete delivery directory.
    """
    _binding(descriptor)
    authorization = _authorization(descriptor, authorization)
    directory = _safe_path(Path(analysis_directory).absolute())
    _require(directory.is_dir() and (before_commit is None or callable(before_commit)),
             'analysis directory or before_commit callback is invalid')
    already = _committed(descriptor, directory, authorization)
    if already is not None:
        return already
    _base(descriptor, directory)
    private, manifest, command = _prepared(descriptor)
    execution = _execution(descriptor, manifest, execution)
    _require(not private.is_relative_to(directory) and not directory.is_relative_to(private),
             'private preparation must be separate from the analysis directory')
    stage = Path(tempfile.mkdtemp(prefix='corrected-delivery-', dir=private.parent))
    seal = stage/MANIFEST_NAME
    try:
        records = {name: dict(record) for name, record in manifest['artifacts'].items() if name not in REPORT_FILES}
        for name, record in records.items():
            _require(_file_record(private/name) == record, 'private source artifact changed before delivery copy')
            _copy(private/name, stage/name, record['sha256'])
        with ResultStorage(directory) as storage:
            result = storage.read_json(private/'report.json')
            _validate_result(storage, stage, result, descriptor)
            if execution['pose_provider_calls'] > 0:
                selected = (result.get('provenance') or {}).get('checkpointHashes', {}).get('players', {})
                _require(selected.get('sha256') == execution['selected_player_checkpoint_sha256']
                         and selected.get('filename') == execution['selected_player_checkpoint_filename'],
                         'execution checkpoint differs from original selected-player inference provenance')
            correction = result['analysisCorrection']
            _require(correction.get('requestSha256') == descriptor.request_sha256
                     and correction.get('review', {}).get('correctionId') == descriptor.correction_id,
                     'private correction metadata differs from command')
            correction.pop('published', None)
            correction.update(status='OWNER_REVIEWED_RESULT',
                authorizationStatus='OWNER_AUTHORIZED' if authorization['authorization_method'] == 'AUTHENTICATED_OWNER'
                    else 'EXPLICIT_LOCAL_OWNER',
                authorizationReceipt=authorization, publicationPolicy='ATOMIC_COMPLETE_GENERATION',
                privatePreparationManifestSha256=descriptor.manifest_sha256, execution=execution)
            result['summary']['artifacts'] = sorted(set(records) | set(REPORT_FILES) | {AUTHORIZATION_NAME, EXECUTION_NAME, MANIFEST_NAME})
            _object(stage/AUTHORIZATION_NAME, authorization)
            records[AUTHORIZATION_NAME] = _file_record(stage/AUTHORIZATION_NAME)
            _object(stage/EXECUTION_NAME, execution)
            records[EXECUTION_NAME] = _file_record(stage/EXECUTION_NAME)

            def artifacts():
                for name, expected in records.items():
                    _require(_file_record(stage/name) == expected, 'delivery artifact bytes changed: '+name)

            artifacts()
            records['report.json'] = _write_report(stage/'report.json', storage.json_chunks(result))
            for name, chunks in (('readable_report.html', readable_report_chunks(result, stage)),
                                 ('telemetry.csv', telemetry_csv_chunks(result))):
                records[name] = _write_report(stage/name, chunks)
                artifacts()
            _require(_json_digest(storage, result) == records['report.json']['sha256'], 'delivery result changed during reports')
            _object(seal, {**_binding(descriptor), 'schema_version': '1.0', 'status': 'OWNER_REVIEWED_RESULT',
                'publication_policy': 'ATOMIC_COMPLETE_GENERATION',
                'private_preparation_manifest_sha256': descriptor.manifest_sha256,
                'authorization_sha256': records[AUTHORIZATION_NAME]['sha256'], 'artifacts': records,
                'human_review_dependent': True, 'independent_accuracy_validated': False})
            records = {**records, MANIFEST_NAME: _file_record(seal)}
            storage.set_root(result)
            summary_sha256 = _json_digest(storage, result['summary'])

            def final_check():
                _base(descriptor, directory)
                _prepared(descriptor)
                artifacts()
                # publish_result closes this storage into its UUID database
                # before invoking us. Verify the copied candidate, too; a
                # transient source change must not become a new sealed truth.
                candidate = directory/'.result'/(storage._path.stem+'.files')
                database_record = _file_record(storage._path)

                def candidate_files():
                    _require(candidate.is_dir(), 'complete candidate files are missing')
                    _require(set(path.name for path in candidate.iterdir()) == set(records) | {'result.json', 'summary.json'},
                             'complete candidate inventory differs')
                    for name, expected in records.items():
                        _require(_file_record(candidate/name) == expected, 'complete candidate copy differs: '+name)
                    _require(artifact_sha256(candidate/'result.json') == records['report.json']['sha256'],
                             'complete candidate result differs from delivery report')
                    _require(artifact_sha256(candidate/'summary.json') == summary_sha256,
                             'complete candidate summary differs from delivery result')
                    _require(_file_record(storage._path) == database_record,
                             'complete candidate database changed during final authorization')

                candidate_files()
                if before_commit is not None:
                    before_commit()
                _base(descriptor, directory)
                artifacts()
                candidate_files()

            final_check_before_copy = lambda: (_base(descriptor, directory), _prepared(descriptor), artifacts())
            final_check_before_copy()
            return publish_result(storage, result, directory,
                generation_files={name: stage/name for name in records},
                analysis_revision=descriptor.analysis_revision,
                expected_generation=descriptor.base_generation, before_commit=final_check)
    except BaseException as error:
        committed = _committed(descriptor, directory, authorization)
        if committed is not None:
            return committed
        try:
            seal.unlink(missing_ok=True)
        except BaseException as cleanup:
            error.add_note('Delivery seal cleanup also failed: '+str(cleanup))
        raise
