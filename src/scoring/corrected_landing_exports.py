"""Fresh display landings from corrected accepted contacts and frame geometry.

This private export never qualifies contact or camera accuracy. In particular,
the line caller's confidence and engineering uncertainty cannot supply missing
human/producer ground-contact qualification or an explicit contact uncertainty.
"""
from contextlib import closing
import math
from pathlib import Path

from src.scoring.corrected_forecasts import _admit
from src.scoring.corrected_rule_rebuild import _canonical, _digest, _scratch
from src.scoring.corrected_shot_rebuild import _current, MAX_REQUEST_BYTES


def _require(condition, message):
    if not condition:
        raise ValueError('Corrected landing export: ' + message)


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _settings(snapshot):
    import yaml
    from src.visualization.landing_zones import partition_zones
    path = snapshot.resolve_file('run_config.yaml')
    _require(path is not None and path.stat().st_size <= MAX_REQUEST_BYTES,
             'sealed bounded run configuration is required')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    _require(type(config) is dict, 'configuration must be an object')
    visualization = config.get('visualization', {})
    _require(type(visualization) is dict, 'visualization settings must be an object')
    settings = visualization.get('landing_zones', {})
    _require(type(settings) is dict, 'landing settings must be an object')
    enabled = settings.get('enabled', False)
    partition = settings.get('partition', 'tactical_3x3')
    duration = settings.get('duration_seconds', .8)
    _require(type(enabled) is bool and type(partition) is str,
             'landing enabled/partition types differ')
    partition_zones(partition)
    _require(type(duration) in (int, float) and math.isfinite(duration) and .6 <= duration <= 1.,
             'landing duration must be between .6 and 1 second')
    return enabled, partition, duration


def _qualified(evidence):
    if type(evidence) is not dict:
        return False
    radius, source = evidence.get('position_uncertainty_m'), evidence.get('ground_contact_source')
    return (evidence.get('review_status') in ('confirmed', 'corrected')
            and evidence.get('active_play') is True
            and evidence.get('position_type') == 'ground_contact'
            and type(source) is str and bool(source.strip())
            and type(radius) in (int, float) and math.isfinite(radius) and radius >= 0)


def write_corrected_landing_zones(rebuilt, shot_analysis, directory):
    """Write one fresh revision's private ``landing_zones.json`` and summary.

    The caller keeps both correction contexts open and owns a separate scratch
    directory. Fresh events receive current display bindings for the subsequent
    event/report export. Fault-triggering contacts remain descriptive landings;
    events already classified DEAD_BALL_IGNORED by the corrected ledger do not.
    No prior landing document, registry, or display coordinates are consumed.
    """
    from src.scoring.rule_replay_source import _source_event
    from src.utils.event_analysis_sequence import EventAnalysisSequence
    from src.utils.landing_zone_sequence import LandingZoneSequence
    from src.visualization.landing_zones import landing_record
    import numpy as np

    _admit(rebuilt, shot_analysis)
    snapshot, owner = rebuilt.snapshot, rebuilt.owner
    event_owner = shot_analysis.event_owner
    _require(isinstance(event_owner, EventAnalysisSequence)
             and shot_analysis.events is event_owner.events,
             'fresh physical event owner and exact sequence alias are required')
    event_owner._open()
    directory = _scratch(directory, snapshot.files_directory.parent.parent.resolve())
    destination = directory / 'landing_zones.json'
    _require(not destination.exists() and not destination.is_symlink()
             and not getattr(destination, 'is_junction', lambda: False)(),
             'landing destination must be an unused private scratch file')
    enabled, partition, duration = _settings(snapshot)
    summary = {'schema_version': 1, 'status': 'PRIVATE_CORRECTED_LANDING_EXPORT',
        'analysis_id': owner.analysis_id, 'analysis_revision': owner.revision,
        'source_sha256': owner.source_fingerprint, 'base_generation': rebuilt.base_generation,
        'enabled': enabled, 'physical_event_count': 0, 'record_count': 0,
        'calibration_count': 0, 'nonbounce_count': 0, 'dead_contact_count': 0,
        'unqualified_contact_count': 0, 'unavailable_geometry_count': 0,
        'unavailable_position_count': 0, 'disabled_contact_count': 0,
        'qualification_source': 'UNCHANGED_ACCEPTED_SOURCE_GROUND_CONTACT_EVIDENCE',
        'position_source': 'NEW_ACCEPTED_LINE_CALL_AND_CURRENT_EVENT_FRAME_HOMOGRAPHY',
        'fault_contact_policy': 'DESCRIPTIVE_CONTACT_RETAINED_UNLESS_CORRECTED_DEAD',
        'old_landings_reused': False, 'human_review_dependent': True,
        'independent_accuracy_validated': False, 'published': False}
    dead = owner.dead_event_ids()
    with LandingZoneSequence(directory, owner.analysis_id, owner.revision) as output:
        with closing(iter(owner.rows('replay_inputs'))) as rows:
            for row in rows:
                source = row['source_input']
                _require(row['input_fingerprint'] == _digest(source), 'source fingerprint differs')
                original, context = _source_event(source, rebuilt.sources.source_frames,
                    owner.analysis_id, owner.source_fingerprint)
                if source['reviewed_evidence_ref'] is not None:
                    continue
                index = summary['physical_event_count']
                _require(index < len(shot_analysis.events), 'fresh physical event is missing')
                event = shot_analysis.events[index]
                summary['physical_event_count'] += 1
                _require(type(event.event_id) is int and event.event_id == original.event_id
                    and type(event.frame_index) is int and event.frame_index == original.frame_index
                    and _canonical(event.timestamp_s) == _canonical(original.timestamp_s)
                    and event.event_type.value == original.event_type.value
                    and _canonical(event.player_id) == _canonical(original.player_id)
                    and _canonical(event.ball_position_px) == _canonical(original.ball_position_px)
                    and _canonical(event.confidence) == _canonical(original.confidence)
                    and _canonical(event.trajectory_state) == _canonical(original.trajectory_state),
                    'fresh physical event/source join differs')
                # Never keep old revision's display scope. Ground qualification
                # below comes from immutable source, not this mutable snapshot.
                _require(type(original.evidence) is dict, 'physical contact evidence must be an object')
                evidence = dict(original.evidence)
                for key in ('analysis_id', 'analysis_revision', 'calibration_id',
                            'contact_position_px', 'line_call'):
                    evidence.pop(key, None)
                event.evidence = evidence
                if event.event_type.value != 'BOUNCE':
                    _require(event.court_position_m is None,
                             'nonbounce court positions must remain unavailable')
                    summary['nonbounce_count'] += 1
                    event_owner.events[index] = event
                    continue
                call = row['scoring_input'].get('line_call')
                _require(type(call) is dict and type(call.get('event_id')) is int
                    and call['event_id'] == event.event_id and type(call.get('bounce_frame')) is int
                    and call['bounce_frame'] == event.frame_index,
                    'fresh bounce call/source frame differs')
                _require(_canonical(event.court_position_m) == _canonical(call.get('bounce_position_m')),
                         'fresh bounce position differs from its accepted corrected call')
                event.evidence['line_call'] = {'IN': 'in', 'SERVE_IN': 'in',
                    'OUT': 'out', 'SERVE_FAULT': 'out'}.get(call.get('decision'), 'uncertain')
                reason = None
                if event.event_id in dead:
                    reason = 'dead_contact_count'
                    event.evidence['active_play'] = False
                elif not enabled:
                    reason = 'disabled_contact_count'
                elif not _qualified(evidence):
                    reason = 'unqualified_contact_count'
                geometry = None
                if reason is None:
                    geometry = rebuilt.sources.event_geometry(event.frame_index, context)
                    if not geometry.usable or row['rule_environment'].get('geometry_available') is not True:
                        reason = 'unavailable_geometry_count'
                if reason is None:
                    position, pixel = call.get('bounce_position_m'), call.get('bounce_position_px')
                    if not _point(position) or not _point(pixel):
                        reason = 'unavailable_position_count'
                    else:
                        mapped = geometry.homography_matrix @ np.array([*pixel, 1.])
                        _require(np.isfinite(mapped).all() and abs(mapped[2]) >= 1e-10
                            and np.allclose(mapped[:2]/mapped[2], position, rtol=0, atol=1e-7),
                            'accepted contact pixels and current frame mapping disagree')
                if reason is not None:
                    summary[reason] += 1
                    event_owner.events[index] = event
                    continue
                calibration_id = f'frame_{event.frame_index}'
                # landing_record recomputes complete descriptive polygons and
                # the inverse-projected contact from the fresh metric point.
                for kind in dict.fromkeys(('service', partition)):
                    record = landing_record({'event_id': event.event_id,
                        'event_type': 'BOUNCE', 'timestamp_s': event.timestamp_s,
                        'court_position_m': event.court_position_m, 'evidence': event.evidence},
                        geometry.homography_matrix, analysis_id=owner.analysis_id,
                        revision=owner.revision, calibration_id=calibration_id,
                        partition=kind, duration_seconds=duration)
                    _require(record is not None, 'admitted contact did not produce a landing')
                    event.evidence.update(analysis_id=owner.analysis_id, analysis_revision=owner.revision,
                        calibration_id=calibration_id, contact_position_px=record['contact_position_px'])
                    output.append(record, {'analysis_id': owner.analysis_id,
                        'analysis_revision': owner.revision, 'frame_index': event.frame_index,
                        'image_to_court': geometry.homography_matrix.tolist()})
                    summary['record_count'] += 1
                event_owner.events[index] = event
        _require(summary['physical_event_count'] == len(shot_analysis.events),
                 'fresh physical events contain additional unmatched observations')
        event_owner.flush()
        summary['calibration_count'] = len(output.calibrations)
        _admit(rebuilt, shot_analysis)
        output.seal().write_json(destination)
    _current(rebuilt)
    summary['path'] = str(destination)
    return summary
