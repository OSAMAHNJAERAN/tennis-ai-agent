"""Actual corrected landing geometry with explicitly synthetic reviewed contacts.

These fixtures execute native rules, shot linking, landing projection/storage
and independent artifact consistency validation, never physical qualification.
"""
import builtins
from contextlib import contextmanager
import json
from pathlib import Path

import pytest

from tests.test_corrected_shot_rebuild import build_shot_base
from tests.test_corrected_rule_rebuild import unchanged


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
            raise AssertionError('Landing export imported inference: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def reviewed_base(tmp_path, monkeypatch, *, evidence_change=None, config_tail='', dead_bounce=False):
    from src.events.event_detector import EventType
    from src.scoring import point_timeline
    original = point_timeline.process_timeline_event
    def reviewed(event, *args, **kwargs):
        if isinstance(getattr(event, 'evidence', None), dict):
            if dead_bounce and event.event_type is EventType.PLAYER_2_HIT:
                event.event_type, event.player_id = EventType.BOUNCE, None
            if event.event_type is EventType.BOUNCE:
                event.evidence.update(review_status='confirmed', active_play=True,
                    position_type='ground_contact',
                    ground_contact_source='EXPLICIT_SYNTHETIC_CONTACT_NOT_INDEPENDENT_GROUND_TRUTH',
                    position_uncertainty_m=.1, analysis_id='stale', analysis_revision=999,
                    calibration_id='stale-initial-camera', contact_position_px=[999., 999.],
                    line_call='in')
                event.evidence.update(evidence_change or {})
        return original(event, *args, **kwargs)
    with monkeypatch.context() as scoped:
        scoped.setattr(point_timeline, 'process_timeline_event', reviewed)
        return build_shot_base(tmp_path, config_tail=(
            'visualization:\n  landing_zones:\n    enabled: true\n'
            '    partition: tactical_3x3\n    duration_seconds: 0.8\n' + config_tail))


@contextmanager
def corrected(fixture):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.scoring.corrected_shot_rebuild import relink_corrected_shots
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch,
                                        request=fixture.request) as rebuilt:
        with relink_corrected_shots(rebuilt, fixture.scratch) as analysis:
            yield rebuilt, analysis


def export_directory(fixture, name='landing-output'):
    path = fixture.scratch / name
    path.mkdir()
    return path


def event_document(event):
    return {'event_id': event.event_id, 'event_type': event.event_type.value,
        'frame_index': event.frame_index, 'timestamp_s': event.timestamp_s,
        'court_position_m': event.court_position_m, 'evidence': event.evidence}


def test_fresh_corrected_fault_contact_replaces_scope_and_line_call_using_current_geometry(tmp_path, monkeypatch):
    from src.api.landing_zone_contract import validate_landing_zones
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = reviewed_base(tmp_path, monkeypatch)
    destination = export_directory(fixture)
    with corrected(fixture) as (rebuilt, analysis):
        summary = write_corrected_landing_zones(rebuilt, analysis, destination)
        assert summary['record_count'] == 2 and summary['calibration_count'] == 1
        assert summary['published'] is False and summary['old_landings_reused'] is False
        raw = json.loads((destination / 'landing_zones.json').read_text())
        assert raw['analysis_revision'] == 1
        assert {row['partition'] for row in raw['records']} == {'service', 'tactical_3x3'}
        for row in raw['records']:
            assert row['contact_position_m'] == pytest.approx([3., 8.])
            assert row['contact_position_px'] == pytest.approx([300., 800.])
            assert row['position_uncertainty_m'] == .1
            assert row['line_call'] == 'out'  # Original legal deuce serve was IN.
            assert row['analysis_id'] == 'fixture' and row['calibration_id'] == 'frame_4'
        fresh_event = analysis.events[1]
        assert fresh_event.evidence['analysis_revision'] == 1
        frames = [{**{'frame_index': index}, 'isValid': True,
                   'imageToCourt': [[.01, 0., 0.], [0., .01, 0.], [0., 0., 1.]]}
                  for index in range(fixture.snapshot.frames)]
        assert len(validate_landing_zones(raw, 'fixture',
            [event_document(event) for event in analysis.events],
            frame_calibrations=frames, expected_revision=1)) == 2
        assert sorted(path.name for path in destination.iterdir()) == ['landing_zones.json']
    unchanged(fixture)


@pytest.mark.parametrize('change', [
    {'review_status': 'provisional'}, {'active_play': False},
    {'position_type': 'airborne_projection'}, {'ground_contact_source': ''},
    {'ground_contact_source': '   '}, {'position_uncertainty_m': None},
    {'position_uncertainty_m': True}, {'position_uncertainty_m': -.1},
])
def test_line_call_confidence_never_supplies_missing_contact_qualification(tmp_path, monkeypatch, change):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = reviewed_base(tmp_path, monkeypatch, evidence_change=change)
    with corrected(fixture) as (rebuilt, analysis):
        summary = write_corrected_landing_zones(rebuilt, analysis, export_directory(fixture))
        assert summary['record_count'] == 0 and summary['unqualified_contact_count'] == 1
        raw = json.loads(Path(summary['path']).read_text())
        assert raw['analysis_revision'] == 1 and raw['records'] == [] and raw['calibrations'] == {}
        assert 'analysis_revision' not in analysis.events[1].evidence
        assert 'contact_position_px' not in analysis.events[1].evidence
    unchanged(fixture)


def test_unqualified_source_cannot_be_upgraded_by_mutating_fresh_event_snapshot(tmp_path):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = build_shot_base(tmp_path, config_tail='visualization:\n  landing_zones:\n    enabled: true\n')
    with corrected(fixture) as (rebuilt, analysis):
        event = analysis.events[1]
        event.evidence.update(review_status='corrected', active_play=True, position_type='ground_contact',
            ground_contact_source='MUTABLE_AND_UNTRUSTED', position_uncertainty_m=.01)
        analysis.events[1] = event
        summary = write_corrected_landing_zones(rebuilt, analysis, export_directory(fixture))
        assert summary['record_count'] == 0 and summary['unqualified_contact_count'] == 1
        assert 'review_status' not in analysis.events[1].evidence
    unchanged(fixture)


def test_corrected_dead_bounce_has_no_landing_despite_original_contact_review(tmp_path, monkeypatch):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = reviewed_base(tmp_path, monkeypatch, dead_bounce=True)
    with corrected(fixture) as (rebuilt, analysis):
        assert rebuilt.owner.event(3)['outcome_type'] == 'DEAD_BALL_IGNORED'
        summary = write_corrected_landing_zones(rebuilt, analysis, export_directory(fixture))
        assert summary['dead_contact_count'] == 1 and summary['record_count'] == 2
        raw = json.loads(Path(summary['path']).read_text())
        assert {row['event_id'] for row in raw['records']} == {2}
        assert 'calibration_id' not in analysis.events[2].evidence
        assert analysis.events[2].evidence['active_play'] is False
    unchanged(fixture)


def test_fresh_source_join_damage_rejects_before_export_and_preserves_generation(tmp_path, monkeypatch):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = reviewed_base(tmp_path, monkeypatch)
    destination = export_directory(fixture)
    with corrected(fixture) as (rebuilt, analysis):
        event = analysis.events[1]
        event.court_position_m = (999., 999.)
        analysis.events[1] = event
        with pytest.raises(ValueError, match='accepted corrected call'):
            write_corrected_landing_zones(rebuilt, analysis, destination)
        assert list(destination.iterdir()) == []
    unchanged(fixture)


def test_existing_destination_and_analysis_directory_are_rejected(tmp_path, monkeypatch):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = reviewed_base(tmp_path, monkeypatch)
    destination = export_directory(fixture)
    (destination / 'landing_zones.json').write_text('PREEXISTING PRIVATE FILE')
    with corrected(fixture) as (rebuilt, analysis):
        with pytest.raises(ValueError, match='unused private'):
            write_corrected_landing_zones(rebuilt, analysis, destination)
        with pytest.raises(ValueError, match='separate from the immutable'):
            write_corrected_landing_zones(rebuilt, analysis, fixture.directory)
        assert (destination / 'landing_zones.json').read_text() == 'PREEXISTING PRIVATE FILE'
    unchanged(fixture)


def test_geometry_withheld_does_not_borrow_initial_or_old_landing_geometry(tmp_path, monkeypatch):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    from src.scoring.correction_sources import EventGeometry
    fixture = reviewed_base(tmp_path, monkeypatch)
    with corrected(fixture) as (rebuilt, analysis):
        monkeypatch.setattr(rebuilt.sources, 'event_geometry', lambda *_: EventGeometry(None, False, 'TEST_WITHHELD'))
        summary = write_corrected_landing_zones(rebuilt, analysis, export_directory(fixture))
        assert summary['record_count'] == 0 and summary['unavailable_geometry_count'] == 1
        assert json.loads(Path(summary['path']).read_text())['calibrations'] == {}
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['confidence', 'trajectory_state', 'nonbounce_court_position'])
def test_fresh_display_cannot_contradict_immutable_optical_source(tmp_path, damage):
    from src.scoring.corrected_landing_exports import write_corrected_landing_zones
    fixture = build_shot_base(tmp_path)
    destination = export_directory(fixture)
    with corrected(fixture) as (rebuilt, analysis):
        event = analysis.events[0]
        if damage == 'confidence':
            event.confidence = .001
        elif damage == 'trajectory_state':
            event.trajectory_state = 'MUTABLE_UNSUPPORTED_STATE'
        else:
            event.court_position_m = (999., 999.)
        analysis.events[0] = event
        with pytest.raises(ValueError, match='source join|nonbounce court'):
            write_corrected_landing_zones(rebuilt, analysis, destination)
        assert not (destination/'landing_zones.json').exists()
    unchanged(fixture)
