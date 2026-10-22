"""Actual native event/box adapters on sealed, explicitly synthetic sources."""
import builtins
import copy
import gc
import hashlib
import json
import tracemalloc

import pytest

from src.scoring.corrected_shot_sources import corrected_shot_sources


CAPACITY_OBSERVATIONS = []


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.', 1)[0] in ('torch', 'ultralytics'):
            raise AssertionError('Shot source software must not import models: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)


def build_native_sealed_base(tmp_path, monkeypatch, *, frame_count=12, reviewed_let=False,
                             mutate_detections=None, missing_players=False):
    """Reuse the real rule fixture with dictionary physical evidence at creation.

    The original rule-only fixture permits opaque strings. Native TennisEvent
    uses a dictionary, so this fixture constructs that actual input type before
    source capture/publication; no sealed artifact or production code is patched.
    All coordinates, identities, and geometry remain explicitly synthetic.
    """
    from tests import test_corrected_rule_rebuild as rules
    from tests import test_correction_sources as observations
    original_namespace, original_writer = rules.SimpleNamespace, observations.write_observation_assets
    def namespace(**kwargs):
        if 'evidence' in kwargs:
            kwargs['evidence'] = {'scope': 'EXPLICIT_SYNTHETIC_MEASUREMENTS', 'note': kwargs['evidence']}
            kwargs['court_position_m'] = (99.1234567890123, 77.9876543210987)
        return original_namespace(**kwargs)
    def write(*args, **kwargs):
        files = original_writer(*args, **kwargs)
        path = files['detections.json']
        document = json.loads(path.read_text(encoding='utf-8'))
        if not missing_players:
            for frame in document['frames']:
                frame['player_1'] = {'bbox': [100.1234567890123, 200.9876543210987,
                    300.1234567890123, 600.9876543210987],
                    'source_track_id': 7, 'confidence': .8765432101234567}
                frame['player_2'] = {'bbox': [700., 100., 800., 300.],
                    'source_track_id': None, 'confidence': .75}
        if mutate_detections is not None:
            mutate_detections(document)
        path.write_text(json.dumps(document, allow_nan=False), encoding='utf-8')
        return files
    with monkeypatch.context() as patch:
        patch.setattr(rules, 'SimpleNamespace', namespace)
        patch.setattr(observations, 'write_observation_assets', write)
        return rules.build_sealed_base(tmp_path, frame_count=frame_count, reviewed_let=reviewed_let)


def unchanged(fixture):
    from tests.test_corrected_rule_rebuild import unchanged as check
    check(fixture)


def test_native_events_use_new_bounce_coordinates_and_never_old_contact_projection(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    from src.events.event_detector import TennisEvent, EventType
    from src.utils.bbox_utils import BBox
    fixture = build_native_sealed_base(tmp_path, monkeypatch)
    # Aliases are not authority after publication.
    (fixture.directory / 'detections.json').write_text('{}', encoding='utf-8')
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
            assert sources.events is sources.event_owner.events
            assert sources.image_size == (1920, 1080) and type(sources.image_size) is tuple
            assert len(sources.events) == 2
            contact, bounce = sources.events[0], sources.events[1]
            assert isinstance(contact, TennisEvent) and contact.event_type is EventType.SERVE_CONTACT
            assert isinstance(bounce, TennisEvent) and bounce.event_type is EventType.BOUNCE
            assert contact.evidence == {'scope': 'EXPLICIT_SYNTHETIC_MEASUREMENTS', 'note': 'SYNTHETIC_CONTACT'}
            assert contact.court_position_m is None
            assert bounce.court_position_m == pytest.approx((3., 8.))
            assert bounce.court_position_m != tuple(fixture.document['inputs'][1]['source_input']['court_position_m'])
            box = sources.player1_boxes[0]
            assert isinstance(box, BBox)
            assert (box.x1, box.y1, box.x2, box.y2) == (100.1234567890123, 200.9876543210987,
                                                     300.1234567890123, 600.9876543210987)
            assert box.confidence == .8765432101234567 and box.track_id == 7
            assert not hasattr(box, 'ground_point_px') and not hasattr(box, 'keypoints')
            box.x1 = -999.
            assert sources.player1_boxes[0].x1 == 100.1234567890123
            assert sources.player2_boxes[-1].track_id is None
            selected = sources.player1_boxes[::-2]
            assert len(selected) == 6 and selected[0].x1 == sources.player1_boxes[-1].x1
            assert sources.audit['physical_event_count'] == 2
            assert sources.audit['nonbounce_positions_withheld'] == 1
            assert sources.audit['bounce_positions_available'] == 1
            assert sources.audit['old_shot_links_reused'] is False
            assert rebuilt.sources.ball_points[0].court_x_m is None
            assert rebuilt.sources.ball_points[0].speed_kmh is None
            event_path = sources.event_owner.path
        assert not event_path.exists()
        with pytest.raises(RuntimeError, match='closed'):
            selected[0]
        with pytest.raises(RuntimeError):
            sources.events[0]
        assert rebuilt.owner.sealed is False and not len(rebuilt.owner.member_rows())
    unchanged(fixture)


def test_reviewed_rule_signals_are_not_native_physical_events(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch, reviewed_let=True)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
            assert [event.event_id for event in sources.events] == [1, 4, 2]
            assert sources.audit['accepted_input_count'] == 4
            assert sources.audit['reviewed_signal_count'] == 1
    unchanged(fixture)


def test_missing_player_observations_stay_missing(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch, missing_players=True)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
            assert sources.player1_boxes[0] is sources.player2_boxes[-1] is None
            with pytest.raises(TypeError):
                sources.player1_boxes[True]
            with pytest.raises(IndexError):
                sources.player1_boxes[len(sources.player1_boxes)]


@pytest.mark.parametrize('damage', ['bbox_bool', 'bbox_reversed', 'confidence_bool', 'confidence_high',
    'track_bool', 'track_fraction', 'track_negative', 'extra_player_field', 'missing_player', 'match_identity'])
def test_malformed_sealed_player_observations_fail_closed(tmp_path, monkeypatch, damage):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    def mutate(document):
        frame = document['frames'][0]
        player = frame['player_1']
        if damage == 'bbox_bool': player['bbox'][0] = False
        elif damage == 'bbox_reversed': player['bbox'][2] = player['bbox'][0]
        elif damage == 'confidence_bool': player['confidence'] = True
        elif damage == 'confidence_high': player['confidence'] = 1.1
        elif damage == 'track_bool': player['source_track_id'] = True
        elif damage == 'track_fraction': player['source_track_id'] = 7.0
        elif damage == 'track_negative': player['source_track_id'] = -1
        elif damage == 'extra_player_field': player['pose'] = []
        elif damage == 'missing_player': del frame['player_1']
        elif damage == 'match_identity': frame['match_identity'] = [{'state': 'CONFIRMED', 'player_id': 'B'}]
    fixture = build_native_sealed_base(tmp_path, monkeypatch, mutate_detections=mutate)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        existing = set(fixture.scratch.iterdir())
        with pytest.raises(ValueError):
            with corrected_shot_sources(rebuilt, fixture.scratch):
                pytest.fail('Malformed selected player observation yielded')
        assert set(fixture.scratch.iterdir()) == existing
    unchanged(fixture)
    assert not list(fixture.scratch.glob('event_analysis_*.sqlite3*'))


@pytest.mark.parametrize('damage', ['scoring_call', 'source_fingerprint', 'transition',
    'source_role_bool', 'evidence_string', 'authority_off'])
def test_mutable_pending_rows_cannot_inject_old_geometry_or_unsupported_native_types(tmp_path, monkeypatch, damage):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        if damage == 'authority_off':
            rebuilt.authoritative_events_enabled = False
        else:
            index = 1 if damage == 'scoring_call' else 0
            row = rebuilt.owner.rows('replay_inputs')[index]
            if damage == 'scoring_call': row['scoring_input']['line_call']['bounce_position_m'] = [99., 88.]
            elif damage == 'source_fingerprint': row['input_fingerprint'] = 'b' * 64
            elif damage == 'transition': row['accepted_transition']['outcome_type'] = 'RALLY_CONTINUES'
            elif damage == 'source_role_bool': row['source_input']['player_id'] = True
            elif damage == 'evidence_string': row['source_input']['evidence'] = 'NOT_NATIVE_DICTIONARY'
            if damage in ('source_role_bool', 'evidence_string'):
                row['input_fingerprint'] = hashlib.sha256(json.dumps(row['source_input'],
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                transition = rebuilt.owner.event(row['source_input']['event_id'])
                transition['input_fingerprint'] = row['input_fingerprint']
                rebuilt.owner.put('transitions', transition['uid'], transition,
                    event_id=transition['event_id'], frame=transition['frame'])
                row['accepted_transition'] = transition
            rebuilt.owner.put('replay_inputs', row['accepted_transition']['uid'], row,
                event_id=row['source_input']['event_id'], frame=row['source_input']['frame_index'])
        with pytest.raises(ValueError):
            with corrected_shot_sources(rebuilt, fixture.scratch):
                pytest.fail('Mutated pending source yielded')
    unchanged(fixture)
    assert not list(fixture.scratch.glob('event_analysis_*.sqlite3*'))


def test_unavailable_new_bounce_geometry_does_not_reuse_old_source_projection(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch)
    request = copy.deepcopy(fixture.request)
    source = fixture.document['inputs'][1]
    replacement = copy.deepcopy(source['source_input'])
    request['input_amendments'] = [{'event_id': 2,
        'expected_input_fingerprint': source['input_fingerprint'], 'replacement_source_input': replacement}]
    # The supported amendment contract preserves original source anchor tokens.
    # A current context with a missing geometry review is instead supplied by
    # changing only that review binding field to unavailable.
    replacement['context']['binding']['geometry_qualification_ref'] = ''
    replacement['context']['binding_audit_matches'] = False
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=request) as rebuilt:
        with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
            assert sources.events[1].court_position_m is None
            assert sources.audit['bounce_positions_unavailable'] == 1
            assert fixture.document['inputs'][1]['source_input']['court_position_m'] is not None
    unchanged(fixture)


@pytest.mark.parametrize('where', ['analysis', 'parent', 'missing', 'traversal'])
def test_direct_scratch_admission_cannot_write_into_immutable_analysis(tmp_path, monkeypatch, where):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch)
    directory = {'analysis': fixture.directory, 'parent': tmp_path, 'missing': tmp_path / 'missing',
                 'traversal': fixture.scratch / '..' / 'private-rebuild'}[where]
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        with pytest.raises(ValueError):
            with corrected_shot_sources(rebuilt, directory):
                pytest.fail('Unsafe scratch yielded')
    unchanged(fixture)


def test_body_failure_closes_event_and_box_owners_without_publication(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    fixture = build_native_sealed_base(tmp_path, monkeypatch)
    with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
        existing = set(fixture.scratch.iterdir())
        with pytest.raises(RuntimeError, match='EXPLICIT_INJECTED_BODY_FAILURE'):
            with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
                path = sources.event_owner.path
                boxes = sources.player1_boxes[:]
                raise RuntimeError('EXPLICIT_INJECTED_BODY_FAILURE')
        assert not path.exists()
        assert set(fixture.scratch.iterdir()) == existing
        with pytest.raises(RuntimeError): boxes[0]
        with pytest.raises(RuntimeError): sources.events[0]
        assert not len(rebuilt.owner.member_rows())
    unchanged(fixture)


def test_adapter_duration_capacity_retains_no_python_frame_or_event_cache(tmp_path, monkeypatch):
    from src.scoring.corrected_rule_rebuild import rebuild_corrected_rule_timeline
    # Warm native numerical imports before measured adapter-only scope.
    import numpy
    import src.events.event_detector
    measured = []
    for frame_count in (257, 4097):
        root = tmp_path / str(frame_count)
        root.mkdir()
        fixture = build_native_sealed_base(root, monkeypatch, frame_count=frame_count)
        with rebuild_corrected_rule_timeline(fixture.directory, fixture.scratch, request=fixture.request) as rebuilt:
            gc.collect()
            tracemalloc.start()
            with corrected_shot_sources(rebuilt, fixture.scratch) as sources:
                assert isinstance(sources.player1_boxes.indices, range)
                assert isinstance(sources._frames, __import__('src.api.result_storage', fromlist=['DiskSeries']).DiskSeries)
                assert sources.player1_boxes[-1].track_id == 7 and len(sources.events) == 2
                gc.collect()
                retained, peak = tracemalloc.get_traced_memory()
                measured.append({'frames': frame_count, 'events': len(sources.events),
                    'retained_bytes': retained, 'peak_bytes': peak,
                    'scope': 'ADAPTER_ONLY; FIXTURE_AND_RULE_REBUILD_AND_NATIVE_IMPORTS_EXCLUDED'})
            tracemalloc.stop()
        unchanged(fixture)
    CAPACITY_OBSERVATIONS.extend(measured)
    assert measured[1]['retained_bytes'] - measured[0]['retained_bytes'] < 256 * 1024
    assert measured[1]['peak_bytes'] - measured[0]['peak_bytes'] < 2 * 1024 * 1024
