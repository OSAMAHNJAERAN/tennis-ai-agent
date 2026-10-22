"""Actual Phase6 mini-court drawing and source-query work; synthetic data only."""
import ast
from contextlib import ExitStack
import hashlib
import math
import gc
import tracemalloc
import weakref
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.events.event_detector import EventType, TennisEvent
from src.utils import event_analysis_sequence as event_module
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.visualization.landing_zones import draw_landing, landing_record
from src.visualization.mini_court import MiniCourt
from src.visualization.bounce_history import MiniCourtBounceHistory

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = []
CAPACITY_OBSERVATIONS = []


def source_blocks(source=None):
    source = source or ROOT / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    render = next(node for node in run.body if isinstance(node, ast.With)
                  and any(isinstance(item.context_expr, ast.Call) and getattr(item.context_expr.func, 'id', None) == 'ExitStack' for item in node.items))
    frame_loop = next(node for node in render.body if isinstance(node, ast.For) and getattr(node.target, 'id', None) == 'i')
    section = next(node for node in frame_loop.body if isinstance(node, ast.If)
                   and any(isinstance(child, ast.Assign) and any(getattr(target, 'id', None) == 'mc_canvas' for target in child.targets)
                           for child in node.body))
    initializers = [node for node in render.body if isinstance(node, ast.Assign)
                    and any(getattr(target, 'id', None) == 'mini_history' for target in node.targets)]
    return source, initializers, section


class Camera:
    def __getitem__(self, frame):
        return SimpleNamespace(is_valid=frame % 17 != 0)


def add_source_events(owner, count):
    # Deliberately reverse insertion order, retain same-frame ties, absent
    # positions and non-bounce events. The actual indexed query defines order.
    for index in range(count - 1, -1, -1):
        owner.events.append(TennisEvent(count - index,
            EventType.PLAYER_1_HIT if index % 7 == 0 else EventType.BOUNCE,
            index // 2, index / 50., None, None,
            None if index % 11 == 0 else ((index % 13) / 2. - .5, (index % 29) - 2.),
            .99, 'SYNTHETIC_ONLY', {'source': 'SYNTHETIC_PIXEL_WORK_FIXTURE'}))


def active_contact(frame):
    timestamp = (frame // 23) * 23 / 25.
    return landing_record({'event_type': 'BOUNCE', 'event_id': frame // 23 + 1,
        'timestamp_s': timestamp, 'court_position_m': [5.485, 8.],
        'evidence': {'review_status': 'confirmed', 'active_play': True,
            'position_type': 'ground_contact', 'ground_contact_source': 'SYNTHETIC_PIXEL_WORK_FIXTURE',
            'position_uncertainty_m': .1}}, np.eye(3), analysis_id='synthetic', revision=0,
        calibration_id='synthetic-camera', partition='service')


@pytest.mark.parametrize('count', [129, 257, 513])
def test_actual_native_mini_court_pixels_and_read_work(tmp_path, monkeypatch, count):
    source, initializers, section = source_blocks()
    decoded = 0
    decode = event_module._decode_record
    def count_reads(kind, payload):
        nonlocal decoded
        if kind == 'events': decoded += 1
        return decode(kind, payload)
    with EventAnalysisSequence(tmp_path) as owner:
        add_source_events(owner, count)
        monkeypatch.setattr(event_module, '_decode_record', count_reads)
        namespace = dict(mini_court=MiniCourt(), event_output=owner, EventType=EventType,
            MiniCourtBounceHistory=MiniCourtBounceHistory,
            disk_players=True, authoritative_enabled=True, authoritative_events=owner.events,
            frame_calibrations=Camera(), cv2=cv2, draw_landing=draw_landing,
            fps=25., ann_frame=np.zeros((720, 1280, 3), np.uint8),
            VideoAnnotator=SimpleNamespace(compose_frame=lambda image, miniature: image))
        exec(compile(ast.Module(body=initializers, type_ignores=[]), str(source), 'exec'), namespace)
        pixels = hashlib.sha256()
        valid = 0
        for frame in range(count):
            namespace.update(i=frame, p1_c_pos=(2. + math.sin(frame), 20.),
                p2_c_pos=(7., 3. + math.cos(frame)), ball_court_pos=(5.485, 8.),
                active_landings=(active_contact(frame),))
            exec(compile(ast.Module(body=[section], type_ignores=[]), str(source), 'exec'), namespace)
            if namespace['frame_calibrations'][frame].is_valid:
                valid += 1
                assert namespace['mc_img'].shape == (500, 250, 3)
                pixels.update(namespace['mc_img'].tobytes())
        row = dict(events=count, rendered_source_frames=count, valid_mini_court_frames=valid,
                   source_event_decodes=decoded, miniature_pixels_sha256=pixels.hexdigest(),
                   actual_native_source_block=True, actual_court_and_zone_drawing=True,
                   final_video_composition_mocked=True, qualification_evidence=False)
        reference_path = ROOT / 'artifacts/validation/yolo26_execution_2026-10-17/mini_court_history_checks_v1/redraw_red_v1/receipt.json'
        assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == '259414907a6e80975e70a9b0ced850971b9125465e1a3b7c878be8abc2602291'
        expected = next(item for item in json.loads(reference_path.read_text())['rows'] if item['events'] == count)
        assert row['miniature_pixels_sha256'] == expected['miniature_pixels_sha256']
        row['retained_red_pixels_equal'] = True
        OBSERVATIONS.append(row)
    assert not list(tmp_path.glob('event_analysis_*.sqlite3*'))


def test_native_cumulative_history_reads_each_source_event_at_most_once():
    rows = sorted(OBSERVATIONS, key=lambda row: row['events'])
    assert [row['events'] for row in rows] == [129, 257, 513]
    assert all(row['source_event_decodes'] <= row['events'] for row in rows)


def test_generation_changes_only_for_successful_physical_event_writes(tmp_path):
    with EventAnalysisSequence(tmp_path) as owner:
        assert owner.events_generation == 0
        owner.verification_traces.append({'synthetic': True})
        assert owner.events_generation == 0
        add_source_events(owner, 1)
        assert owner.events_generation == 1
        value = owner.events[0]
        value.court_position_m = (4., 8.)
        owner.events[0] = value
        assert owner.events_generation == 2
        value.timestamp_s = float('nan')
        with pytest.raises(ValueError): owner.events[0] = value
        assert owner.events_generation == 2
    with pytest.raises(RuntimeError): owner.events_generation


@pytest.mark.parametrize('disk,authority', [(False, True), (False, False), (True, False)])
def test_legacy_and_withheld_paths_keep_original_pixels_without_diagnostic_queries(tmp_path, monkeypatch, disk, authority):
    baseline = ROOT / 'artifacts/validation/yolo26_execution_2026-10-17/mini_court_history_checks_v1/redraw_red_v1/source_snapshot/src/pipeline/phase6_pipeline.py'
    sources = (source_blocks(baseline), source_blocks())
    with EventAnalysisSequence(tmp_path) as owner:
        add_source_events(owner, 9)
        events = list(owner.events) if authority else []
        def unexpected(*args, **kwargs):
            raise AssertionError('Legacy or withheld authority must not query diagnostic event storage')
        monkeypatch.setattr(owner, 'events_by_frame', unexpected)
        results = []
        for path, initializers, section in sources:
            namespace = dict(mini_court=MiniCourt(), event_output=owner, EventType=EventType,
                MiniCourtBounceHistory=MiniCourtBounceHistory, disk_players=disk,
                authoritative_enabled=authority, authoritative_events=events,
                frame_calibrations=Camera(), cv2=cv2, draw_landing=draw_landing,
                fps=25., ann_frame=np.zeros((720, 1280, 3), np.uint8),
                VideoAnnotator=SimpleNamespace(compose_frame=lambda image, miniature: image))
            exec(compile(ast.Module(body=initializers, type_ignores=[]), str(path), 'exec'), namespace)
            images = []
            for frame in (1, 3, 11, 23):
                namespace.update(i=frame, p1_c_pos=(2., 20.), p2_c_pos=(7., 3.),
                    ball_court_pos=(5.485, 8.), active_landings=(active_contact(frame),))
                exec(compile(ast.Module(body=[section], type_ignores=[]), str(path), 'exec'), namespace)
                images.append(namespace['mc_img'].tobytes())
            results.append(images)
        assert results[0] == results[1]


def test_history_is_causal_repeatable_and_refuses_old_frames_and_source_changes(tmp_path):
    court = MiniCourt()
    with EventAnalysisSequence(tmp_path) as owner:
        owner.events.append(TennisEvent(1, EventType.BOUNCE, 3, .12, None, None,
                                       (2., 8.), .9, 'SYNTHETIC_ONLY', {}))
        history = MiniCourtBounceHistory(court, owner, EventType.BOUNCE)
        before = court.draw_court()
        assert history.draw(before.copy(), 2).tobytes() == before.tobytes()
        after = history.draw(before.copy(), 3)
        expected = before.copy()
        point = court.court_to_mini_court(2., 8.)
        cv2.circle(expected, point, 5, (0, 165, 255), -1)
        cv2.circle(expected, point, 7, (255, 255, 255), 1)
        np.testing.assert_array_equal(after, expected)
        np.testing.assert_array_equal(history.draw(before.copy(), 3), expected)
        with pytest.raises(ValueError, match='forward'): history.draw(before.copy(), 2)
        value = owner.events[0]
        value.court_position_m = (3., 8.)
        owner.events[0] = value
        unchanged = before.copy()
        with pytest.raises(ValueError, match='Changed source'): history.draw(unchanged, 4)
        np.testing.assert_array_equal(unchanged, before)
        replacement = MiniCourtBounceHistory(court, owner, EventType.BOUNCE)
        assert not np.array_equal(replacement.draw(before.copy(), 4), expected)
    with pytest.raises(RuntimeError, match='not open'): replacement.draw(before.copy(), 5)


@pytest.mark.parametrize('field,value', [('margin', 21), ('scale_x', 1.), ('width', 251)])
def test_changed_projection_requires_new_history(tmp_path, field, value):
    with EventAnalysisSequence(tmp_path) as owner:
        court = MiniCourt()
        history = MiniCourtBounceHistory(court, owner, EventType.BOUNCE)
        image = court.draw_court()
        setattr(court, field, value)
        with pytest.raises(ValueError, match='original raster geometry'): history.draw(image, 0)


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_source_failure_closes_iterator_and_cannot_resume_a_partial_raster(tmp_path, monkeypatch, failure):
    with EventAnalysisSequence(tmp_path) as owner:
        add_source_events(owner, 3)
        history = MiniCourtBounceHistory(MiniCourt(), owner, EventType.BOUNCE)
        closed = []
        original = owner.events_by_frame
        def interrupted(*args, **kwargs):
            reader = original(*args, **kwargs)
            try:
                yield next(reader)
                raise failure('Supplied source interruption')
            finally:
                reader.close()
                closed.append(True)
        monkeypatch.setattr(owner, 'events_by_frame', interrupted)
        image = history.court.draw_court()
        untouched = image.copy()
        with pytest.raises(failure, match='Supplied source'): history.draw(image, 3)
        assert closed == [True]
        np.testing.assert_array_equal(image, untouched)
        with pytest.raises(RuntimeError, match='rebuilt'): history.draw(image, 3)


@pytest.mark.parametrize('count', [257, 1025, 4097])
def test_raster_and_live_event_memory_do_not_accumulate_records(tmp_path, monkeypatch, count):
    with EventAnalysisSequence(tmp_path) as owner:
        add_source_events(owner, count)
        live = peak_live = 0
        decode = event_module._decode_record
        def release():
            nonlocal live
            live -= 1
        def observe(kind, payload):
            nonlocal live, peak_live
            result = decode(kind, payload)
            if kind == 'events':
                live += 1
                peak_live = max(peak_live, live)
                weakref.finalize(result, release)
            return result
        monkeypatch.setattr(event_module, '_decode_record', observe)
        court = MiniCourt()
        image = court.draw_court()
        gc.collect()
        tracemalloc.start()
        try:
            history = MiniCourtBounceHistory(court, owner, EventType.BOUNCE)
            history.draw(image, count)
            retained, peak = tracemalloc.get_traced_memory()
            assert history._colors.nbytes + history._mask.nbytes == 500000
            assert live == 0 and peak_live <= 3
            assert retained < 1024**2 and peak < 2 * 1024**2
            CAPACITY_OBSERVATIONS.append(dict(events=count, raster_bytes=500000,
                retained_traced_bytes=retained, peak_traced_bytes=peak,
                peak_live_events=peak_live, live_events_after_draw=live,
                qualification_evidence=False))
        finally:
            tracemalloc.stop()
