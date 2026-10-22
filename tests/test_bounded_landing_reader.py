"""Actual native producer, typed event writeback and bounded saved validation.

All contacts/calibration are explicit synthetic fixtures; no images or models.
"""
import ast
from contextlib import ExitStack
import gc
import json
import os
from pathlib import Path
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.api.landing_zone_contract import validate_landing_zones
from src.api.result_storage import DiskSeries, IndexedJsonObject, ResultStorage
from src.events.event_detector import EventType, TennisEvent
from src.utils.event_analysis_sequence import EventAnalysisSequence
from src.utils.json_array_artifact import write_json_array_artifact
from src.utils.landing_zone_sequence import LandingZoneSequence
from src.visualization.landing_zones import landing_record
from tests.test_native_landing_retention import Calibrations, Events, records_digest, source_blocks

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = []


class CameraFrames:
    def __init__(self, count): self.count = count
    def __getitem__(self, frame):
        if type(frame) is not int or not 0 <= frame < self.count:
            raise IndexError(frame)
        return {'frame_index': frame, 'isValid': True,
                'imageToCourt': [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]}


@pytest.mark.parametrize('count', [257, 1025, 4097])
def test_actual_new_landing_exports_validate_without_resident_registry_or_event_indexes(tmp_path, count):
    source, flow, _ = source_blocks()
    tree = ast.parse(source.read_text(encoding='utf-8'))
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ('sanitize_json_value', 'event_export_records')]
    assert len(helpers) == 2
    with ExitStack() as stack:
        event_owner = stack.enter_context(EventAnalysisSequence(tmp_path))
        for value in Events(count):
            event_owner.events.append(TennisEvent(value.event_id, EventType.BOUNCE,
                value.frame_index, value.timestamp_s, None, (2., 8.), value.court_position_m,
                .99, 'SYNTHETIC_ONLY', value.evidence))
        namespace = dict(self=SimpleNamespace(config={
            'match_configuration': {'match_id': 'synthetic-landing-retention'},
            'analysis_revision': 3, 'visualization': {'landing_zones': {
                'enabled': True, 'partition': 'tactical_3x3', 'duration_seconds': .8}}}),
            authoritative_events=event_owner.events, frame_calibrations=Calibrations(),
            event_output=event_owner, disk_players=True, dead_event_ids=(),
            landing_record=landing_record, LandingZoneSequence=LandingZoneSequence,
            os=os, json=json, output_dir=str(tmp_path), export_stack=stack, np=np, Any=Any)
        exec(compile(ast.Module(body=helpers + flow, type_ignores=[]), str(source), 'exec'), namespace)
        events_path = tmp_path / 'match_events.json'
        write_json_array_artifact(events_path, {'schema_version': '1.0',
            'analysis_id': 'synthetic-landing-retention', 'analysis_revision': 3}, 'events',
            namespace['event_export_records'](event_owner.events), allow_nan=False)
        # The physical source above is exported from actual typed event
        # writeback, never reconstructed from zone or scoring output.
        with ResultStorage(tmp_path) as storage:
            gc.collect()
            tracemalloc.start()
            clock = time.perf_counter()
            try:
                events = storage.read_json(events_path)['events']
                raw = storage.read_json(tmp_path / 'landing_zones.json', indexed_root_objects=('calibrations',))
                assert isinstance(events, DiskSeries)
                assert isinstance(raw['calibrations'], IndexedJsonObject)
                assert len(raw['calibrations']) == count
                validated = validate_landing_zones(raw, 'synthetic-landing-retention', events,
                    frame_calibrations=CameraFrames(count), expected_revision=3)
                assert isinstance(validated, DiskSeries) and len(validated) == 2 * count
                retained, peak = tracemalloc.get_traced_memory()
                total, digest = records_digest(validated)
                reference = json.loads((ROOT / 'artifacts/validation/yolo26_execution_2026-10-17/native_landing_retention_red_v1/attempt1/receipt.json').read_text())
                expected = next(row for row in reference['rows'] if row['events'] == count)
                assert digest == expected['values_sha256']
                OBSERVATIONS.append(dict(events=count, validated_records=total,
                    retained_traced_bytes=retained, peak_traced_bytes=peak,
                    values_sha256=digest, original_red_values_equal=True,
                    seconds=time.perf_counter() - clock,
                    actual_typed_physical_export_and_writeback=True,
                    registry_type=type(raw['calibrations']).__name__, qualification_evidence=False))
            finally:
                tracemalloc.stop()
    assert not list(tmp_path.glob('landing_zones_*.sqlite3*'))
    assert not list(tmp_path.glob('event_analysis_*.sqlite3*'))


def test_bounded_reader_retention_does_not_scale_with_all_calibrations_and_event_ids():
    rows = sorted(OBSERVATIONS, key=lambda row: row['events'])
    assert [row['events'] for row in rows] == [257, 1025, 4097]
    assert rows[-1]['retained_traced_bytes'] < rows[0]['retained_traced_bytes'] + 1024**2
    assert rows[-1]['peak_traced_bytes'] < rows[0]['peak_traced_bytes'] + 2 * 1024**2
