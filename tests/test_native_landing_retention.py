"""Actual native landing geometry/storage statements with synthetic contacts.

No video, independent annotation, inference or accuracy qualification is used.
The retained pre-repair capacity run is expected to expose resident growth.
"""
import ast
from collections.abc import Sequence
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import tracemalloc
from types import SimpleNamespace

import numpy as np
import pytest

from src.visualization.landing_zones import landing_record
from src.utils.landing_zone_sequence import LandingZoneSequence

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = []


class Events(Sequence):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)
        return SimpleNamespace(event_id=index + 1, frame_index=index,
            timestamp_s=index / 25., event_type=SimpleNamespace(value='BOUNCE'),
            court_position_m=(2., 8.), evidence={
                'review_status': 'confirmed', 'active_play': True,
                'position_type': 'ground_contact',
                'ground_contact_source': 'SUPPLIED_SYNTHETIC_CONTACT_NOT_GROUND_TRUTH',
                'position_uncertainty_m': .1})


class Calibrations:
    def __init__(self):
        self.matrix = np.eye(3)

    def __getitem__(self, index):
        return SimpleNamespace(is_valid=True, image_to_court=self.matrix)


class Writeback:
    count = 0
    last = None

    def __setitem__(self, index, event):
        self.count += 1
        self.last = index, event.event_id, dict(event.evidence)


def source_blocks():
    path = ROOT / 'src/pipeline/phase6_pipeline.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    pipeline = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Phase6Pipeline')
    run = next(node for node in pipeline.body if isinstance(node, ast.FunctionDef) and node.name == '_run_with_frame_exports')
    start = next(index for index, node in enumerate(run.body) if isinstance(node, ast.Assign)
        and any(getattr(target, 'id', None) == 'landing_settings' for target in node.targets))
    end = next(index + 1 for index in range(start, len(run.body))
        if isinstance(run.body[index], ast.If)
        and ast.unparse(run.body[index].test).startswith("landing_settings.get('enabled',"))
    active = next(node for node in ast.walk(run) if isinstance(node, ast.Assign)
        and any(getattr(target, 'id', None) == 'active_landings' for target in node.targets))
    return path, run.body[start:end], active


def records_digest(records):
    digest = hashlib.sha256()
    count = 0
    for record in records:
        digest.update(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        digest.update(b'\n')
        count += 1
    return count, digest.hexdigest()


def probe(directory, count):
    directory.mkdir()
    path, flow, active = source_blocks()
    writeback = Writeback()
    namespace = dict(self=SimpleNamespace(config={
        'match_configuration': {'match_id': 'synthetic-landing-retention'},
        'analysis_revision': 3, 'visualization': {'landing_zones': {
            'enabled': True, 'partition': 'tactical_3x3', 'duration_seconds': .8}}}),
        authoritative_events=Events(count), frame_calibrations=Calibrations(),
        event_output=SimpleNamespace(events=writeback), disk_players=True,
        dead_event_ids=(), landing_record=landing_record, os=os, json=json,
        LandingZoneSequence=LandingZoneSequence,
        output_dir=str(directory), fps=25.)
    gc.collect()
    tracemalloc.start()
    clock = time.perf_counter()
    try:
        with ExitStack() as stack:
            namespace['export_stack'] = stack
            exec(compile(ast.Module(body=flow, type_ignores=[]), str(path), 'exec'), namespace)
            retained, peak = tracemalloc.get_traced_memory()
            total, digest = records_digest(namespace['landing_records'])
            assert total == count * 2
            assert len(namespace['landing_calibrations']) == count
            assert writeback.count == count * 2
            assert writeback.last[2]['analysis_revision'] == 3
            selections = []
            for frame in (0, 1, 20, max(0, count - 1), count + 25):
                namespace['i'] = frame
                exec(compile(ast.Module(body=[active], type_ignores=[]), str(path), 'exec'), namespace)
                length, selected = records_digest(namespace['active_landings'])
                selections.append({'frame': frame, 'count': length, 'sha256': selected})
            raw = json.loads((directory / 'landing_zones.json').read_text())
            assert records_digest(raw['records']) == (total, digest)
            assert raw['calibrations'] == dict(namespace['landing_calibrations'])
            row = dict(events=count, landing_records=total, calibrations=count,
                retained_traced_bytes=retained, peak_traced_bytes=peak,
                values_sha256=digest, active_selections=selections,
                actual_native_geometry=True, synthetic_writeback_assignments=writeback.count,
                seconds=time.perf_counter() - clock)
            OBSERVATIONS.append(row)
            return row
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize('count', [257, 1025, 4097])
def test_actual_native_landing_geometry_and_export_values(tmp_path, count):
    probe(tmp_path / f'contacts-{count}', count)


def test_native_landing_retention_remains_bounded():
    rows = sorted(OBSERVATIONS, key=lambda row: row['events'])
    assert [row['events'] for row in rows] == [257, 1025, 4097]
    assert rows[-1]['retained_traced_bytes'] < rows[0]['retained_traced_bytes'] + 1024**2
