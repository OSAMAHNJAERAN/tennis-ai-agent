"""Bind display zones to reviewed contacts, current revision and actual frame calibration."""
import math
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
import json
import sqlite3

import numpy as np

from src.visualization.landing_zones import landing_record
from src.api.result_storage import DiskSeries


def _event_key(ident):
    return json.dumps([type(ident).__name__, ident], separators=(',', ':'))


class _EventLookup:
    def __init__(self, index, events):
        self.index, self.events = index, events

    def get(self, ident):
        row = self.index.execute('SELECT position FROM events WHERE id=?', (_event_key(ident),)).fetchone()
        return self.events[row[0]] if row is not None else None


class _SeenPairs:
    def __init__(self, index):
        self.index = index

    def __contains__(self, pair):
        return self.index.execute('SELECT 1 FROM seen WHERE event=? AND partition=?',
                                  (_event_key(pair[0]), pair[1])).fetchone() is not None

    def add(self, pair):
        self.index.execute('INSERT INTO seen VALUES(?,?)', (_event_key(pair[0]), pair[1]))


@contextmanager
def _event_index(events):
    # Worker events are an owned DiskSeries. Preserve the historical iterable
    # interface for explicit compatibility callers; it may materialize there.
    events = events if isinstance(events, Sequence) else list(events)
    with closing(sqlite3.connect('')) as index:
        for pragma in ('cache_size=-2048', 'temp_store=FILE', 'mmap_size=0'):
            index.execute('PRAGMA ' + pragma)
        index.execute('CREATE TABLE events(id TEXT PRIMARY KEY,position INTEGER NOT NULL)')
        index.execute('CREATE TABLE seen(event TEXT,partition TEXT,PRIMARY KEY(event,partition))')
        for position, event in enumerate(events):
            if not isinstance(event, dict) or event.get('event_id') is None:
                raise ValueError('Invalid authoritative contact event')
            ident = event['event_id']
            if type(ident) not in (int, str) or isinstance(ident, str) and not ident:
                raise ValueError('Invalid authoritative event ID')
            try:
                index.execute('INSERT INTO events VALUES(?,?)', (_event_key(ident), position))
            except sqlite3.IntegrityError:
                raise ValueError('Authoritative event IDs must be unique') from None
        yield _EventLookup(index, events), _SeenPairs(index)


def _finite_number(value):
    try:
        return type(value) in (float, int) and math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _point(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(_finite_number(item) for item in value))


def _geometry_equal(first, second):
    def finite_geometry(value):
        return (isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0
                and all(finite_geometry(item) if isinstance(item, (list, tuple, np.ndarray)) else _finite_number(item)
                        for item in value))
    try:
        if not finite_geometry(first) or not finite_geometry(second):
            return False
        a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, atol=1e-7, rtol=0)
    except (ValueError, TypeError, OverflowError):
        return False


def validate_landing_zones(raw, analysis_id, events, *, frame_calibrations=None, expected_revision=None):
    """Recompute semantics; the document registry never supplies calibration authority.

    Use the actual normalized camera record at the event frame and the revision
    of the authoritative event document. Contact pixels may be a reviewed point's
    inverse projection, so this checks consistency rather than physical accuracy.
    """
    if (not isinstance(raw, dict) or raw.get('schema_version') != '1.0' or raw.get('analysis_id') != analysis_id
            or type(raw.get('analysis_revision')) is not int or raw['analysis_revision'] < 0
            or not isinstance(raw.get('records'), (list, DiskSeries))):
        raise ValueError('Landing-zone document is not bound to this analysis/revision')
    if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 0
                                         or raw['analysis_revision'] != expected_revision):
        raise ValueError('Landing-zone revision differs from the authoritative event document')
    if not raw['records']:
        return []
    if frame_calibrations is None or expected_revision is None or not isinstance(raw.get('calibrations'), Mapping):
        raise ValueError('Landing-zone records require independent frame calibration and current revision bindings')
    with _event_index(events) as (event_by_id, seen):
        return _validate_records(raw, analysis_id, event_by_id, seen,
            frame_calibrations=frame_calibrations, expected_revision=expected_revision)


def _validate_records(raw, analysis_id, event_by_id, seen, *, frame_calibrations, expected_revision):
    for row in raw['records']:
        if not isinstance(row, dict) or type(row.get('event_id')) not in (int, str) or not isinstance(row.get('partition'), str):
            raise ValueError('Landing-zone records require a source event and partition')
        event = event_by_id.get(row['event_id'])
        evidence = event.get('evidence', {}) if event else {}
        if not isinstance(evidence, dict):
            raise ValueError('Invalid contact event evidence')
        pair = (row['event_id'], row['partition'])
        frame_index = event.get('frame_index', event.get('frame')) if event else None
        calibration_id = evidence.get('calibration_id')
        if (not event or event.get('event_type') != 'BOUNCE' or pair in seen
                or evidence.get('active_play') is not True or evidence.get('position_type') != 'ground_contact'
                or evidence.get('review_status') not in {'confirmed', 'corrected'}
                or evidence.get('analysis_id') != analysis_id or evidence.get('analysis_revision') != expected_revision
                or type(evidence.get('analysis_revision')) is not int
                or type(frame_index) is not int or frame_index < 0
                or calibration_id != f'frame_{frame_index}' or row.get('calibration_id') != calibration_id
                or row.get('analysis_id') != analysis_id or row.get('analysis_revision') != expected_revision
                or type(row.get('analysis_revision')) is not int or row.get('schema_version') != '1.0'
                or not _finite_number(row.get('timestamp_s')) or row['timestamp_s'] < 0
                or not _finite_number(event.get('timestamp_s')) or abs(row['timestamp_s']-event['timestamp_s']) > 1e-6
                or not _finite_number(row.get('duration_s')) or not .6 <= row['duration_s'] <= 1.
                or not isinstance(row.get('zones'), list)):
            raise ValueError('Landing-zone evidence conflicts with its event or revision')
        registry = raw['calibrations'].get(calibration_id)
        try:
            frame = frame_calibrations[frame_index]
        except (IndexError, KeyError, TypeError):
            raise ValueError('Landing-zone calibration frame is unavailable') from None
        if (not isinstance(registry, dict) or not isinstance(frame, dict)
                or registry.get('analysis_id') != analysis_id or registry.get('analysis_revision') != expected_revision
                or type(registry.get('analysis_revision')) is not int or registry.get('frame_index') != frame_index
                or type(registry.get('frame_index')) is not int or frame.get('frame_index') != frame_index
                or type(frame.get('frame_index')) is not int
                or frame.get('isValid') is not True
                or not _geometry_equal(registry.get('image_to_court'), frame.get('imageToCourt'))):
            raise ValueError('Landing-zone matrix is not bound to the actual valid source frame calibration')
        matrix = np.asarray(frame.get('imageToCourt'), dtype=float)
        if matrix.shape != (3,3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
            raise ValueError('Landing-zone source calibration is not finite/invertible')
        position, pixel = event.get('court_position_m'), evidence.get('contact_position_px')
        radius = evidence.get('position_uncertainty_m')
        if (not _point(position) or not _point(pixel) or not _finite_number(radius) or radius < 0
                or not _point(row.get('contact_position_m')) or not _point(row.get('contact_position_px'))
                or not _geometry_equal(row['contact_position_m'], position)
                or not _geometry_equal(row['contact_position_px'], pixel)
                or row.get('position_uncertainty_m') != radius or not _finite_number(row.get('position_uncertainty_m'))):
            raise ValueError('Landing-zone requires finite ground-contact coordinates and uncertainty')
        mapped = matrix @ np.array([*pixel, 1.])
        if abs(mapped[2]) < 1e-10 or not np.isfinite(mapped).all() or np.linalg.norm(mapped[:2]/mapped[2]-position) > radius+1e-6:
            raise ValueError('Ground contact pixels conflict with the source calibration and position uncertainty')
        source = evidence.get('ground_contact_source')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Ground contact requires evidence provenance')
        expected = landing_record(event, matrix, analysis_id=analysis_id, revision=expected_revision,
                                  calibration_id=calibration_id, partition=row['partition'], duration_seconds=row['duration_s'])
        if expected is None:
            raise ValueError('Landing-zone record is not supported by confirmed active ground contact')
        for key in ('status', 'label', 'source', 'review_status', 'line_call'):
            if row.get(key) != expected.get(key):
                raise ValueError('Landing-zone status/label/provenance must be recomputed from the contact evidence')
        if len(row['zones']) != len(expected['zones']):
            raise ValueError('Landing-zone candidate cells differ from the contact uncertainty envelope')
        for actual_zone, expected_zone in zip(row['zones'], expected['zones']):
            if (not isinstance(actual_zone, dict) or actual_zone.get('id') != expected_zone['id']
                    or actual_zone.get('label') != expected_zone['label']
                    or not _geometry_equal(actual_zone.get('polygon_m'), expected_zone['polygon_m'])
                    or not _geometry_equal(actual_zone.get('polygon_px'), expected_zone['polygon_px'])):
                raise ValueError('Landing-zone cell or perspective polygon differs from recomputed ground geometry')
        seen.add(pair)
    return raw['records']
