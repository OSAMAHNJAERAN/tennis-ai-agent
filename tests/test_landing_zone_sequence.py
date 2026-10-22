"""Finite evidence storage, causal overlay selection and atomic publication."""
import json
import math

import pytest

from src.utils import landing_zone_sequence as module
from src.utils.landing_zone_sequence import LandingZoneSequence
from src.utils import analytics_json_artifact as artifacts


def pair(ident, *, timestamp=None, duration=.8, partition='tactical_3x3', calibration_id=None):
    record = dict(schema_version='1.0', analysis_id='synthetic', analysis_revision=3,
        event_id=ident, timestamp_s=ident / 25. if timestamp is None else timestamp,
        duration_s=duration, partition=partition, calibration_id=calibration_id or f'frame_{ident}',
        contact_position_m=[2., 8.], line_call=None, zones=[{'polygon_m': [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]}])
    calibration = dict(analysis_id='synthetic', analysis_revision=3, frame_index=ident,
                       image_to_court=[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    return record, calibration


def test_owned_views_keep_order_and_detached_finite_values(tmp_path):
    values = [pair(ident, calibration_id=key) for ident, key in ((1, 'b'), (2, 'a'), (3, 'b'))]
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        for record, calibration in values:
            owner.append(record, calibration)
        assert list(owner.records) == [value[0] for value in values]
        assert list(owner.records[::-1]) == [value[0] for value in reversed(values)]
        assert owner.records[-1] == values[-1][0]
        assert list(owner.calibrations) == ['b', 'a']
        assert list(owner.calibrations.items()) == [('b', values[2][1]), ('a', values[1][1])]
        value = owner.records[0]
        value['zones'][0]['polygon_m'][0][0] = 300.
        assert owner.records[0] == values[0][0]
        registry = owner.calibrations['b']
        registry['frame_index'] = 100
        assert owner.calibrations['b'] == values[2][1]
        path = tmp_path / 'landing_zones.json'
        with pytest.raises(ValueError, match='Seal'):
            owner.write_json(path)
        assert not path.exists()
        owner.seal().write_json(path)
        assert json.loads(path.read_text()) == dict(schema_version='1.0', analysis_id='synthetic',
            analysis_revision=3, calibrations={'b': values[2][1], 'a': values[1][1]}, records=[value[0] for value in values])
        with pytest.raises(ValueError, match='Sealed'):
            owner.append(*pair(4))
    assert not list(tmp_path.glob('landing_zones_*.sqlite3*'))


@pytest.mark.parametrize('offset', [0., 10**12, 10**18])
def test_active_window_repeats_exact_python_endpoints_without_future_paint(tmp_path, offset):
    values = [pair(ident + 1, timestamp=offset + delta, duration=duration, partition=partition)
        for ident, (delta, duration, partition) in enumerate((
            (0., .6, 'tactical_3x3'), (.2, .8, 'tactical_3x3'),
            (.6, 1., 'service'), (1., 1., 'tactical_3x3')))]
    times = (offset - .1, offset, math.nextafter(offset + .6, -math.inf), offset + .6,
             math.nextafter(offset + .6, math.inf), offset + 1., offset + 2.)
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        for value in values:
            owner.append(*value)
        owner.seal()
        for timestamp in times:
            expected = [record for record, _ in values if record['partition'] == 'tactical_3x3'
                and record['timestamp_s'] <= timestamp < record['timestamp_s'] + record['duration_s']]
            selected = owner.active_records('tactical_3x3', timestamp)
            assert list(selected) == expected
            assert list(selected) == expected  # Original and mini-court can iterate independently.
        plans = owner.connection.execute('EXPLAIN QUERY PLAN SELECT payload FROM records '
            'WHERE partition=? AND start>=? AND start<=? ORDER BY position', ('tactical_3x3', 0., 1.)).fetchall()
        assert any('SEARCH records USING INDEX landing_time' in row[-1] for row in plans)


@pytest.mark.parametrize('field,value', [('timestamp_s', True), ('timestamp_s', float('nan')),
    ('duration_s', True), ('duration_s', 1.01), ('analysis_revision', True),
    ('analysis_id', 'foreign'), ('partition', 'arbitrary'), ('calibration_id', '')])
def test_invalid_evidence_does_not_mutate_prior_records(tmp_path, field, value):
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1))
        record, calibration = pair(2)
        record[field] = value
        with pytest.raises((ValueError, OverflowError)):
            owner.append(record, calibration)
        assert len(owner.records) == len(owner.calibrations) == 1
        assert owner.records[0] == pair(1)[0]
        owner.append(*pair(3))
        assert len(owner.records) == 2


def test_failed_pair_savepoint_preserves_accepted_batch_and_registry(tmp_path, monkeypatch):
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1, calibration_id='same'))
        execute = owner._execute
        def fail_record(sql, *args, **kwargs):
            if sql.startswith('INSERT INTO records'):
                raise OSError('Supplied record write failure')
            return execute(sql, *args, **kwargs)
        monkeypatch.setattr(owner, '_execute', fail_record)
        with pytest.raises(OSError, match='Supplied record'):
            owner.append(*pair(2, calibration_id='same'))
        monkeypatch.setattr(owner, '_execute', execute)
        assert len(owner.records) == 1
        assert owner.calibrations['same'] == pair(1)[1]
        owner.append(*pair(3))
        assert [row['event_id'] for row in owner.records] == [1, 3]


@pytest.mark.parametrize('committed', [False, True])
def test_uncertain_commit_rejects_all_later_reads_and_publication(tmp_path, monkeypatch, committed):
    class Connection:
        def __init__(self, actual): self.actual = actual
        def __getattr__(self, name): return getattr(self.actual, name)
        def commit(self):
            if committed: self.actual.commit()
            raise OSError('Uncertain supplied commit')
    destination = tmp_path / 'landing_zones.json'
    destination.write_bytes(b'previous-completed-artifact')
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1))
        owner.connection = Connection(owner.connection)
        with pytest.raises(OSError, match='Uncertain'):
            owner.seal()
        for action in (lambda: len(owner.records), lambda: list(owner.calibrations),
                       lambda: list(owner.active_records('service', 0.)), lambda: owner.append(*pair(2)),
                       lambda: owner.write_json(destination)):
            with pytest.raises(RuntimeError, match='uncertain'):
                action()
        assert destination.read_bytes() == b'previous-completed-artifact'
    assert not list(tmp_path.glob('landing_zones_*.sqlite3*'))


@pytest.mark.parametrize('failure', ['encode_record', 'encode_calibration', 'fsync', 'replace'])
def test_export_failure_keeps_previous_file_and_closes_both_readers(tmp_path, monkeypatch, failure):
    destination = tmp_path / 'landing_zones.json'
    destination.write_bytes(b'previous-completed-artifact')
    def fail(*args, **kwargs): raise OSError('Supplied export failure')
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1)); owner.seal()
        if failure.startswith('encode_'):
            encode = module._json
            field = 'event_id' if failure == 'encode_record' else 'image_to_court'
            def fail_one(value):
                if isinstance(value, dict) and field in value:
                    fail()
                return encode(value)
            monkeypatch.setattr(module, '_json', fail_one)
        else:
            monkeypatch.setattr(artifacts.os, failure, fail)
        with pytest.raises(OSError, match='Supplied export'):
            owner.write_json(destination)
        assert owner.readers == set()
        assert destination.read_bytes() == b'previous-completed-artifact'
        assert not list(tmp_path.glob('.landing_zones.json.*.tmp'))


def test_closed_owner_disposes_suspended_readers_and_rejects_reuse(tmp_path):
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1)); owner.seal()
        readers = [iter(owner.records), iter(owner.calibrations.items()), iter(owner.active_records('tactical_3x3', .1))]
        for reader in readers: next(reader)
        assert len(owner.readers) == 3
    assert owner.readers == set()
    for reader in readers:
        with pytest.raises(RuntimeError, match='not open'): next(reader)
    with pytest.raises(RuntimeError, match='reused'): owner.__enter__()
    assert not list(tmp_path.glob('landing_zones_*.sqlite3*'))


def test_calibration_mapping_rejects_numeric_key_alias(tmp_path):
    with LandingZoneSequence(tmp_path, 'synthetic', 3) as owner:
        owner.append(*pair(1, calibration_id='1'))
        assert owner.calibrations['1']['frame_index'] == 1
        with pytest.raises(KeyError): owner.calibrations[1]
        with pytest.raises(TypeError): owner.calibrations[[]]
