"""Synchronized independent-reference speed evaluator; synthetic fixtures are labelled."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.analytics.flight_reconstruction import (FlightBoundary, FlightConfig, MetricCamera,
                                                PixelObservation, reconstruct_flight_segment)


def evaluate_flight_manifest(manifest, *, config=None):
    if manifest.get('schema_version') != '1.0' or manifest.get('usage_rights') != 'authorized_for_evaluation':
        raise ValueError('Versioned source manifest with authorized evaluation rights is required.')
    if manifest.get('all_strokes_enumerated') is not True:
        raise ValueError('All strokes in the declared scope must be enumerated, including unknown/unmeasurable cases.')
    records = manifest.get('strokes', [])
    if not records:
        raise ValueError('No independent synchronized stroke references supplied.')
    config = config or FlightConfig()
    identities, results, groups = set(), [], {}
    for record in records:
        ident = record.get('stroke_id')
        if not ident or ident in identities or not record.get('match_id') or not record.get('stroke_type'):
            raise ValueError('Stroke IDs must be unique with match and stroke-type provenance.')
        identities.add(ident)
        estimate = None
        if record.get('camera') and record.get('observations'):
            camera = MetricCamera(**record['camera'])
            points = [PixelObservation(**point) for point in record['observations']]
            start = FlightBoundary(**record['start_boundary']) if record.get('start_boundary') else None
            end = FlightBoundary(**record['end_boundary']) if record.get('end_boundary') else None
            estimate = reconstruct_flight_segment(camera, points, segment_id=ident,
                                                  start_boundary=start, end_boundary=end, config=config)
        row = {'stroke_id': ident, 'match_id': record['match_id'], 'stroke_type': record['stroke_type'],
               'estimate': estimate.to_dict() if estimate else None, 'reference_status': 'unavailable',
               'measurement_definition': None, 'absolute_error_kmh': None, 'signed_error_kmh': None,
               'reference_unavailable_reason': record.get('reference_unavailable_reason', 'No synchronized reference supplied.')}
        reference = record.get('reference')
        if reference:
            definition = reference.get('definition')
            if definition not in ('segment_average_physical_speed', 'racket_impact_speed'):
                raise ValueError('Reference must distinguish path-average physical speed from racket-impact speed.')
            if reference.get('status') != 'independently_measured' or not reference.get('source') or not reference.get('synchronization_evidence_id'):
                raise ValueError('Reference speed needs independent source and explicit synchronized timing evidence.')
            speed = reference.get('speed_kmh')
            if speed is None or not np.isfinite(speed) or speed < 0:
                raise ValueError('Reference speed must be finite and nonnegative.')
            row.update(reference_status='independently_measured', measurement_definition=definition,
                       reference_unavailable_reason=None, reference_speed_kmh=speed)
            if estimate and estimate.status == 'estimated_unqualified':
                if definition == 'segment_average_physical_speed':
                    expected = (estimate.start_seconds, estimate.end_seconds)
                    supplied = (reference.get('start_seconds'), reference.get('end_seconds'))
                    if None in supplied or not np.allclose(supplied, expected, atol=1e-6, rtol=0):
                        raise ValueError('Path-average reference and reconstruction must use exactly the same source-time window.')
                    measured = estimate.segment_average_physical_speed_kmh
                else:
                    if estimate.racket_impact_speed_kmh is None:
                        measured = None
                    else:
                        if reference.get('impact_speed_side') != estimate.impact_speed_side:
                            raise ValueError('Impact reference must match incoming/outgoing contact-speed definition.')
                        timestamp = reference.get('timestamp_seconds')
                        if timestamp is None or not np.isclose(timestamp, estimate.impact_timestamp_seconds, atol=1e-6, rtol=0):
                            raise ValueError('Impact reference must be synchronized to the same evidenced racket contact.')
                        measured = estimate.racket_impact_speed_kmh
                if measured is not None:
                    row.update(absolute_error_kmh=abs(measured-speed), signed_error_kmh=measured-speed)
        results.append(row)
        groups.setdefault(record['stroke_type'], []).append(row)
    summaries = {}
    for stroke_type, rows in groups.items():
        measured_rows = [row for row in rows if row['estimate'] and row['estimate']['status'] == 'estimated_unqualified']
        references = [row for row in rows if row['reference_status'] == 'independently_measured']
        definitions = {}
        for definition in ('segment_average_physical_speed', 'racket_impact_speed'):
            eligible = [row for row in references if row['measurement_definition'] == definition]
            evaluated = [row for row in eligible if row['absolute_error_kmh'] is not None]
            errors = np.array([row['absolute_error_kmh'] for row in evaluated])
            median = float(np.median(errors)) if len(errors) else None
            p95 = float(np.percentile(errors, 95)) if len(errors) else None
            definitions[definition] = {'reference_eligible_count': len(eligible), 'evaluated_count': len(evaluated),
                'unresolved_count': len(eligible)-len(evaluated),
                'coverage_of_reference_strokes': len(evaluated)/len(eligible) if eligible else None,
                'match_count': len({row['match_id'] for row in evaluated}),
                'median_absolute_error_kmh': median, 'p95_absolute_error_kmh': p95,
                'mae_kmh': float(np.mean(errors)) if len(errors) else None,
                'bias_kmh': float(np.mean([row['signed_error_kmh'] for row in evaluated])) if evaluated else None,
                'proposed_error_gate_passed': median <= 5 and p95 <= 10 if len(errors) else None}
        summaries[stroke_type] = {'all_stroke_count': len(rows), 'estimated_count': len(measured_rows),
            'unknown_count': len(rows)-len(measured_rows), 'physical_estimate_coverage': len(measured_rows)/len(rows),
            'independent_reference_count': len(references), 'reference_unknown_count': len(rows)-len(references),
            'definitions': definitions}
    return {'schema_version': '1.0', 'data_kind': manifest.get('data_kind', 'declared_independent_reference_requires_audit'),
            'qualification_status': 'unqualified_pending_independent_reference_audit_and_sample_review',
            'all_stroke_count': len(results), 'proposed_gates': {'median_absolute_error_kmh': 5, 'p95_absolute_error_kmh': 10},
            'error_gate_scope': 'Matched measurement definition/window; every unresolved/reference-missing stroke retained.',
            'config': vars(config), 'by_stroke_type': summaries, 'strokes': results}


def evaluate_flight_file(source: Path, output: Path):
    raw = source.read_bytes()
    report = evaluate_flight_manifest(json.loads(raw))
    report['manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    report['implementation_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(__file__).with_name('flight_reconstruction.py'))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
