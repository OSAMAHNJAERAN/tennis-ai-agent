"""Perspective-aligned zone presentation from confirmed ground-contact evidence.

Tactical partitions are descriptive. They never establish a legal serve or line
call. Ground-contact/uncertainty evidence must come from a qualified producer or
an explicit human correction, not an airborne pixel projected by homography.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from src.court.court_geometry import TennisCourtGeometry as Court


@dataclass(frozen=True)
class Zone:
    id: str
    label: str
    bounds: tuple[float, float, float, float]

    @property
    def polygon(self):
        left, top, right, bottom = self.bounds
        return [[left, top], [right, top], [right, bottom], [left, bottom]]


def partition_zones(kind):
    left = (Court.COURT_WIDTH_DOUBLES - Court.COURT_WIDTH_SINGLES) / 2
    right, center, net = Court.COURT_WIDTH_DOUBLES - left, Court.COURT_WIDTH_DOUBLES / 2, Court.NET_TO_BASELINE
    zones = []
    if kind == 'service':
        for side, top, bottom in [('far', net - Court.SERVICE_LINE_DISTANCE, net), ('near', net, net + Court.SERVICE_LINE_DISTANCE)]:
            for corridor, x1, x2 in [('left', left, center), ('right', center, right)]:
                zones.append(Zone(f'{side}_service_{corridor}', f'{side.title()} {corridor} service box', (x1, top, x2, bottom)))
    elif kind in {'tactical_3x3', 'tactical_4x3'}:
        columns, rows = 3, 3 if kind == 'tactical_3x3' else 4
        for side, offset in [('far', 0.), ('near', net)]:
            for row in range(rows):
                for column in range(columns):
                    x1, x2 = left + Court.COURT_WIDTH_SINGLES * column / columns, left + Court.COURT_WIDTH_SINGLES * (column + 1) / columns
                    y1, y2 = offset + net * row / rows, offset + net * (row + 1) / rows
                    zones.append(Zone(f'{side}_r{row}_c{column}', f'{side.title()} tactical row {row + 1}, column {column + 1}', (x1, y1, x2, y2)))
    else:
        raise ValueError('Choose service, tactical_3x3 or tactical_4x3 partition')
    return zones


def _touches(zone, point, radius):
    x1, y1, x2, y2 = zone.bounds
    dx = max(x1 - point[0], 0, point[0] - x2)
    dy = max(y1 - point[1], 0, point[1] - y2)
    return math.hypot(dx, dy) <= radius + 1e-10


def project_polygon(polygon, image_to_court):
    matrix = np.asarray(image_to_court, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('A finite invertible calibrated image-to-court homography is required')
    inverse = np.linalg.inv(matrix)
    homogeneous = np.c_[np.asarray(polygon, dtype=float), np.ones(len(polygon))] @ inverse.T
    if not np.isfinite(homogeneous).all() or (np.abs(homogeneous[:, 2]) < 1e-10).any():
        raise ValueError('Zone projection intersects the projective horizon')
    if not ((homogeneous[:, 2] > 0).all() or (homogeneous[:, 2] < 0).all()):
        raise ValueError('Zone crosses the projective horizon')
    return (homogeneous[:, :2] / homogeneous[:, 2, None]).tolist()


def landing_record(event, image_to_court, *, analysis_id, revision, calibration_id, partition='tactical_3x3', duration_seconds=.8):
    """Build display-only geometry, never infer the existence of a bounce.

    Return None for unconfirmed, dead-ball, airborne or missing-uncertainty
    evidence. The caller retains those unknown events in its review history.
    """
    evidence = event.get('evidence', {})
    if (event.get('event_type') != 'BOUNCE' or evidence.get('review_status') not in {'confirmed', 'corrected'}
            or evidence.get('active_play') is not True or evidence.get('position_type') != 'ground_contact'
            or not evidence.get('ground_contact_source')):
        return None
    position = event.get('court_position_m')
    radius = evidence.get('position_uncertainty_m')
    if (not isinstance(position, (tuple, list)) or len(position) != 2
            or not all(type(v) in (float, int) and math.isfinite(v) for v in position)
            or not isinstance(radius, (float, int)) or isinstance(radius, bool) or not math.isfinite(radius) or radius < 0):
        return None
    timestamp = event.get('timestamp_s')
    if (type(timestamp) not in (float, int) or not math.isfinite(timestamp) or timestamp < 0
            or not .6 <= duration_seconds <= 1.0 or type(revision) is not int or revision < 0
            or not analysis_id or not calibration_id):
        raise ValueError('Require bound analysis/revision/calibration and a valid source timestamp/highlight duration')
    touched = [zone for zone in partition_zones(partition) if _touches(zone, position, radius)]
    contained = []
    for zone in touched:
        left, top, right, bottom = zone.bounds
        if left < position[0] - radius and position[0] + radius < right and top < position[1] - radius and position[1] + radius < bottom:
            contained.append(zone)
    status = 'inside' if len(contained) == 1 and len(touched) == 1 else 'uncertain' if touched else 'outside_partition'
    label = touched[0].label if status == 'inside' else 'Landing zone uncertain' if touched else 'Outside the selected zones'
    return {'schema_version': '1.0', 'analysis_id': analysis_id, 'analysis_revision': revision,
            'calibration_id': calibration_id, 'event_id': event['event_id'], 'timestamp_s': timestamp,
            'duration_s': duration_seconds, 'partition': partition, 'status': status, 'label': label,
            'contact_position_m': list(position), 'contact_position_px': project_polygon([position], image_to_court)[0],
            'position_uncertainty_m': radius,
            'source': evidence['ground_contact_source'], 'review_status': evidence['review_status'],
            'line_call': evidence.get('line_call') if evidence.get('line_call') in {'in', 'out', 'uncertain'} else None,
            'zones': [{'id': zone.id, 'label': zone.label, 'polygon_m': zone.polygon,
                       'polygon_px': project_polygon(zone.polygon, image_to_court)} for zone in touched]}


def draw_landing(frame, record, *, timestamp_seconds, transform=None, alpha=.22):
    """Fill complete polygons in the original image or transformed mini-court."""
    if not record or not record['timestamp_s'] <= timestamp_seconds < record['timestamp_s'] + record['duration_s']:
        return frame
    output, fill = frame.copy(), frame.copy()
    color = (70, 190, 120) if record['status'] == 'inside' else (70, 175, 235)
    for zone in record['zones']:
        points = zone['polygon_px'] if transform is None else [transform(*point) for point in zone['polygon_m']]
        cv2.fillPoly(fill, [np.rint(points).astype(np.int32)], color)
    cv2.addWeighted(fill, alpha, output, 1 - alpha, 0, dst=output)
    contact = record['contact_position_px'] if transform is None else transform(*record['contact_position_m'])
    center = tuple(np.rint(contact).astype(int))
    cv2.circle(output, center, 4, color, -1, cv2.LINE_AA)
    cv2.circle(output, center, 6, (240, 240, 240), 1, cv2.LINE_AA)
    # Keep the explanatory label inside either the image or small court canvas.
    cv2.putText(output, record['label'], (12, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                .38 if transform else .55, color, 1, cv2.LINE_AA)
    return output
