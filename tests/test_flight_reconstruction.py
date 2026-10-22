"""Known-camera synthetic checks; no real-world speed qualification is established."""
from dataclasses import replace

import numpy as np
import pytest

from src.analytics.flight_reconstruction import (FlightBoundary, FlightConfig, MetricCamera,
                                                PixelObservation, reconstruct_flight_segment,
                                                split_flight_observations)


def known_camera():
    centre = np.array([5.485, -12., 9.])
    forward = np.array([5.485, 11., 0.])-centre
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.array([right, down, forward])
    intrinsic = np.array([[1600., 0., 960.], [0., 1600., 540.], [0., 0., 1.]])
    return MetricCamera(intrinsic, rotation, -rotation@centre, 'camera-1',
                        'synthetic_calibration_no_independent_measurement', 'synthetic_known_camera')


def trajectory(camera=None, position=(3., 3., 1.5), velocity=(2., 18., 3.), duration=.7, count=43):
    camera = camera or known_camera()
    _, _, _, projection = camera.matrices()
    times = np.linspace(0, duration, count)
    xyz = np.asarray(position) + times[:, None]*np.asarray(velocity) + .5*times[:, None]**2*[0., 0., -9.80665]
    homogeneous = np.column_stack((xyz, np.ones(count)))@projection.T
    pixels = homogeneous[:, :2]/homogeneous[:, 2, None]
    points = [PixelObservation(f'observation-{i}', float(time+120), float(x), float(y))
              for i, (time, (x, y)) in enumerate(zip(times, pixels))]
    return points, xyz


def test_known_camera_recovers_ballistic_3d_and_segment_speed_with_no_physical_claim():
    points, expected = trajectory()
    result = reconstruct_flight_segment(known_camera(), points)
    assert result.status == 'estimated_unqualified', result.reason
    assert np.allclose(result.positions_m, expected, atol=1e-8)
    assert np.allclose(result.initial_velocity_mps, [2, 18, 3], atol=1e-8)
    # Independent dense trapezoid integral for travelled arc length / time.
    times = np.linspace(0, .7, 10001)
    speeds = np.linalg.norm(np.array([2,18,3]) + times[:, None]*[0,0,-9.80665], axis=1)
    expected_speed = np.trapezoid(speeds, times)/.7*3.6
    assert result.segment_average_physical_speed_kmh == pytest.approx(expected_speed, rel=1e-9)
    assert result.racket_impact_speed_kmh is None and result.uncertainty_interval is None
    assert result.qualification_status.startswith('unqualified')


def test_contact_speed_uses_evidenced_boundary_and_is_distinct_from_segment_average():
    points, _ = trajectory()
    boundary = FlightBoundary('racket_contact', 120, 1.5, 'reviewed-contact', position_xy_m=(3.,3.))
    result = reconstruct_flight_segment(known_camera(), points, start_boundary=boundary)
    assert result.status == 'estimated_unqualified', result.reason
    assert result.racket_impact_speed_kmh == pytest.approx(np.linalg.norm([2,18,3])*3.6, rel=1e-10)
    assert result.impact_evidence_id == 'reviewed-contact'
    assert result.racket_impact_speed_kmh != result.segment_average_physical_speed_kmh


def test_ground_contact_constraint_and_high_speed_are_rejected_when_incompatible():
    points, _ = trajectory()
    wrong = FlightBoundary('bounce', points[-1].timestamp_seconds, .034, 'ground-contact',
                           position_xy_m=(30., 40.), position_tolerance_m=.1)
    result = reconstruct_flight_segment(known_camera(), points, end_boundary=wrong)
    assert result.status == 'unavailable' and result.positions_m is None
    fast, _ = trajectory(velocity=(200,18,3))
    result = reconstruct_flight_segment(known_camera(), fast)
    assert result.status == 'unavailable' and 'speed' in result.reason


@pytest.mark.parametrize('change,reason', [
    (lambda p: p[:5], 'too few'),
    (lambda p: [p[1], p[0], *p[2:]], 'strictly ordered'),
    (lambda p: [*p[:10], replace(p[10], state='interpolated'), *p[11:]], 'unobserved'),
    (lambda p: [*p[:10], replace(p[10], boundary_before='bounce'), *p[11:]], 'impact'),
    (lambda p: [*p[:10], replace(p[10], camera_id='different'), *p[11:]], 'camera'),
    (lambda p: [*p[:10], *[replace(point, timestamp_seconds=point.timestamp_seconds+1) for point in p[10:]]], 'gap'),
    (lambda p: [replace(point, x_px=float('nan')) for point in p], 'finite'),
])
def test_bad_observations_and_boundaries_withhold_speed(change, reason):
    points, _ = trajectory()
    result = reconstruct_flight_segment(known_camera(), change(points))
    assert result.status == 'unavailable' and reason in result.reason
    assert result.segment_average_physical_speed_kmh is None


def test_homography_invalid_rotation_and_lens_distortion_are_not_metric_cameras():
    points, _ = trajectory()
    for camera in (replace(known_camera(), world_to_camera_translation_m=np.eye(3)),
                   replace(known_camera(), world_to_camera_rotation=np.ones((3,3))),
                   replace(known_camera(), calibration_method='homography_only'),
                   replace(known_camera(), pixels_undistorted=False)):
        assert reconstruct_flight_segment(camera, points).status == 'unavailable'


def test_rank_deficient_negative_depth_and_excessive_perturbation_are_unavailable():
    constant = [PixelObservation(str(i), i/60, 960, 540) for i in range(20)]
    assert 'Rank-deficient' in reconstruct_flight_segment(known_camera(), constant).reason
    behind, _ = trajectory(position=(3,-50,1.5))
    result = reconstruct_flight_segment(known_camera(), behind)
    assert result.status == 'unavailable' and 'depth' in result.reason
    short, _ = trajectory(duration=.1, count=7)
    result = reconstruct_flight_segment(known_camera(), short,
        config=FlightConfig(maximum_speed_sensitivity_fraction=.01))
    assert result.status == 'unavailable' and 'ambiguous' in result.reason


def test_splitter_retains_pts_splits_impacts_and_excludes_predictions():
    points, _ = trajectory(count=30)
    points[10] = replace(points[10], boundary_before='racket_contact')
    points[20] = replace(points[20], state='predicted')
    segments = split_flight_observations(points)
    assert list(map(len, segments)) == [10,10,9]
    assert segments[1][0].timestamp_seconds == points[10].timestamp_seconds
    assert segments[-1][0].timestamp_seconds == points[21].timestamp_seconds


def test_wrong_contact_timestamp_or_unevidenced_contact_withholds_speed():
    points, _ = trajectory()
    for boundary in (FlightBoundary('racket_contact', 121., 1.5, 'misaligned'),
                     FlightBoundary('racket_contact', 120., 1.5, ''),
                     FlightBoundary('bounce', 120., 2., 'implausible-bounce')):
        assert reconstruct_flight_segment(known_camera(), points, start_boundary=boundary).status == 'unavailable'
