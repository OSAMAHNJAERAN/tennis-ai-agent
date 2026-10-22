"""Estimated monocular ballistic flights with a metric camera, never a ground homography.

Known gravity constrains one impact-free segment. Drag, wind and spin are neglected.
Fit residuals/perturbation sensitivity are numerical diagnostics, not independent
physical accuracy or confidence intervals. All successful estimates are unqualified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class MetricCamera:
    intrinsic: Any
    world_to_camera_rotation: Any
    world_to_camera_translation_m: Any
    camera_id: str
    calibration_source: str
    calibration_method: str = "independent_intrinsic_extrinsic"
    pixels_undistorted: bool = True

    def matrices(self):
        k = np.asarray(self.intrinsic, dtype=float)
        r = np.asarray(self.world_to_camera_rotation, dtype=float)
        t = np.asarray(self.world_to_camera_translation_m, dtype=float)
        if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
            raise ValueError("Metric intrinsics, 3D rotation and translation are required; homography alone is insufficient.")
        if not all(np.isfinite(value).all() for value in (k, r, t)):
            raise ValueError("Camera parameters must be finite.")
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
            raise ValueError("Camera rotation must be a proper orthonormal metric rotation.")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or abs(k[1, 0]) > 1e-9 or not np.allclose(k[2], [0, 0, 1], atol=1e-9):
            raise ValueError("Pinhole camera intrinsics must have positive focal lengths.")
        if self.calibration_method not in ("independent_intrinsic_extrinsic", "synthetic_known_camera"):
            raise ValueError("Camera calibration cannot be supplied by a ground homography alone.")
        if not self.camera_id or not self.calibration_source or not self.pixels_undistorted:
            raise ValueError("Camera provenance and undistorted observed pixels are required.")
        return k, r, t, k @ np.column_stack((r, t))


@dataclass(frozen=True)
class PixelObservation:
    observation_id: str
    timestamp_seconds: float
    x_px: float
    y_px: float
    state: str = "observed"
    camera_id: str = "camera-1"
    boundary_before: str | None = None  # cut/racket_contact/bounce/gap


@dataclass(frozen=True)
class FlightBoundary:
    kind: str  # racket_contact/bounce
    timestamp_seconds: float
    height_m: float
    evidence_id: str
    height_tolerance_m: float = .25
    position_xy_m: tuple[float, float] | None = None
    position_tolerance_m: float = .5


@dataclass(frozen=True)
class FlightConfig:
    gravity_mps2: float = 9.80665
    minimum_observations: int = 6
    minimum_duration_seconds: float = .10
    maximum_gap_seconds: float = .10
    maximum_duration_seconds: float = 2.5
    maximum_condition_number: float = 1e6
    maximum_reprojection_error_px: float = 2.0
    maximum_height_m: float = 20.0
    maximum_speed_kmh: float = 400.0
    numerical_pixel_perturbation_px: float = .5
    maximum_speed_sensitivity_fraction: float = .25

    def __post_init__(self):
        scalars = [value for key, value in asdict(self).items() if key != 'minimum_observations']
        if not all(np.isfinite(value) and value > 0 for value in scalars) or self.minimum_observations < 6:
            raise ValueError("Flight limits must be positive finite values and require at least six observations.")


@dataclass
class FlightEstimate:
    segment_id: str
    status: str
    reason: str
    observation_ids: list[str]
    sample_count: int
    start_seconds: float | None = None
    end_seconds: float | None = None
    positions_m: list[list[float]] | None = None
    initial_velocity_mps: list[float] | None = None
    segment_average_physical_speed_kmh: float | None = None
    racket_impact_speed_kmh: float | None = None
    impact_timestamp_seconds: float | None = None
    impact_evidence_id: str | None = None
    impact_speed_side: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    uncertainty_interval: None = None
    calibration_source: str | None = None
    qualification_status: str = "unqualified_no_independent_synchronized_reference"
    assumptions: str = "Metric pinhole camera; known gravity in world +z up; drag, spin and wind neglected; one impact-free segment."

    def to_dict(self):
        return asdict(self)


def split_flight_observations(observations: Iterable[PixelObservation], *, maximum_gap_seconds=.10):
    """Split on explicit impacts/cuts, unknown observations, camera changes and real PTS gaps."""
    if not np.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
        raise ValueError("Gap duration must be a finite positive value.")
    segments, current = [], []
    previous_time = None
    seen = set()
    for observation in observations:
        if not np.isfinite(observation.timestamp_seconds) or observation.timestamp_seconds < 0:
            raise ValueError("Observation timestamp must be finite and nonnegative.")
        if previous_time is not None and observation.timestamp_seconds <= previous_time:
            raise ValueError("Ball observations must be strictly ordered in original source time.")
        if not observation.observation_id or observation.observation_id in seen:
            raise ValueError("Observation IDs must be unique and nonempty.")
        seen.add(observation.observation_id)
        previous_time = observation.timestamp_seconds
        if current and (observation.boundary_before is not None or observation.camera_id != current[-1].camera_id
                        or observation.timestamp_seconds-current[-1].timestamp_seconds > maximum_gap_seconds):
            segments.append(current)
            current = []
        if observation.state != "observed":
            if current:
                segments.append(current)
                current = []
            continue
        current.append(observation)
    if current:
        segments.append(current)
    return segments


def _linear_fit(projection, times, pixels, gravity, height_constraints):
    duration = times[-1]
    rows, values = [], []
    for dt, (x, y) in zip(times, pixels):
        for axis, coordinate in ((0, x), (1, y)):
            ray = projection[axis, :3] - coordinate * projection[2, :3]
            norm = np.linalg.norm(ray)
            rows.append(np.concatenate((ray, ray * dt/duration)) / norm)
            constant = projection[axis, 3] - coordinate * projection[2, 3]
            values.append((-constant - .5*dt*dt*np.dot(ray, gravity)) / norm)
    for dt, height in height_constraints:
        rows.append(np.array([0., 0., 1., 0., 0., dt/duration]))
        values.append(height - .5*dt*dt*gravity[2])
    a, b = np.asarray(rows), np.asarray(values)
    solution, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf')
    return solution[:3], solution[3:]/duration, rank, condition


def _speed_average(velocity, gravity, duration):
    # Fixed Gauss-Legendre quadrature integrates travelled arc length / time,
    # rather than endpoint displacement (which loses vertical curvature).
    locations, weights = np.polynomial.legendre.leggauss(32)
    times = (locations+1) * duration/2
    speeds = np.linalg.norm(velocity[None, :] + times[:, None]*gravity[None, :], axis=1)
    return float(np.dot(weights, speeds)/2*3.6)


def reconstruct_flight_segment(camera: MetricCamera, observations: Iterable[PixelObservation], *,
                               segment_id="flight-1", start_boundary: FlightBoundary | None = None,
                               end_boundary: FlightBoundary | None = None,
                               config: FlightConfig | None = None) -> FlightEstimate:
    """Fit one observed, impact-free projectile; success always remains estimated/unqualified."""
    config = config or FlightConfig()
    points = list(observations)
    result = FlightEstimate(segment_id, "unavailable", "", [p.observation_id for p in points], len(points),
                            calibration_source=camera.calibration_source)
    def reject(reason, **diagnostics):
        result.reason = reason
        result.diagnostics.update(diagnostics)
        return result
    try:
        _, rotation, translation, projection = camera.matrices()
    except ValueError as error:
        return reject(str(error))
    if len(points) < config.minimum_observations:
        return reject("Underconstrained: too few independent observed points.")
    try:
        segments = split_flight_observations(points, maximum_gap_seconds=config.maximum_gap_seconds)
    except ValueError as error:
        return reject(str(error))
    if len(segments) != 1 or len(segments[0]) != len(points) or points[0].camera_id != camera.camera_id:
        return reject("Segment crosses an impact, cut, camera change, unobserved sample or time gap.")
    times = np.array([p.timestamp_seconds-points[0].timestamp_seconds for p in points])
    duration = float(times[-1])
    result.start_seconds, result.end_seconds = points[0].timestamp_seconds, points[-1].timestamp_seconds
    if duration < config.minimum_duration_seconds-1e-9 or duration > config.maximum_duration_seconds+1e-9:
        return reject("Flight duration is outside the supported fit window.")
    pixels = np.array([[p.x_px, p.y_px] for p in points], dtype=float)
    if not np.isfinite(pixels).all():
        return reject("Observed pixel coordinates must be finite.")
    boundaries, heights = [], []
    for boundary, expected_time in ((start_boundary, points[0].timestamp_seconds), (end_boundary, points[-1].timestamp_seconds)):
        if boundary is None:
            continue
        if boundary.kind not in ("racket_contact", "bounce") or not boundary.evidence_id:
            return reject("Impact/bounce constraints require a supported type and evidence ID.")
        if not all(np.isfinite(value) for value in (boundary.timestamp_seconds, boundary.height_m, boundary.height_tolerance_m)):
            return reject("Boundary constraint parameters must be finite.")
        if boundary.height_tolerance_m <= 0 or boundary.height_m < 0 or not np.isclose(boundary.timestamp_seconds, expected_time, atol=1e-6, rtol=0):
            return reject("A constraint must coincide with an observed segment endpoint and have valid height.")
        if boundary.kind == "bounce" and not 0 <= boundary.height_m <= .08:
            return reject("Bounce constraint must describe ball-centre contact height near the ground.")
        boundaries.append(boundary)
        # Only independently supplied endpoint heights constrain the solution.
        heights.append((boundary.timestamp_seconds-points[0].timestamp_seconds, boundary.height_m))
    gravity = np.array([0., 0., -config.gravity_mps2])
    position, velocity, rank, condition = _linear_fit(projection, times, pixels, gravity, heights)
    result.diagnostics.update(rank=int(rank), condition_number=condition if np.isfinite(condition) else None)
    if rank < 6 or condition > config.maximum_condition_number or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        return reject("Rank-deficient or ill-conditioned monocular flight; metric depth remains ambiguous.")
    positions = position + times[:, None]*velocity + .5*times[:, None]**2*gravity
    camera_points = positions @ rotation.T + translation
    if np.any(camera_points[:, 2] <= 0):
        return reject("Estimated trajectory has nonpositive camera depth.")
    image = np.column_stack((positions, np.ones(len(positions)))) @ projection.T
    projected = image[:, :2] / image[:, 2, None]
    errors = np.linalg.norm(projected-pixels, axis=1)
    result.diagnostics.update(reprojection_median_px=float(np.median(errors)), reprojection_max_px=float(np.max(errors)))
    if np.max(errors) > config.maximum_reprojection_error_px:
        return reject("Flight violates observed reprojection consistency; segmentation or ballistic assumptions may be wrong.")
    # Check continuous interval endpoints and the interior height apex, not only sampled frames.
    apex_time = np.clip(-velocity[2]/gravity[2], 0, duration)
    apex_height = position[2] + apex_time*velocity[2] + .5*apex_time**2*gravity[2]
    if np.min(positions[:, 2]) < -1e-6 or apex_height > config.maximum_height_m:
        return reject("Estimated flight has unsupported/negative physical height.")
    for boundary in boundaries:
        dt = boundary.timestamp_seconds-points[0].timestamp_seconds
        endpoint = position + dt*velocity + .5*dt*dt*gravity
        if abs(endpoint[2]-boundary.height_m) > boundary.height_tolerance_m:
            return reject("Estimated flight conflicts with independently supplied contact/bounce height.")
        if boundary.position_xy_m is not None:
            xy = np.asarray(boundary.position_xy_m, dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(boundary.position_tolerance_m) or boundary.position_tolerance_m <= 0:
                return reject("Boundary ground-position constraint is invalid.")
            if np.linalg.norm(endpoint[:2]-xy) > boundary.position_tolerance_m:
                return reject("Estimated flight conflicts with independent contact/bounce ground position.")
    average = _speed_average(velocity, gravity, duration)
    maximum_speed = max(np.linalg.norm(velocity), np.linalg.norm(velocity+gravity*duration))*3.6
    if maximum_speed > config.maximum_speed_kmh:
        return reject("Estimated speed exceeds the declared physical plausibility limit.")
    # Deterministic pixel stress is a sensitivity test, explicitly not a confidence interval.
    perturbations = []
    for axis in (0, 1):
        for pattern in (np.ones(len(points)), (-1.)**np.arange(len(points))):
            for sign in (-1, 1):
                perturbed = pixels.copy()
                perturbed[:, axis] += sign*pattern*config.numerical_pixel_perturbation_px
                stress_position, stress_velocity, stress_rank, _ = _linear_fit(projection, times, perturbed, gravity, heights)
                if stress_rank < 6:
                    return reject("Pixel perturbation exposes an ambiguous reconstruction.")
                stress_positions = stress_position + times[:, None]*stress_velocity + .5*times[:, None]**2*gravity
                if np.any((stress_positions @ rotation.T + translation)[:, 2] <= 0):
                    return reject("Pixel perturbation exposes ambiguous camera depth.")
                perturbations.append(_speed_average(stress_velocity, gravity, duration))
                perturbations.append(float(np.linalg.norm(stress_velocity-velocity)*3.6 + average))
    sensitivity = max(abs(value-average) for value in perturbations) / max(average, .01)
    result.diagnostics.update(pixel_stress_px=config.numerical_pixel_perturbation_px,
                              maximum_relative_speed_sensitivity=float(sensitivity),
                              diagnostic_status="numerical_only_not_physical_uncertainty")
    if sensitivity > config.maximum_speed_sensitivity_fraction:
        return reject("Small pixel perturbations change speed excessively; physical reconstruction remains ambiguous.")
    result.status = "estimated_unqualified"
    result.reason = "Ballistic estimate passed numerical/physical constraints; independent synchronized speed qualification is pending."
    result.positions_m = positions.tolist()
    result.initial_velocity_mps = velocity.tolist()
    result.segment_average_physical_speed_kmh = average
    # Impact speed only when the measured interval starts/ends at evidenced racket contact.
    contact = next((boundary for boundary in boundaries if boundary.kind == "racket_contact"), None)
    if contact:
        dt = contact.timestamp_seconds-points[0].timestamp_seconds
        result.racket_impact_speed_kmh = float(np.linalg.norm(velocity+dt*gravity)*3.6)
        result.impact_timestamp_seconds = contact.timestamp_seconds
        result.impact_evidence_id = contact.evidence_id
        result.impact_speed_side = "outgoing" if np.isclose(contact.timestamp_seconds, result.start_seconds, atol=1e-6, rtol=0) else "incoming"
    return result
