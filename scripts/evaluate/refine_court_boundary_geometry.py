"""Research-only refinement that preserves outer lines without direct evidence."""
import numpy as np
from scipy.optimize import least_squares

from scripts.evaluate.audit_caltennis_projection import LINES
from scripts.evaluate.audit_court_line_support import support


def project(points, matrix):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if np.any(homogeneous[:, 2] <= .1):
        raise ValueError('Refinement approaches or crosses the projective horizon')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def match_lines(points, segments):
    """Assign finite candidate segments locally, preserving canonical line IDs."""
    matches = []
    for identity, (a, b) in enumerate(LINES):
        start, end = points[a], points[b]
        vector = end-start
        length = np.linalg.norm(vector)
        if length < 30:
            continue
        direction = vector/length
        choices = []
        for index, segment in enumerate(segments):
            first, last = segment
            delta = last-first
            size = np.linalg.norm(delta)
            if size < 30:
                continue
            tangent = delta/size
            angle = np.degrees(np.arccos(np.clip(abs(tangent @ direction), 0, 1)))
            if angle > 6:
                continue
            along = sorted([(first-start) @ direction, (last-start) @ direction])
            overlap = max(0., min(length, along[1])-max(0., along[0]))
            if overlap < min(length, size)*.5 or overlap < 30:
                continue
            normal = np.array([-tangent[1], tangent[0]])
            offsets = np.abs((np.stack((start, end))-first) @ normal)
            if offsets.max() > 20:
                continue
            cost = float(offsets.mean() + angle*.5)
            choices.append((cost, index, normal, float(-normal @ first)))
        if choices:
            cost, index, normal, offset = min(choices, key=lambda item: (item[0], item[1]))
            matches.append({'line': identity, 'segment': index, 'cost': cost,
                            'normal': normal.tolist(), 'offset': offset})
    # One image segment cannot support two different semantic court lines.
    matches.sort(key=lambda item: item['cost'])
    used, unique = set(), []
    for match in matches:
        if match['segment'] not in used:
            used.add(match['segment'])
            unique.append(match)
    return sorted(unique, key=lambda item: item['line'])



def boundary_nullspace(points, locked_lines):
    """Linear homography updates preserving each unobserved outer line.

    For original endpoint X and original line l, l.T @ H @ X = 0.
    Identity satisfies this equation, so valid updates lie in its nullspace.
    Coordinates are normalized to the existing 960-pixel reference scale.
    """
    normalized = np.asarray(points, dtype=float)/960.
    equations = []
    for identity in sorted(locked_lines):
        a, b = LINES[identity]
        first, last = normalized[a], normalized[b]
        delta = last-first
        length = np.linalg.norm(delta)
        if length <= 1e-10:
            raise ValueError('Cannot constrain a degenerate court boundary')
        nx, ny = np.array([-delta[1], delta[0]])/length
        offset = -np.dot([nx, ny], first)
        for x, y in (first, last):
            equations.append([nx*x, nx*y, nx, ny*x, ny*y, ny, offset*x, offset*y])
    if not equations:
        return np.eye(8)
    _, singular, vectors = np.linalg.svd(np.asarray(equations), full_matrices=True)
    rank = int(np.count_nonzero(singular > 1e-10))
    return vectors[rank:].T


def refine(points, segments, width=960, height=540):
    """Fit a single image homography; return unchanged points on rejection.

    Inputs are already projected canonical landmarks at reference width960.
    A small residual measures image-line agreement, not independent accuracy.
    """
    original = np.asarray(points, dtype=float)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if original.shape != (14, 2) or not np.isfinite(original).all() or not np.isfinite(segments).all():
        raise ValueError('Require fourteen finite projected points and finite segments')
    before = support(original, segments, width, height)
    current = original.copy()
    history = []
    locked_lines = set()
    result = {'accepted': False, 'reason': 'INSUFFICIENT_COMPATIBLE_LINES',
              'before': before, 'iterations': history}
    for _ in range(3):
        matches = match_lines(current, segments)
        ids = {m['line'] for m in matches}
        if len(matches) < 6 or len(ids & {0,1,6,7}) < 2 or len(ids & {2,3,4,5,8}) < 2:
            return original.copy(), result
        # Once a boundary lacks evidence, retain its original infinite line.
        # Later assignment changes cannot unlock it within this fit.
        locked_lines.update({0, 1, 2, 3}-ids)
        basis = boundary_nullspace(original, locked_lines)
        if basis.shape[1] == 0:
            result['reason'] = 'NO_BOUNDARY_PRESERVING_DEGREES_OF_FREEDOM'
            return original.copy(), result
        normalized = original/960.
        def matrix(parameters):
            update = basis @ parameters
            return np.append(np.eye(3).ravel()[:8]+update, 1.).reshape(3,3)
        def residual(parameters):
            try:
                moved = project(normalized, matrix(parameters))*960.
            except ValueError:
                return np.full(len(matches)*11+28, 1e6)
            values = []
            for match in matches:
                a,b = LINES[match['line']]
                samples = moved[a] + np.linspace(.05,.95,11)[:,None]*(moved[b]-moved[a])
                values.extend(samples @ np.asarray(match['normal']) + match['offset'])
            # A weak image-space prior discourages drift when lines are ambiguous.
            values.extend(((moved-original)*.05).ravel())
            return np.asarray(values)
        fit = least_squares(residual, np.zeros(basis.shape[1]), loss='soft_l1', f_scale=2.,
                            max_nfev=150, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        try:
            current = project(normalized, matrix(fit.x))*960.
        except ValueError:
            result['reason'] = 'INVALID_PROJECTIVE_TRANSFORM'
            return original.copy(), result
        displacement = float(np.linalg.norm(current-original, axis=1).max())
        history.append({'matches': matches, 'solver_success': bool(fit.success),
                        'locked_boundary_lines': sorted(locked_lines),
                        'free_transform_parameters': int(basis.shape[1]),
                        'maximum_displacement_reference_px': displacement,
                        'normalized_image_transform': matrix(fit.x).tolist()})
        if not fit.success or displacement > 20:
            result['reason'] = 'UNSTABLE_OR_EXCESSIVE_REFINEMENT'
            return original.copy(), result
    after = support(current, segments, width, height)
    result['proposed_after'] = after
    if (not after['diagnostic_gate_pass'] or
            sum(row['supported_fraction'] >= .5 for row in after['lines']) < 6 or
            after['mean_visible_line_support'] < before['mean_visible_line_support']+.02):
        result['reason'] = 'INSUFFICIENT_IMAGE_SUPPORT_IMPROVEMENT'
        return original.copy(), result
    result.update(accepted=True, reason='PROJECTIVE_IMAGE_SUPPORT_IMPROVED')
    return current, result
