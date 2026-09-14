"""Observed image correspondences for experimental short-gap identity linking."""
from collections import defaultdict
import math

import cv2
import numpy as np

from src.tracking.player_fragment_linking import box_iou, link_player_fragments as validate_fragments


def measured_translation(first, second, box):
    """Return native-pixel displacement only with distributed consistent evidence."""
    if first.shape != second.shape or first.ndim != 2 or first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError('Require equally sized uint8 grayscale images')
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Invalid person box')
    height, width = first.shape
    size = box[2:]-box[:2]
    low = np.ceil(box[:2]+.05*size).astype(int)
    high = np.floor(box[2:]-.05*size).astype(int)
    low = np.maximum(low, 0)
    high = np.minimum(high, [width, height])
    audit = {'initial_features': 0, 'consistent_features': 0, 'inliers': 0, 'displacement_px': None}
    if np.any(high <= low):
        return {**audit, 'reason': 'EMPTY_FEATURE_REGION'}
    mask = np.zeros_like(first)
    mask[low[1]:high[1], low[0]:high[0]] = 255
    points = cv2.goodFeaturesToTrack(first, maxCorners=80, qualityLevel=.01, minDistance=2, mask=mask, blockSize=3)
    if points is None or len(points) < 4:
        return {**audit, 'initial_features': 0 if points is None else len(points), 'reason': 'INSUFFICIENT_FEATURES'}
    audit['initial_features'] = len(points)
    params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    moved, status, error = cv2.calcOpticalFlowPyrLK(first, second, points, None, **params)
    if moved is None:
        return {**audit, 'reason': 'FORWARD_FLOW_FAILED'}
    returned, backward_status, backward_error = cv2.calcOpticalFlowPyrLK(second, first, moved, None, **params)
    if returned is None:
        return {**audit, 'reason': 'BACKWARD_FLOW_FAILED'}
    p, q, back = points.reshape(-1, 2), moved.reshape(-1, 2), returned.reshape(-1, 2)
    valid = (status.ravel() == 1) & (backward_status.ravel() == 1)
    valid &= np.isfinite(q).all(axis=1) & np.isfinite(back).all(axis=1)
    valid &= (np.linalg.norm(back-p, axis=1) <= 1.5) & (error.ravel() <= 20) & (backward_error.ravel() <= 20)
    valid &= (q[:, 0] >= 0) & (q[:, 0] < width) & (q[:, 1] >= 0) & (q[:, 1] < height)
    audit['consistent_features'] = int(valid.sum())
    if valid.sum() < 4:
        return {**audit, 'reason': 'INSUFFICIENT_CONSISTENT_FLOW'}
    displacement = q-p
    median = np.median(displacement[valid], axis=0)
    inliers = valid & (np.linalg.norm(displacement-median, axis=1) <= 2)
    audit['inliers'] = int(inliers.sum())
    if inliers.sum() < 4 or inliers.sum() < .5*len(points):
        return {**audit, 'reason': 'INSUFFICIENT_COHERENT_FLOW'}
    span = np.ptp(p[inliers], axis=0)/size
    audit['source_span_fraction'] = span.tolist()
    audit['source_points'] = p[inliers].tolist()
    audit['target_points'] = q[inliers].tolist()
    if np.any(span < .25):
        return {**audit, 'reason': 'FEATURES_TOO_LOCALIZED'}
    audit.update(reason=None, displacement_px=np.median(displacement[inliers], axis=0).tolist())
    return audit


def link_player_fragments(detections, fps, frames):
    mapping, _ = validate_fragments(detections, fps)
    if len(frames) != len(detections):
        raise ValueError('Video and detections differ in length')
    tracks = defaultdict(list)
    for index, rows in enumerate(detections):
        for row in rows:
            tracks[row['id']].append((index, row['box']))
    outgoing, incoming, diagnostics = defaultdict(list), defaultdict(list), []
    for first, a in tracks.items():
        for second, b in tracks.items():
            elapsed = (b[0][0]-a[-1][0])/fps
            if not 0 < elapsed <= .2:
                continue
            before = cv2.cvtColor(frames[a[-1][0]], cv2.COLOR_BGR2GRAY)
            after = cv2.cvtColor(frames[b[0][0]], cv2.COLOR_BGR2GRAY)
            forward = measured_translation(before, after, a[-1][1])
            backward = measured_translation(after, before, b[0][1])
            edge = {'predecessor': first, 'successor': second, 'end_frame': a[-1][0], 'start_frame': b[0][0],
                    'elapsed_seconds': elapsed, 'forward': forward, 'backward': backward, 'eligible': False}
            diagnostics.append(edge)
            if forward['displacement_px'] is None or backward['displacement_px'] is None:
                continue
            fa = np.array(a[-1][1])+np.tile(forward['displacement_px'], 2)
            ba = np.array(b[0][1])+np.tile(backward['displacement_px'], 2)
            fi, bi = box_iou(fa, b[0][1]), box_iou(ba, a[-1][1])
            edge.update(forward_iou=fi, backward_iou=bi, iou=min(fi, bi))
            if min(fi, bi) < .5:
                continue
            edge['eligible'] = True
            outgoing[first].append(edge)
            incoming[second].append(edge)

    def best(edges):
        edges = sorted(edges, key=lambda edge: -edge['iou'])
        if len(edges) > 1 and math.isclose(edges[0]['iou'], edges[1]['iou'], rel_tol=0, abs_tol=1e-9):
            return None
        return edges[0] if edges else None

    links = [best(edges) for edges in outgoing.values() if best(edges) is not None]
    links = [edge for edge in links if best(incoming[edge['successor']]) == edge]
    links.sort(key=lambda edge: (edge['start_frame'], edge['predecessor'], edge['successor']))
    parents = {edge['successor']: edge['predecessor'] for edge in links}
    for identity in mapping:
        canonical = identity
        while canonical in parents:
            canonical = parents[canonical]
        mapping[identity] = canonical
    return mapping, links, diagnostics
