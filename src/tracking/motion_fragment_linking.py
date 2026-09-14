"""Experimental bidirectional motion agreement for observed player fragments."""
from collections import defaultdict
import math
import numpy as np

from src.tracking.player_fragment_linking import box_iou, link_player_fragments as validate_fragments


def velocity(rows, boundary, fps):
    nearby = [(index, box) for index, box in rows if abs(index-boundary)/fps <= .1+1e-12]
    if len(nearby) < 3:
        return None
    times = np.array([(index-boundary)/fps for index, _ in nearby])
    boxes = np.asarray([box for _, box in nearby])
    centers = (boxes[:, :2]+boxes[:, 2:])/2
    design = np.column_stack((times, np.ones(len(times))))
    coefficients = np.linalg.lstsq(design, centers, rcond=None)[0]
    # A noisy or curved boundary is insufficient evidence for a linear link.
    error = np.linalg.norm(design@coefficients-centers, axis=1).max()
    width = float(np.median(boxes[:, 2]-boxes[:, 0]))
    return coefficients[0] if error <= .25*width else None


def link_player_fragments(detections, fps):
    mapping, _ = validate_fragments(detections, fps)
    tracks = defaultdict(list)
    for index, rows in enumerate(detections):
        for row in rows:
            tracks[row['id']].append((index, row['box']))
    first_velocity = {i: velocity(rows, rows[0][0], fps) for i, rows in tracks.items()}
    last_velocity = {i: velocity(rows, rows[-1][0], fps) for i, rows in tracks.items()}
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for first, a in tracks.items():
        for second, b in tracks.items():
            elapsed = (b[0][0]-a[-1][0])/fps
            va, vb = last_velocity[first], first_velocity[second]
            if not 0 < elapsed <= .2 or va is None or vb is None:
                continue
            first_box, second_box = np.array(a[-1][1]), np.array(b[0][1])
            forward = first_box+np.tile(va*elapsed, 2)
            backward = second_box-np.tile(vb*elapsed, 2)
            forward_iou, backward_iou = box_iou(forward, second_box), box_iou(backward, first_box)
            score = min(forward_iou, backward_iou)
            if score < .5:
                continue
            edge = {'predecessor': first, 'successor': second, 'end_frame': a[-1][0], 'start_frame': b[0][0],
                    'elapsed_seconds': elapsed, 'iou': float(score), 'forward_iou': float(forward_iou),
                    'backward_iou': float(backward_iou), 'boundary_iou': box_iou(first_box, second_box),
                    'forward_velocity_px_s': va.tolist(), 'backward_velocity_px_s': vb.tolist()}
            outgoing[first].append(edge)
            incoming[second].append(edge)

    def best(edges):
        edges = sorted(edges, key=lambda r: -r['iou'])
        if len(edges) > 1 and math.isclose(edges[0]['iou'], edges[1]['iou'], rel_tol=0, abs_tol=1e-9):
            return None
        return edges[0] if edges else None

    links = [best(edges) for edges in outgoing.values() if best(edges) is not None]
    links = [edge for edge in links if best(incoming[edge['successor']]) == edge]
    links.sort(key=lambda r: (r['start_frame'], r['predecessor'], r['successor']))
    parents = {edge['successor']: edge['predecessor'] for edge in links}
    for identity in mapping:
        canonical = identity
        while canonical in parents:
            canonical = parents[canonical]
        mapping[identity] = canonical
    return mapping, links
