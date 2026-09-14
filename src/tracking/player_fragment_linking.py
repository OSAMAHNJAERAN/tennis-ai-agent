"""Offline boundary-IoU linking of non-overlapping source tracks."""
from collections import defaultdict
import math


def box_iou(first, second):
    overlap = max(0, min(first[2], second[2])-max(first[0], second[0])) * max(0, min(first[3], second[3])-max(first[1], second[1]))
    union = (first[2]-first[0])*(first[3]-first[1]) + (second[2]-second[0])*(second[3]-second[1])-overlap
    return overlap/union


def link_player_fragments(detections, fps):
    """Return canonical identities and accepted edges without synthesizing boxes."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid frame rate')
    tracks = {}
    for index, rows in enumerate(detections):
        seen = set()
        for row in rows:
            identity, box = row['id'], row['box']
            if not isinstance(identity, int) or isinstance(identity, bool) or identity < 0 or identity in seen:
                raise ValueError('Invalid or duplicate source identity')
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid source box')
            seen.add(identity)
            if identity not in tracks:
                tracks[identity] = {'start': index, 'first_box': box}
            tracks[identity].update(end=index, last_box=box)
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for first, a in tracks.items():
        for second, b in tracks.items():
            gap = b['start']-a['end']
            if not 0 < gap/fps <= .2:
                continue
            overlap = box_iou(a['last_box'], b['first_box'])
            if overlap < .5:
                continue
            edge = {'predecessor': first, 'successor': second, 'end_frame': a['end'],
                    'start_frame': b['start'], 'elapsed_seconds': gap/fps, 'iou': overlap}
            outgoing[first].append(edge)
            incoming[second].append(edge)

    def unique_best(edges):
        ranked = sorted(edges, key=lambda edge: -edge['iou'])
        if len(ranked) > 1 and math.isclose(ranked[0]['iou'], ranked[1]['iou'], rel_tol=0, abs_tol=1e-9):
            return None
        return ranked[0] if ranked else None

    links = []
    for edges in outgoing.values():
        edge = unique_best(edges)
        if edge is not None and unique_best(incoming[edge['successor']]) == edge:
            links.append(edge)
    links.sort(key=lambda edge: (edge['start_frame'], edge['predecessor'], edge['successor']))
    parents = {edge['successor']: edge['predecessor'] for edge in links}
    mapping = {}
    for identity in tracks:
        canonical = identity
        while canonical in parents:
            canonical = parents[canonical]
        mapping[identity] = canonical
    return mapping, links


def remap_observations(detections, racket_evidence, mapping):
    remapped = []
    for rows in detections:
        converted = [{**row, 'id': mapping[row['id']], 'source_id': row['id']} for row in rows]
        if len({r['id'] for r in converted}) != len(converted):
            raise ValueError('Linked source tracks overlap in time')
        remapped.append(converted)
    evidence = [{**row, 'source_id': mapping[row['source_id']], 'original_source_id': row['source_id']} for row in racket_evidence]
    return remapped, evidence
