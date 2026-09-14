"""Experimental reconciliation of brief, consistently overlapping source IDs."""
from collections import defaultdict
import math

from src.tracking.player_fragment_linking import box_iou, link_player_fragments as validate


def reconcile_handoffs(detections, fps, evidence, prior_mapping):
    identities, _ = validate(detections, fps)
    if set(prior_mapping) != set(identities):
        raise ValueError('Prior mapping must cover exactly all source IDs')
    tracks = defaultdict(dict)
    for index, rows in enumerate(detections):
        for row in rows:
            if not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1:
                raise ValueError('Invalid person confidence')
            tracks[row['id']][index] = row['box']
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for a, first in tracks.items():
        for b, second in tracks.items():
            if not min(first) < min(second) <= max(first) < max(second):
                continue
            interval = list(range(min(second), max(first)+1))
            if len(interval)/fps > .1 or any(i not in first or i not in second for i in interval):
                continue
            overlap = min(box_iou(first[i], second[i]) for i in interval)
            if overlap < .5:
                continue
            edge = {'predecessor': a, 'successor': b, 'shared_frames': interval, 'iou': overlap}
            outgoing[a].append(edge)
            incoming[b].append(edge)

    def best(edges):
        ranked = sorted(edges, key=lambda edge: -edge['iou'])
        if len(ranked) > 1 and math.isclose(ranked[0]['iou'], ranked[1]['iou'], abs_tol=1e-9, rel_tol=0):
            return None
        return ranked[0] if ranked else None

    candidates = [best(edges) for edges in outgoing.values() if best(edges) is not None]
    candidates = [edge for edge in candidates if best(incoming[edge['successor']]) == edge]
    components = defaultdict(set)
    for identity, canonical in prior_mapping.items():
        if canonical not in identities:
            raise ValueError('Unknown prior canonical ID')
        components[canonical].add(identity)
    mapping = dict(prior_mapping)
    accepted, rejected = [], []
    for edge in sorted(candidates, key=lambda e: (e['shared_frames'][0], e['predecessor'], e['successor'])):
        first, second = mapping[edge['predecessor']], mapping[edge['successor']]
        members = components[first] | components[second]
        conflict = False
        for a in members:
            for b in members:
                if a >= b:
                    continue
                shared = set(tracks[a]) & set(tracks[b])
                if any(box_iou(tracks[a][i], tracks[b][i]) < .5 for i in shared):
                    conflict = True
        if conflict:
            rejected.append({**edge, 'reason': 'CHAIN_HAS_DISTINCT_CONCURRENT_BOXES'})
            continue
        canonical = min(members, key=lambda identity: (min(tracks[identity]), identity))
        components.pop(first, None)
        components.pop(second, None)
        components[canonical] = members
        for identity in members:
            mapping[identity] = canonical
        accepted.append(edge)
    remapped, suppressions = [], []
    for index, rows in enumerate(detections):
        groups = defaultdict(list)
        for row in rows:
            groups[mapping[row['id']]].append(row)
        converted = []
        for identity, group in sorted(groups.items()):
            group.sort(key=lambda row: (-row['confidence'], row['id']))
            kept = group[0]
            converted.append({**kept, 'id': identity, 'source_id': kept['id']})
            if len(group) > 1:
                suppressions.append({'frame': index, 'canonical_id': identity, 'kept_source_id': kept['id'],
                                     'suppressed_source_ids': [r['id'] for r in group[1:]]})
        remapped.append(converted)
    supports = {}
    for row in evidence:
        key = row['frame'], mapping[row['source_id']]
        if row['frame'] not in tracks[row['source_id']] or not math.isfinite(row['confidence']) or not .25 <= row['confidence'] <= 1:
            raise ValueError('Invalid racket evidence')
        changed = {**row, 'source_id': key[1], 'original_source_id': row['source_id']}
        if key not in supports or (changed['confidence'], -changed['original_source_id']) > (supports[key]['confidence'], -supports[key]['original_source_id']):
            supports[key] = changed
    return remapped, [supports[k] for k in sorted(supports)], mapping, accepted, rejected, suppressions
