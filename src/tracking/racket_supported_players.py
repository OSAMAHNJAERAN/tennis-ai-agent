"""Offline person-track selection using repeated visual racket evidence."""
from collections import Counter,defaultdict
import math


def select_racket_supported_people(detections,fps,sampled_frames,racket_evidence):
    if not math.isfinite(fps) or fps<=0:
        raise ValueError('Invalid frame rate')
    if len(set(sampled_frames))!=len(sampled_frames) or any(not isinstance(i,int) or not 0<=i<len(detections) for i in sampled_frames):
        raise ValueError('Invalid sampled frames')
    sampled=set(sampled_frames)
    counts,opportunities=Counter(),Counter()
    observed=set()
    for index,rows in enumerate(detections):
        ids=set()
        for row in rows:
            identity=row['id'];box=row['box'];confidence=row['confidence']
            if identity in ids or len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=box[0] or box[3]<=box[1] or not math.isfinite(confidence):
                raise ValueError('Invalid or duplicate person observation')
            ids.add(identity);observed.add((index,identity));counts[identity]+=1
            if index in sampled:opportunities[identity]+=1
    support=defaultdict(list)
    seen=set()
    for row in racket_evidence:
        key=row['frame'],row['source_id'];confidence=row['confidence']
        if key in seen or key not in observed or row['frame'] not in sampled or not math.isfinite(confidence) or not .25<=confidence<=1:
            raise ValueError('Invalid, duplicate or unobserved racket support')
        seen.add(key);support[row['source_id']].append(confidence)
    statistics={}
    for identity in counts:
        values=support[identity]
        statistics[identity]={'observed_frames':counts[identity],'sampled_opportunities':opportunities[identity],
                              'support_frames':len(values),'score':sum(values)/(opportunities[identity]+5),
                              'eligible':counts[identity]/fps>=.5 and len(values)>=2}
    selected=[]
    for rows in detections:
        eligible=[r for r in rows if statistics[r['id']]['eligible']]
        eligible.sort(key=lambda r:(-statistics[r['id']]['score'],-statistics[r['id']]['support_frames'],-r['confidence'],r['id']))
        selected.append([dict(r) for r in eligible[:2]])
    return selected,statistics
