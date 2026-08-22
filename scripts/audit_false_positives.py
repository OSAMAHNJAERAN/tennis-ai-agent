import json
import os

with open('data/benchmarks/cross_match_final_holdout/ground_truth_events.json') as f:
    gt_events = json.load(f)['events']
with open('data/benchmarks/cross_match_final_holdout/ground_truth_shots.json') as f:
    gt_shots = json.load(f)['shots']

print("=== DETAILED EVENT MATCHING & FALSE POSITIVE FORENSICS ===")
categories = {}

for vid in ['video_08', 'video_09', 'video_10']:
    p_path = f'outputs/phase6_4_qualification/final_cross_match_holdout/{vid}/match_events.json'
    if not os.path.exists(p_path):
        continue
    with open(p_path) as f:
        preds = json.load(f).get('events', [])
    gts = gt_events.get(vid, [])
    
    print(f"\n==================== {vid} ====================")
    matched_gts = set()
    
    for p in preds:
        f_idx = p['frame']
        e_type = p['event_type']
        p_id = p.get('player_id')
        is_dead = p.get('is_dead_ball', False)
        
        # Check matching against unmatched GTs
        match = None
        for g in gts:
            if g['event_id'] in matched_gts:
                continue
            if abs(g['frame_best'] - f_idx) <= 6 and g['event_type'] == e_type:
                match = g
                matched_gts.add(g['event_id'])
                break
                
        if match:
            print(f"TP: Frame {f_idx:4d} | {e_type:15s} | Matched GT #{match['event_id']} (f{match['frame_best']})")
        else:
            closest_gt = min(gts, key=lambda g: abs(g['frame_best'] - f_idx))
            dist = closest_gt['frame_best'] - f_idx
            # Classify reason
            if is_dead:
                cat = "POST_RALLY_DEAD_BALL"
            elif closest_gt['event_type'] != e_type and abs(dist) <= 6:
                if e_type == "PLAYER_HIT" and closest_gt['event_type'] == "BOUNCE":
                    cat = "BOUNCE_AS_HIT"
                elif e_type == "BOUNCE" and closest_gt['event_type'] == "PLAYER_HIT":
                    cat = "HIT_AS_BOUNCE"
                else:
                    cat = "EVENT_TYPE_CONFUSION"
            elif abs(dist) > 20:
                cat = "BALL_NOISE_OR_ARTIFACT"
            else:
                cat = "TIMING_DRIFT_OR_JITTER"
                
            categories[cat] = categories.get(cat, 0) + 1
            print(f"FP: Frame {f_idx:4d} | {e_type:15s} | P{p_id} | Dead:{str(is_dead):5s} | Cat: {cat:22s} | Near GT: {closest_gt['event_type']} @ f{closest_gt['frame_best']} ({dist:+d}f)")

print("\n=== FALSE POSITIVE CATEGORY TOTALS ===")
for c, cnt in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {c:25s}: {cnt}")
