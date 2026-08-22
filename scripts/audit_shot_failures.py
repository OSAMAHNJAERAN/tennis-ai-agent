import json
import os

with open('data/benchmarks/cross_match_final_holdout/ground_truth_events.json') as f:
    gt_events = json.load(f)['events']
with open('data/benchmarks/cross_match_final_holdout/ground_truth_shots.json') as f:
    gt_shots = json.load(f)['shots']

print("=== SHOT CLASSIFIER FORENSIC AUDIT ===")

for vid in ['video_08', 'video_09', 'video_10']:
    s_path = f'outputs/phase6_4_qualification/final_cross_match_holdout/{vid}/shot_events.json'
    if not os.path.exists(s_path):
        continue
    with open(s_path) as f:
        pred_shots = json.load(f).get('shot_events', [])
    g_shots = gt_shots.get(vid, [])
    
    print(f"\n==================== {vid} ({len(pred_shots)} preds vs {len(g_shots)} GTs) ====================")
    for g in g_shots:
        f_best = g['frame_index']
        gt_type = g['shot_type']
        p_id = g['player_id']
        side = g['court_side']
        hand = g.get('handedness', 'RIGHT_HANDED')
        
        # Find closest predicted shot event
        closest = None
        min_d = 999
        for p in pred_shots:
            d = abs(p['frame_index'] - f_best)
            if d < min_d:
                min_d = d
                closest = p
                
        if closest and min_d <= 15:
            pred_type = closest['shot_type']
            conf = closest.get('shot_confidence', 0.0)
            src = closest.get('classification_source', 'N/A')
            is_match = (pred_type == gt_type)
            status = "CORRECT" if is_match else "MISCLASSIFIED"
            print(f"GT f{f_best:4d} [{gt_type:8s}] (P{p_id} {side:4s} {hand:12s}) -> Pred f{closest['frame_index']:4d} [{pred_type:8s}] (Conf: {conf:.2f}, Src: {src:22s}) | {status}")
        else:
            print(f"GT f{f_best:4d} [{gt_type:8s}] (P{p_id} {side:4s} {hand:12s}) -> NO PREDICTED SHOT NEARBY (Min dist: {min_d}f)")
