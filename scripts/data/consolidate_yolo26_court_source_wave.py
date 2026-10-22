"""Hash-bound court64 review consolidation; static private development only."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = 'artifacts/validation/yolo26_execution_2026-10-15/'
INPUTS = {
    BASE+'court_wave64_native_review_aids_v1/review_aids_manifest.json': '3f4112bb190e6e8277a3406ea329d95868356237745c97f44ac01a9056729366',
    BASE+'court_wave64_native_review_first32_agent_v1/review_v2.json': '3ac7a1207bd464cdc5e6f01642cce25f63639055b203e29d9eed982564a54032',
    BASE+'court_wave64_native_review_last32_root_v1/review.json': '934a473dbb3ba1ebff6a514c10f85891d0c2d165adbbf73ae9b6ddbb65378abb',
    'data/yolo26/court_review_decisions_v1.json': 'b085770df3330b89a8371806d2376ea9106db9f2f82d4e5436c78f5af7a634cb',
    'data/yolo26/court_development_frozen_v1/grouped_development_manifest.json': '35ae2e741c6e8abdd6eb9b0ed516316be2ec5b1f73ef257b61867ea9e5cd851e',
    'data/yolo26/court_reviewed_development_v1/source_manifest.json': '0fadd340acc2fdfa5db9083474f620b396eea1d1eaf8fe025fdf050e43c3c90f',
    'data/external/court_source_wave64_v1/manifest.json': '8fc0aee1883f10b69dd56e17858bb9d66b5604b65b0a51c0d46f560cf37676a5',
}

def sha(path):
    with path.open('rb') as stream:
        result = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result

def context_components(old_rows, old_groups, new_rows):
    """Retain every prior/rejected link and make prior DEV membership monotone."""
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def join(left, right):
        a, b = find(left), find(right)
        parent[max(a,b)] = min(a,b)
    for group_id, group in old_groups.items():
        for video in group['publisher_video_ids']:
            join('group:'+group_id, 'video:'+video)
        for family in group['visual_event_families']:
            join('group:'+group_id, 'event:'+family)
    for row in [*old_rows, *new_rows]:
        if not row.get('source_video_id') or not row.get('visual_event_family'):
            raise ValueError('Unknown video/event links cannot be assigned a split')
        join('video:'+row['source_video_id'], 'event:'+row['visual_event_family'])
    for row in old_rows:
        if row.get('context_group') not in old_groups or row.get('development_split') not in {'train','val'}:
            raise ValueError('Malformed old grouped split')
        join('group:'+row['context_group'], 'video:'+row['source_video_id'])
    groups = collections.defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)
    components, assignments = {}, {}
    for members in groups.values():
        identifier = 'context_'+hashlib.sha256(json.dumps(sorted(members), separators=(',',':')).encode()).hexdigest()[:16]
        old = [row for row in old_rows if 'video:'+row['source_video_id'] in members]
        families = sorted(node[6:] for node in members if node.startswith('event:'))
        split = 'val' if any(row['development_split']=='val' for row in old) or any(f in {'MonteCarlo','LondonFinals'} for f in families) else 'train'
        components[identifier] = {'publisher_video_ids': sorted(node[6:] for node in members if node.startswith('video:')),
            'visual_event_families': families, 'prior_context_ids': sorted(node[6:] for node in members if node.startswith('group:')),
            'development_split': split, 'independent_original_matches_verified': False}
        for video in components[identifier]['publisher_video_ids']:
            assignments[video] = (identifier, split)
    if any(assignments[r['source_video_id']][1] != 'val' for r in old_rows if r['development_split']=='val'):
        raise ValueError('A previous development source moved to training')
    return components, assignments

def checked_points(points, proposed, accepted):
    if not accepted:
        return None
    if not isinstance(points, list) or len(points)!=14 or len(proposed)!=14:
        raise ValueError('Require fourteen nested visible-or-null slots')
    for slot, point in enumerate(points):
        if point is None:
            continue
        if (not isinstance(point,list) or len(point)!=2 or point != proposed[slot]
                or any(type(value) not in (int,float) or not math.isfinite(value) for value in point)
                or not 0<=point[0]<1280 or not 0<=point[1]<720):
            raise ValueError('Visible coordinate must preserve a reviewed in-image proposal; no clamping')
    if sum(point is not None for point in points)<8:
        raise ValueError('Insufficient visible geometry for development training')
    return points

def prepare_plan(root=ROOT):
    root = Path(root).resolve()
    documents = {}
    for relative, expected in INPUTS.items():
        if sha(root/relative)!=expected:
            raise ValueError('Changed frozen input: '+relative)
        documents[relative] = json.loads((root/relative).read_text(encoding='utf-8-sig'))
    aids, first, last, old, frozen, export, acquisition = [documents[key] for key in INPUTS]
    reviews = [*first['samples'], *last['samples']]
    if len(reviews)!=64 or [r['number'] for r in reviews]!=list(range(1,65)):
        raise ValueError('Require exact ordered exhaustive64 review')
    if len(old['samples'])!=165 or len(export['sources'])!=120:
        raise ValueError('Prior review/export changed')
    old_videos = {r['source_video_id'] for r in old['samples']}
    if len({r['source_video_id'] for r in reviews})!=64 or old_videos & {r['source_video_id'] for r in reviews}:
        raise ValueError('Wave must contain64 distinct previously uncached video IDs')
    sources = []
    verified = dict(INPUTS)
    for proposal, review in zip(aids['samples'], reviews, strict=True):
        if any(review.get(k)!=proposal.get(k) for k in ('number','id','source_video_id','source_frame_ordinal')):
            raise ValueError('Review/source identity mismatch')
        if review['number']<=32:
            viewed = review['actually_viewed_native_overlay'] and review['actually_viewed_unmarked14point_native_detail_board']
            refs = (review['source_image'], review['native_overlay'], review['unmarked_point_detail_board'])
        else:
            viewed = review['actual_native_overlay_viewed'] and review['actual_all14_unmarked_native_patches_viewed']
            refs = (review['source_image'], review['native_review_overlay'], review['point_detail_board'])
        expected_refs = (proposal['source_image'], proposal['native_review_overlay'], proposal['point_detail_board'])
        if not viewed or refs!=expected_refs or len(review['point_review'])!=14:
            raise ValueError('Missing actual native visual review or exact aid binding')
        for ref in refs:
            path = (root/ref['path']).resolve()
            if not path.is_relative_to(root) or sha(path)!=ref['sha256']:
                raise ValueError('Changed reviewed image/aid bytes')
            verified[ref['path']] = ref['sha256']
        accepted = review['decision'].startswith('ACCEPT')
        points = checked_points(review['reviewed_keypoints_px'], proposal['publisher_keypoints_xy_px'], accepted)
        for slot, point in enumerate(points or [None]*14):
            entry = review['point_review'][slot]
            actual = entry.get('reviewed_xy_px') if review['number']<=32 else entry.get('accepted_xy_px')
            if entry['slot']!=slot or actual!=point:
                raise ValueError('Slot visibility/coordinate serialization mismatch')
        sources.append({'id': proposal['id'], 'number': proposal['number'], 'source_video_id': proposal['source_video_id'],
            'visual_event_family': review['visual_event_family'], 'surface': review['surface'],
            'decision': 'ACCEPT' if accepted else 'REJECT', 'original_review_decision': review['decision'],
            'reviewer': review['reviewer'], 'source_image': proposal['source_image'],
            'source_frame_ordinal': proposal['source_frame_ordinal'], 'source_pts': None,'source_time_base':None,'source_fps':None,
            'match_id':None,'camera_session_id':None,'reviewed_keypoints_px':points,
            'publisher_keypoints_xy_px':proposal['publisher_keypoints_xy_px'],
            'review_note':review.get('review_note', review.get('observations')),
            'independent_human_review':False,'qualification_evidence':False})
    components, assigned = context_components(old['samples'],old['context_groups'],sources)
    for row in sources:
        row['context_group'], row['development_split'] = assigned[row['source_video_id']]
    selected = [row for row in sources if row['decision']=='ACCEPT']
    counts = {'reviewed_images':64,'accepted':len(selected),'rejected':64-len(selected),'test':0,
              'train':sum(row['development_split']=='train' for row in selected),
              'val':sum(row['development_split']=='val' for row in selected),
              'visible_slots':sum(point is not None for row in selected for point in row['reviewed_keypoints_px']),
              'null_slots':sum(point is None for row in selected for point in row['reviewed_keypoints_px'])}
    assert counts==dict(reviewed_images=64,accepted=55,rejected=9,test=0,train=49,val=6,visible_slots=764,null_slots=6)
    migrations = [row['id'] for row in old['samples'] if row['development_split']!=assigned[row['source_video_id']][1]]
    return {'schema_version':'1.0','status':'FROZEN_AGENT_REVIEWED_COURT64_DEVELOPMENT_REPAIR_NOT_EXPORTED',
        'input_bindings':verified,'consolidator_sha256':sha(Path(__file__)), 'counts':counts,'sources':sources,
        'components':components,'old_review_rows_retained':165,'old_prior_context_groups_retained':len(old['context_groups']),
        'old_split_migrations_in_this_wave':migrations,
        'prior_development_into_train':False,'rejected_links_retained':True,
        'historical_export_unchanged':True,'prospective_combined_old120_plus_new55_counts':{'train':149,'val':26,'test':0},
        'independent_human_review':False,'source_independence_verified':False,'qualification_evidence':False,
        'labels_exported':False,'training_performed':False,'final_test_used':False,
        'usage_scope':'PRIVATE_LOCAL_NONCOMMERCIAL_ACADEMIC_DEVELOPMENT_ONLY',
        'raw_public_redistribution_cleared':False,
        'root_second_look':{'actual_native_detail_boards_viewed_numbers':[5,10,16,25,32],
            'decision':'AGREE_REJECT16_32_AND_NULL005_SLOT13_010_SLOT5_025_SLOT2',
            'remaining_first32_review_attribution':'AGENT_NOT_ROOT'},
        'limits':['One still per newly cached publisher video does not verify independent matches or unseen compilation content.',
                  'Visual alignment is an agent development decision with uncalibrated pixel uncertainty.',
                  'Per-role engineering DEV separation is not a cross-model independent qualification split.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Choose a fresh workspace output')
    plan=prepare_plan()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    print(json.dumps({'status':plan['status'],'counts':plan['counts'],'sha256':sha(output),
                      'prior_split_migrations':plan['old_split_migrations_in_this_wave']}))

if __name__=='__main__':
    main()
