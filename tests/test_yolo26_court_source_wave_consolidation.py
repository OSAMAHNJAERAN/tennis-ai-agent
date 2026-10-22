"""Data leakage and annotation integrity contracts for the new court wave."""
import copy
import math
import pytest
from scripts.data.consolidate_yolo26_court_source_wave import checked_points, context_components, prepare_plan

def fixture_groups():
    rows=[{'id':'a','source_video_id':'a','visual_event_family':'Rome','context_group':'train','development_split':'train'},
          {'id':'b','source_video_id':'b','visual_event_family':'MonteCarlo','context_group':'dev','development_split':'val'}]
    groups={'train':{'publisher_video_ids':['a'],'visual_event_families':['Rome']},
            'dev':{'publisher_video_ids':['b'],'visual_event_families':['MonteCarlo']}}
    return rows,groups

def test_rejected_compilation_bridge_preserves_development_monotonically():
    old,groups=fixture_groups()
    new=[{'source_video_id':'compilation','visual_event_family':name,'decision':'REJECT'} for name in ('Rome','MonteCarlo')]
    _,assigned=context_components(old,groups,new)
    assert assigned['a']==assigned['b']==assigned['compilation']
    assert assigned['a'][1]=='val'

def test_old_compilation_links_survive_rejected_source_rows_and_aliases():
    old,groups=fixture_groups()
    groups['train']['visual_event_families'].append('Washington')
    _,assigned=context_components(old,groups,[{'source_video_id':'fresh','visual_event_family':'Washington'}])
    assert assigned['a']==assigned['fresh']
    assert assigned['fresh'][1]=='train'

def test_novel_family_is_development_training_not_a_new_test():
    old,groups=fixture_groups()
    components,assigned=context_components(old,groups,[{'source_video_id':'fresh','visual_event_family':'Auckland'}])
    assert assigned['fresh'][1]=='train'
    assert all(value['development_split'] in {'train','val'} for value in components.values())

@pytest.mark.parametrize('key', ['source_video_id','visual_event_family'])
def test_unknown_links_cannot_gain_a_split(key):
    old,groups=fixture_groups()
    row={'source_video_id':'fresh','visual_event_family':'Rome'}
    row[key]=None
    with pytest.raises(ValueError,match='Unknown'):
        context_components(old,groups,[row])

def test_plan_is_deterministic_without_mutating_earlier_grouping():
    old,groups=fixture_groups()
    before=copy.deepcopy((old,groups))
    assert context_components(old,groups,[])==context_components(old,groups,[])
    assert (old,groups)==before

@pytest.mark.parametrize('point', [[1280,1],[-1,0],[1,720],[1,math.nan],[True,1],[2,3,4],(1,2)])
def test_invalid_or_flattened_points_are_not_clamped_or_accepted(point):
    proposed=[[1,2] for _ in range(14)]
    points=copy.deepcopy(proposed)
    points[0]=point
    with pytest.raises(ValueError):
        checked_points(points,proposed,True)

def test_hidden_slot_stays_null_and_proposal_is_unchanged():
    proposed=[[1,2] for _ in range(14)]
    points=copy.deepcopy(proposed)
    points[5]=None
    assert checked_points(points,proposed,True)[5] is None
    assert proposed[5]==[1,2]

def test_insufficient_visible_slots_are_not_completed_automatically():
    with pytest.raises(ValueError,match='Insufficient'):
        checked_points([None]*7+[[1,2]]*7,[[1,2]]*14,True)

def test_actual_complete_wave_preserves_old_development_and_all_denominators():
    plan=prepare_plan()
    assert plan['counts']=={'reviewed_images':64,'accepted':55,'rejected':9,'test':0,'train':49,'val':6,'visible_slots':764,'null_slots':6}
    assert plan['old_split_migrations_in_this_wave']==[]
    assert plan['old_review_rows_retained']==165
    assert plan['rejected_links_retained'] and not plan['labels_exported']
    assert not plan['source_independence_verified'] and not plan['qualification_evidence']
    assert all(row['development_split']=='val' for row in plan['sources'] if row['visual_event_family'] in {'MonteCarlo','LondonFinals'})
