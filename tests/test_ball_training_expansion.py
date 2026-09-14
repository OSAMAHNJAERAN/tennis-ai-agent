import pytest
from scripts.data.audit_ball_training_expansion import validate_labels, validate_split, review_sample


def test_sparse_labels_rescale_visible_only_and_keep_absence_semantics():
    rows=[dict(Frame='3',Visibility='0',X='nan',Y='nan'),dict(Frame='1',Visibility='1',X='960',Y='540')]
    result=validate_labels(rows,5,1280,720)
    assert result==[dict(frame=1,target_xy=[640,360]),dict(frame=3,target_xy=None)]
    assert review_sample(result)==result
    assert len(result)==2  # Unannotated context frames never become targets.


@pytest.mark.parametrize('rows',[
    [dict(Frame='1',Visibility='1',X='nan',Y='0')],
    [dict(Frame='1',Visibility='1',X='1920',Y='0')],
    [dict(Frame='5',Visibility='0',X='0',Y='0')],
    [dict(Frame='1',Visibility='2',X='0',Y='0')],
    [dict(Frame='1',Visibility='0',X='0',Y='0')]*2,
])
def test_bad_labels_fail_before_training(rows):
    with pytest.raises(ValueError):
        validate_labels(rows,5,1280,720)


def test_extending_training_split_preserves_old_roles_and_grouped_rallies():
    clips=[(f'm{i}','0') for i in range(10)]
    first=dict(split='TRAINING_ONLY',revision='r',selected_clips=clips)
    more=[('m4','1'),('m10','0'),('m11','0'),('m12','0'),('m13','0'),('m14','0')]
    second=dict(split='TRAINING_ONLY',revision='r',selected_clips=more)
    old=validate_split([first],set(clips+more),{('v1','0')})
    new=validate_split([first,second],set(clips+more),{('v1','0')})
    assert all(new[k]==v for k,v in old.items())
    assert new['m4_1']==new['m4_0']=='selection'
    assert new['m14_0']=='selection'
    with pytest.raises(ValueError):
        validate_split([first,first],set(clips),set())
    with pytest.raises(ValueError):
        validate_split([first],set(clips),{('m4','1')})


def test_review_sample_is_first_of_each_explicit_class_not_model_selected():
    labels=[dict(frame=2,target_xy=None),dict(frame=3,target_xy=[1,2]),dict(frame=4,target_xy=[3,4])]
    assert [r['frame'] for r in review_sample(labels)]==[3,2]
