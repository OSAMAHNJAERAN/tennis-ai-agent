import pytest
from scripts.train.finetune_wasb_spaced_head import advance_gate,coverage


def result(p=.9,r=.9,f=.9,coverage_count=100):
    return dict(scores=dict(pooled=dict(precision=p,recall=r,f1=f)),coverage=coverage_count)


@pytest.mark.parametrize('candidate,advance',[
    (result(.93,.93,.93,102),True),
    (result(.94,.94,.94,99),False),
    (result(.89,.99,.94,103),False),
    (result(.99,.89,.94,103),False),
    (result(.92,.92,.92,101),False),
    (result(.92,.92,.92,102),True),
    (result(.93,.93,.93,101),True),
])
def test_advance_requires_original_and_adjacent_control_improvement(candidate,advance):
    assert advance_gate(result(),result(.92,.92,.92,101),candidate) is advance


def test_proposal_coverage_uses_reference_scale_and_visible_labels_only():
    clips=[dict(width=1920,height=1080,rows=[
        dict(target_xy=[100,100],candidates=[dict(x=115,y=100)]),
        dict(target_xy=[100,100],candidates=[dict(x=116,y=100)]),
        dict(target_xy=None,candidates=[dict(x=100,y=100)]),
        dict(target_xy=[100,100],candidates=[]),
    ])]
    assert coverage(clips)==1
