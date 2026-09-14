import pytest

from scripts.data.prepare_wasb_spaced_training import real_context, plan_samples


@pytest.mark.parametrize('fps,stride',[(25.,1),(29.97002997,1),(60.,2),(120.,4)])
def test_each_training_slot_is_real_and_target_aligned(fps,stride):
    actual, indices=real_context(40,12,fps)
    assert actual==stride
    for slot in range(3):
        window=indices[2-slot:5-slot]
        assert window[slot]==12
        assert window[1]-window[0]==window[2]-window[1]==stride
        assert len(set(window))==3


@pytest.mark.parametrize('length,frame,fps',[(20,0,60),(20,19,60),(1,0,25),(20,10,0),(20,10,True),(20,10,float('nan'))])
def test_missing_real_context_is_rejected_without_padding(length,frame,fps):
    with pytest.raises(ValueError):
        real_context(length,frame,fps)


def test_plan_preserves_target_and_split_and_rejects_duplicate():
    views=[dict(target=[4.,5.],context_paths=['old']*5) for _ in range(5)]
    sample=dict(clip='match',frame=10,split='train',source_visible=True,views=views)
    clips={'match':dict(frames=30,fps=60.,split='train')}
    planned=plan_samples([sample],clips)[0]
    assert planned['views']==views and planned['split']=='train'
    assert planned['context_indices']==[6,8,10,12,14]
    assert 'stride' not in sample
    with pytest.raises(ValueError,match='Duplicate'):
        plan_samples([sample,sample],clips)
    with pytest.raises(ValueError,match='split'):
        plan_samples([{**sample,'split':'selection'}],clips)
