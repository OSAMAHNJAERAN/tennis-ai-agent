from copy import deepcopy
import pytest
from src.tracking.racket_supported_players import select_racket_supported_people


def people():
    return [[{'id':i,'box':[i*10.,0.,i*10.+5,20.],'confidence':.8} for i in (1,2,3)] for _ in range(20)]


def test_repeated_evidence_selects_people_and_excludes_unsupported_spectator():
    rows=people();before=deepcopy(rows)
    evidence=[{'frame':f,'source_id':i,'confidence':.7} for f in (0,6) for i in (1,2)]
    selected,stats=select_racket_supported_people(rows,30.,[0,6,12,18],evidence)
    assert all([r['id'] for r in frame]==[1,2] for frame in selected)
    assert not stats[3]['eligible'] and rows==before


def test_single_support_short_tracks_and_missing_boxes_stay_missing():
    rows=people();rows[10]=[]
    evidence=[{'frame':f,'source_id':1,'confidence':.7} for f in (0,6)]
    selected,_=select_racket_supported_people(rows,30.,[0,6,12,18],evidence)
    assert selected[10]==[]
    assert all(not frame or [r['id'] for r in frame]==[1] for frame in selected)
    assert not any(select_racket_supported_people(people(),30.,[0,6],evidence[:1])[0])
    assert not any(select_racket_supported_people(people()[:8],30.,[0,6],evidence)[0])


def test_maximum_two_observed_tracks_and_stable_tiebreak():
    evidence=[{'frame':f,'source_id':i,'confidence':.7} for f in (0,6) for i in (1,2,3)]
    selected,_=select_racket_supported_people(people(),30.,[0,6],evidence)
    assert all([r['id'] for r in frame]==[1,2] for frame in selected)


@pytest.mark.parametrize('bad',[{'frame':0,'source_id':99,'confidence':.7},
                              {'frame':1,'source_id':1,'confidence':.7},
                              {'frame':0,'source_id':1,'confidence':float('nan')}])
def test_invalid_support_fails(bad):
    with pytest.raises(ValueError):select_racket_supported_people(people(),30.,[0,6],[bad])


def test_duplicate_support_and_nonpositive_fps_fail():
    evidence={'frame':0,'source_id':1,'confidence':.7}
    with pytest.raises(ValueError):select_racket_supported_people(people(),30.,[0,6],[evidence,evidence])
    with pytest.raises(ValueError):select_racket_supported_people(people(),0.,[0,6],[])
