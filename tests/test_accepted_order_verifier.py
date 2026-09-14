import numpy as np
import pytest
from scripts.evaluate.reselect_feature_verifier import accepted_order_rows


def rows():
    return [{'clip':'a','frame':1,'width':512,'height':288,'target_xy':[10,20],
             'candidates':[{'rank':0,'sample_index':0,'x':10.,'y':20.},
                           {'rank':1,'sample_index':1,'x':70.,'y':80.}]}]


def test_visual_score_does_not_rerank_accepted_localizations():
    assert accepted_order_rows(rows(),np.array([.8,.99]),.5)[0]['prediction_xy']==[10.,20.]


def test_rejected_first_candidate_can_expose_actual_lower_proposal():
    assert accepted_order_rows(rows(),np.array([.1,.99]),.5)[0]['prediction_xy']==[70.,80.]


def test_all_rejected_or_empty_emits_missing():
    assert accepted_order_rows(rows(),np.array([.1,.2]),.5)[0]['prediction_xy'] is None
    r=rows();r[0]['candidates']=[]
    assert accepted_order_rows(r,np.array([]),.5)[0]['prediction_xy'] is None


def test_incomplete_scores_or_changed_order_fail():
    with pytest.raises(ValueError):accepted_order_rows(rows(),np.array([np.nan,.9]),.5)
    r=rows();r[0]['candidates'].reverse()
    with pytest.raises(ValueError):accepted_order_rows(r,np.array([.8,.9]),.5)
