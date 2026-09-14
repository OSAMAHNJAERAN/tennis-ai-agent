import numpy as np
from scripts.evaluate.probe_court_line_segments import line_evidence as baseline
from scripts.evaluate.probe_court_ridge_ablation import line_evidence


def image(left, right, ridge):
    x=np.zeros((80,120,3),dtype=np.uint8);x[:40]=left;x[40:]=right;x[39:42]=ridge
    return x


def test_baseline_copy_preserves_original_evidence():
    x=image((50,50,50),(60,60,60),(180,180,180));segment=[10,40,110,40]
    assert line_evidence(x,segment,'baseline')==baseline(x,segment)


def test_white_ridge_between_different_surfaces_survives_only_side_relaxation():
    x=image((30,30,30),(100,100,100),(220,220,220));segment=[10,40,110,40]
    assert line_evidence(x,segment,'baseline')['support_fraction']==0
    assert line_evidence(x,segment,'no_white')['support_fraction']==0
    assert line_evidence(x,segment,'no_sides')['support_fraction']==1


def test_colored_bright_ridge_survives_only_white_relaxation():
    x=image((30,50,80),(30,50,80),(70,160,230));segment=[10,40,110,40]
    assert line_evidence(x,segment,'baseline')['support_fraction']==0
    assert line_evidence(x,segment,'no_sides')['support_fraction']==0
    assert line_evidence(x,segment,'no_white')['support_fraction']==1


def test_relaxed_filters_still_reject_a_surface_step_without_a_ridge():
    x=image((30,30,30),(180,180,180),(180,180,180))
    assert line_evidence(x,[10,40,110,40],'ridge_only')['support_fraction']==0
