import json
from pathlib import Path
import numpy as np
from scripts.evaluate.refine_court_boundary_geometry import refine


def test_real_missing_baseline_case_preserves_previously_correct_landmarks():
    fixture=json.loads((Path(__file__).parent/'fixtures/court_boundary_regression.json').read_text())
    actual,evidence=refine(fixture['points_reference'],fixture['segments'])
    distances=np.linalg.norm(actual/.75-np.array(fixture['labels_native']),axis=1)
    assert np.all(distances[fixture['original_correct_ids']]<=7), distances.tolist()


import cv2
import pytest
from src.court.court_geometry import TennisCourtGeometry
from scripts.evaluate.refine_court_boundary_geometry import boundary_nullspace, project, LINES


def court():
    canonical=TennisCourtGeometry.get_canonical_keypoints().astype(float)
    transform=cv2.getPerspectiveTransform(canonical[:4].astype(np.float32),
        np.array([[280,130],[680,130],[110,480],[850,480]],np.float32))
    return cv2.perspectiveTransform(canonical[None],transform)[0]


@pytest.mark.parametrize('locked', [{0},{1},{2},{3},{0,2},{1,2},{0,1,2,3}])
def test_projective_updates_preserve_unobserved_boundary_lines(locked):
    points=court();basis=boundary_nullspace(points,locked)
    rng=np.random.default_rng(17)
    for _ in range(10):
        parameters=np.eye(3).ravel()[:8]+basis@rng.normal(0,.005,basis.shape[1])
        moved=project(points/960,np.append(parameters,1).reshape(3,3))*960
        for identity in locked:
            a,b=LINES[identity];delta=points[b]-points[a]
            normal=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
            assert np.abs((moved[[a,b]]-points[a])@normal).max()<1e-8


@pytest.mark.parametrize('shift', [(7,5),(-6,4),(3,-7)])
def test_fully_observed_court_still_recovers_displacement(shift):
    truth=court();actual,evidence=refine(truth+shift,[truth[[a,b]] for a,b in LINES])
    assert evidence['accepted']
    assert np.linalg.norm(actual-truth,axis=1).max()<.1


@pytest.mark.parametrize('missing', [0,1,2,3])
def test_partial_evidence_can_improve_without_moving_missing_boundary(missing):
    initial=court();basis=boundary_nullspace(initial,{missing})
    parameters=np.eye(3).ravel()[:8]+basis@np.random.default_rng(10).normal(0,.01,basis.shape[1])
    truth=project(initial/960,np.append(parameters,1).reshape(3,3))*960
    actual,evidence=refine(initial,[truth[[a,b]] for i,(a,b) in enumerate(LINES) if i!=missing])
    assert evidence['accepted']
    assert np.linalg.norm(actual-truth,axis=1).max()<.1
    assert missing in evidence['iterations'][-1]['locked_boundary_lines']
