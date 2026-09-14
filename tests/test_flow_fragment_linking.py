import cv2
import numpy as np
import pytest

from src.tracking.flow_fragment_linking import measured_translation, link_player_fragments
from src.tracking.player_fragment_linking import remap_observations


def fixture():
    first = np.random.default_rng(7).integers(20, 220, (160, 240), dtype=np.uint8)
    first = cv2.GaussianBlur(first, (3, 3), .5)
    second = cv2.warpAffine(first, np.float32([[1, 0, 12], [0, 1, -3]]), (240, 160))
    return first, second


def test_actual_image_translation_recovered_bidirectionally():
    first, second = fixture()
    a = measured_translation(first, second, [80, 40, 110, 120])
    b = measured_translation(second, first, [92, 37, 122, 117])
    assert a['displacement_px'] == pytest.approx([12, -3], abs=.2)
    assert b['displacement_px'] == pytest.approx([-12, 3], abs=.2)


def test_featureless_and_unrelated_frames_abstain():
    first, _ = fixture()
    assert measured_translation(np.zeros_like(first), first, [80, 40, 110, 120])['displacement_px'] is None
    assert measured_translation(first, np.zeros_like(first), [80, 40, 110, 120])['displacement_px'] is None


def test_link_preserves_real_boxes_and_missing_frames():
    first, second = fixture()
    frames = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (first, first, second, second)]
    detections = [[{'id': 4, 'box': [80, 40, 110, 120], 'confidence': .8}], [], [],
                  [{'id': 230, 'box': [92, 37, 122, 117], 'confidence': .8}]]
    mapping, links, _ = link_player_fragments(detections, 30, frames)
    assert mapping == {4: 4, 230: 4} and len(links) == 1
    changed, _ = remap_observations(detections, [], mapping)
    assert changed[1:3] == [[], []]
    assert changed[3][0]['box'] == detections[3][0]['box']


def test_wrong_successor_box_and_long_elapsed_time_do_not_link():
    first, second = fixture()
    frames = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in (first, second)]
    detections = [[{'id': 1, 'box': [80, 40, 110, 120]}], [{'id': 2, 'box': [160, 37, 190, 117]}]]
    assert link_player_fragments(detections, 30, frames)[1] == []
    detections[1][0]['box'] = [92, 37, 122, 117]
    assert link_player_fragments(detections, 2, frames)[1] == []


def test_resolution_and_geometry_mismatch_fail():
    first, second = fixture()
    with pytest.raises(ValueError):
        measured_translation(first, second[:80], [0, 0, 20, 40])
    with pytest.raises(ValueError):
        measured_translation(first, second, [0, 0, 0, 40])
