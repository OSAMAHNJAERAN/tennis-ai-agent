from src.tracking.duplicate_player_handoff import reconcile_handoffs


def person(identity, x=0, confidence=.8):
    return {'id': identity, 'box': [x, 0, x+20, 60], 'confidence': confidence}


def test_shared_boundary_merges_and_retains_highest_confidence_actual_box():
    frames = [[person(2)], [person(2), person(150, 2, .9)], [person(150, 3)]]
    evidence = [{'frame': 1, 'source_id': i, 'confidence': c} for i, c in [(2, .7), (150, .8)]]
    rows, supports, mapping, accepted, rejected, suppressed = reconcile_handoffs(frames, 30, evidence, {2: 2, 150: 150})
    assert mapping == {2: 2, 150: 2} and len(accepted) == 1 and not rejected
    assert rows[1] == [{**frames[1][1], 'id': 2, 'source_id': 150}]
    assert len(supports) == 1 and supports[0]['confidence'] == .8
    assert suppressed[0]['suppressed_source_ids'] == [2]


def test_two_people_persisting_together_or_long_overlap_do_not_merge():
    frames = [[person(1), person(2)]]*5
    assert reconcile_handoffs(frames, 30, [], {1: 1, 2: 2})[3] == []
    frames = [[person(1)]]+[[person(1), person(2)]]*4+[[person(2)]]
    assert reconcile_handoffs(frames, 30, [], {1: 1, 2: 2})[3] == []


def test_all_shared_boxes_must_agree_and_observations_cover_whole_overlap():
    frames = [[person(1)], [person(1), person(2)], [person(1), person(2, 30)], [person(2)]]
    assert reconcile_handoffs(frames, 30, [], {1: 1, 2: 2})[3] == []
    frames[2] = [person(1)]
    assert reconcile_handoffs(frames, 30, [], {1: 1, 2: 2})[3] == []


def test_ambiguous_duplicate_successors_are_not_merged():
    frames = [[person(1)], [person(1), person(2), person(3)], [person(2), person(3)]]
    assert reconcile_handoffs(frames, 30, [], {1: 1, 2: 2, 3: 3})[3] == []


def test_existing_flow_chain_is_preserved_and_missing_frames_stay_empty():
    frames = [[person(1)], [], [person(2)], [person(2), person(3, 2)], [person(3, 3)]]
    rows, _, mapping, links, _, _ = reconcile_handoffs(frames, 30, [], {1: 1, 2: 1, 3: 3})
    assert mapping == {1: 1, 2: 1, 3: 1} and len(links) == 1
    assert rows[1] == []


def test_existing_chain_conflict_rejects_locally_plausible_duplicate():
    frames = [[person(1)], [person(1), person(2)], [person(2)],
              [person(2), person(4, 100)], [person(2), person(4, 100)]]
    _, _, mapping, accepted, rejected, _ = reconcile_handoffs(frames, 30, [], {1: 1, 2: 2, 4: 1})
    assert not accepted and rejected[0]['reason'] == 'CHAIN_HAS_DISTINCT_CONCURRENT_BOXES'
    assert mapping == {1: 1, 2: 2, 4: 1}
