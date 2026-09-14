import pytest
from src.tracking.player_fragment_linking import link_player_fragments, remap_observations
from src.tracking.racket_supported_players import select_racket_supported_people


def person(identity, x=0):
    return {'id': identity, 'box': [x, 0, x+20, 60], 'confidence': .8}


def test_actual_fragment_reuses_evidence_and_preserves_missing_frames():
    frames = [[person(4)] for _ in range(15)] + [[]] + [[person(230, 1)] for _ in range(8)]
    support = [{'frame': i, 'source_id': 4, 'confidence': .8} for i in (0, 6)]
    old, _ = select_racket_supported_people(frames, 30, [0, 6, 18], support)
    assert old[-1] == []
    mapping, edges = link_player_fragments(frames, 30)
    changed, evidence = remap_observations(frames, support, mapping)
    selected, _ = select_racket_supported_people(changed, 30, [0, 6, 18], evidence)
    assert mapping == {4: 4, 230: 4} and len(edges) == 1
    assert selected[-1][0]['source_id'] == 230
    assert selected[-1][0]['box'] == frames[-1][0]['box']
    assert selected[15] == [] and frames[-1][0]['id'] == 230


def test_overlapping_or_reused_source_tracks_cannot_be_joined():
    frames = [[person(1)], [person(1), person(2)], [person(2)], [person(1)]]
    assert link_player_fragments(frames, 30)[1] == []


def test_distant_box_long_gap_and_ambiguous_ties_do_not_link():
    assert link_player_fragments([[person(1)], [person(2, 50)]], 30)[1] == []
    assert link_player_fragments([[person(1)]]+[[]]*7+[[person(2)]], 30)[1] == []
    assert link_player_fragments([[person(1)], [person(2), person(3)]], 30)[1] == []
    assert link_player_fragments([[person(1), person(2)], [person(3)]], 30)[1] == []


def test_best_links_are_mutual_chained_and_order_independent():
    frames = [[person(9), person(3, 5)], [person(8, 1)], [person(7, 2)]]
    a = link_player_fragments(frames, 30)
    b = link_player_fragments([list(reversed(r)) for r in frames], 30)
    assert a == b
    assert a[0] == {9: 9, 3: 3, 8: 9, 7: 9}


def test_low_fps_does_not_admit_long_elapsed_boundary():
    assert link_player_fragments([[person(1)], [person(2)]], 2)[1] == []


def test_invalid_inputs_and_overlapping_remap_fail():
    with pytest.raises(ValueError):
        link_player_fragments([[person(1), person(1)]], 30)
    with pytest.raises(ValueError):
        link_player_fragments([[person(1)]], float('nan'))
    with pytest.raises(ValueError):
        remap_observations([[person(1), person(2)]], [], {1: 1, 2: 1})
