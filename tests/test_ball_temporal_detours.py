import pytest

from scripts.evaluate.replay_ball_temporal_detours import reject_detours


def test_short_distractor_excursion_rejected_without_coordinate_replacement():
    points = [[i * 5., 80.] for i in range(15)]
    points[6:9] = [[300., 200.], None, [301., 201.]]
    before = [None if p is None else p.copy() for p in points]
    rejected, evidence = reject_detours(points, 30., [512, 288])
    assert rejected == {6, 8}
    assert points == before
    assert any(item['start'] == 6 and item['end_exclusive'] == 9 for item in evidence)


def test_fast_straight_ball_and_sustained_direction_change_are_preserved():
    for points in ([[i * 30., 100.] for i in range(15)],
                   [[i * 5., 20. + abs(i - 7) * 8.] for i in range(15)]):
        assert reject_detours(points, 30., [512, 288])[0] == set()


def test_uncertain_boundaries_missing_anchors_and_static_anchors_preserved():
    assert reject_detours([[300, 200], [10, 10], None], 30, [512, 288])[0] == set()
    points = [[10, 10]] * 12
    points[6] = [300, 200]
    assert reject_detours(points, 30, [512, 288])[0] == set()


def test_reference_geometry_scaling_is_invariant():
    points = [[i * 5., 80.] for i in range(15)]
    points[7] = [300., 200.]
    scaled = [[x * 3, y * 2] for x, y in points]
    assert reject_detours(points, 30, [512, 288])[0] == reject_detours(scaled, 30, [1536, 576])[0] == {7}


@pytest.mark.parametrize('fps,size,points', [(0, [512, 288], []),
                                          (30, [0, 288], []),
                                          (30, [512, 288], [[float('nan'), 2]])])
def test_invalid_inputs_fail(fps, size, points):
    with pytest.raises(ValueError):
        reject_detours(points, fps, size)


def test_low_frame_rate_does_not_exceed_frozen_duration():
    points = [[i * 5., 80.] for i in range(15)]
    points[7] = [300., 200.]
    assert reject_detours(points, 5, [512, 288])[0] == set()


def test_long_detour_is_outside_scope():
    points = [[i * 5., 80.] for i in range(18)]
    points[6:12] = [[300. + i * 5., 200.] for i in range(6)]
    assert reject_detours(points, 30, [512, 288])[0] == set()
