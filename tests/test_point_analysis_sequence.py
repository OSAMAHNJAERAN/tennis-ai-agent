"""Pure SQLite point-owner contracts; synthetic rows are not match evidence."""
from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.utils.point_analysis_sequence import PointAnalysisSequence


EXPECTED = [
    dict(shot_position=0, shot_id=71, event_id=101, point_uid="p0", attempt_uid="a0",
         rally_uid="r0", play_scope="LEGAL_RALLY", participant_id=1),
    dict(shot_position=1, shot_id=88, event_id=102, point_uid="p0", attempt_uid="a1",
         rally_uid=None, play_scope="FIRST_SERVE_FAULT", participant_id=1),
    dict(shot_position=2, shot_id=71, event_id=103, point_uid=None, attempt_uid=None,
         rally_uid=None, play_scope="UNKNOWN", participant_id=None),
    dict(shot_position=3, shot_id=900, event_id=104, point_uid="p1", attempt_uid="a2",
         rally_uid="r1", play_scope="LEGAL_RALLY", participant_id=2),
    dict(shot_position=4, shot_id=0, event_id=105, point_uid="p1", attempt_uid="a2",
         rally_uid="r1", play_scope="LEGAL_RALLY", participant_id=2),
]


def add_members(owner):
    # Deliberately insert out of order. Position, not ID or insertion order,
    # defines sequence order; duplicate and zero shot IDs remain real values.
    for index in (4, 1, 3, 0, 2):
        row = EXPECTED[index]
        shot = SimpleNamespace(shot_id=row["shot_id"], match_event_id=row["event_id"],
            landing_court_position_m=(3., 6.) if index in (3, 4) else None)
        transition = {key: row[key] for key in ("point_uid", "attempt_uid", "rally_uid")}
        transition["physical_event_type"] = "SERVE_CONTACT" if index == 0 else "PLAYER_2_HIT"
        owner.add_member(index, shot, transition, row["play_scope"], row["participant_id"])


@pytest.fixture
def sealed_owner(tmp_path):
    with PointAnalysisSequence(tmp_path, "synthetic-owner", source_fingerprint="synthetic-no-media") as owner:
        add_members(owner)
        owner.seal()
        yield owner


def test_module_import_needs_no_model_or_image_runtime(monkeypatch):
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "ultralytics", "numpy", "cv2", "PIL", "pandas"}:
            raise AssertionError("Pure point owner imported a heavy runtime: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    path = Path(__file__).resolve().parents[1] / "src/utils/point_analysis_sequence.py"
    spec = importlib.util.spec_from_file_location("independent_point_owner_pure_import", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PointAnalysisSequence.__name__ == "PointAnalysisSequence"


def test_membership_iteration_preserves_all_fields_and_position_order(sealed_owner):
    view = sealed_owner.member_rows()
    assert len(view) == 5 and list(view) == EXPECTED
    assert all(set(row) == set(EXPECTED[0]) for row in view)
    assert sealed_owner.readers == set()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4, -1, -2, -5])
def test_membership_integer_and_negative_index_return_the_same_complete_record(sealed_owner, index):
    view = sealed_owner.member_rows()
    assert view[index] == EXPECTED[index]


@pytest.mark.parametrize("selection", [slice(None), slice(1, 4), slice(None, None, 2),
    slice(None, None, -1), slice(4, 0, -2), slice(-4, -1), slice(2, 2)])
def test_membership_slices_keep_full_records_and_python_selection_semantics(sealed_owner, selection):
    selected = sealed_owner.member_rows()[selection]
    assert list(selected) == EXPECTED[selection]
    if len(selected):
        assert selected[0] == EXPECTED[selection][0]
        assert selected[-1] == EXPECTED[selection][-1]


def test_reversed_membership_sequence_keeps_dicts_and_all_null_fields(sealed_owner):
    assert list(reversed(sealed_owner.member_rows())) == EXPECTED[::-1]
    assert sealed_owner.member_rows()[::-1][2] == EXPECTED[2]


@pytest.mark.parametrize("index", [5, 100, -6])
def test_membership_out_of_range_index_raises(sealed_owner, index):
    with pytest.raises(IndexError):
        sealed_owner.member_rows()[index]


def test_empty_membership_views_and_empty_slices(tmp_path):
    with PointAnalysisSequence(tmp_path, "empty") as owner:
        owner.seal()
        view = owner.member_rows()
        assert list(view) == [] and list(view[::-1]) == [] and list(reversed(view)) == []
        for index in (0, -1):
            with pytest.raises(IndexError):
                view[index]


def test_scalar_id_and_eligible_position_views_remain_scalars(sealed_owner):
    ids = sealed_owner.member_ids("point_uid", "p1")
    assert list(ids) == [900, 0] and ids[0] == 900 and ids[-1] == 0
    assert list(ids[::-1]) == [0, 900]
    eligible = sealed_owner.eligible_positions()
    assert list(eligible) == [0, 3, 4] and eligible[0] == 0 and eligible[-1] == 4
    assert list(sealed_owner.eligible_positions(landings=True)) == [3, 4]


def test_indexed_and_iterated_membership_dicts_are_detached_snapshots(sealed_owner):
    first = sealed_owner.member_rows()[0]
    first["shot_id"] = 99999
    iterated = next(iter(sealed_owner.member_rows()))
    iterated["participant_id"] = 42
    assert sealed_owner.member_rows()[0] == EXPECTED[0]
    assert list(sealed_owner.member_rows()) == EXPECTED


def test_seal_rejects_record_and_membership_writes_without_changing_views(sealed_owner):
    before = list(sealed_owner.member_rows())
    assert sealed_owner.pending == 0
    with pytest.raises(ValueError, match="sealed"):
        sealed_owner.put("points", "p-new", {"uid": "p-new"})
    with pytest.raises(ValueError, match="sealed"):
        sealed_owner.add_member(99, SimpleNamespace(shot_id=1), {}, "UNKNOWN", None)
    assert list(sealed_owner.member_rows()) == before


@pytest.mark.parametrize("selector", ["membership", "record", "scalar"])
def test_suspended_readers_close_with_owner_and_cleanup_preserves_neighbor(tmp_path, selector):
    neighbor = tmp_path / "point_analysis_neighbor.sqlite3"
    neighbor.write_bytes(b"preexisting unrelated file")
    with PointAnalysisSequence(tmp_path, "cleanup") as owner:
        add_members(owner)
        owner.put("transitions", "event", {"post_state": {"score": "observed"}}, event_id=1, frame=2)
        owner.seal()
        path = owner.path
        view = (owner.member_rows() if selector == "membership" else owner.rows("transitions")
                if selector == "record" else owner.member_ids("point_uid", "p1"))
        iterator = iter(view)
        next(iterator)
        assert len(owner.readers) == 1
    assert owner.readers == set()
    assert owner.connection is None
    for suffix in ("", "-journal", "-wal", "-shm"):
        assert not Path(str(path) + suffix).exists()
    assert neighbor.read_bytes() == b"preexisting unrelated file"
    iterator.close()  # Closing an already owner-disposed iterator stays safe.
    for access in (lambda: len(view), lambda: view[0], lambda: next(iter(view))):
        with pytest.raises(RuntimeError, match="not open"):
            access()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_primary_exception_survives_suspended_reader_cleanup(tmp_path, error_type):
    primary = error_type("explicit synthetic cancellation")
    with pytest.raises(error_type) as caught:
        with PointAnalysisSequence(tmp_path, "primary-failure") as owner:
            add_members(owner)
            iterator = iter(owner.member_rows())
            next(iterator)
            path = owner.path
            raise primary
    assert caught.value is primary
    assert owner.readers == set() and owner.connection is None
    assert not path.exists()
    assert not list(tmp_path.glob(path.name + "*"))
    iterator.close()


def test_frame_state_is_a_source_frame_prefix_including_stable_same_frame_ties(tmp_path):
    initial = {"score": "initial"}
    with PointAnalysisSequence(tmp_path, "causal-frames") as owner:
        # Future state is inserted first, preventing a last-inserted shortcut.
        for uid, ident, frame, score in (("future", 12, 12, "at12"), ("early", 2, 2, "at2"),
                                       ("tie-first", 8, 8, "at8-first"), ("tie-last", 9, 8, "at8-last")):
            owner.put("transitions", uid, {"post_state": {"score": score}}, event_id=ident, frame=frame)
        owner.seal()
        assert [owner.frame_state(frame, initial) for frame in (0, 1, 2, 7, 8, 11, 12)] == [
            initial, initial, {"score": "at2"}, {"score": "at2"},
            {"score": "at8-last"}, {"score": "at8-last"}, {"score": "at12"}]
        returned = owner.frame_state(8, initial)
        returned["score"] = "mutated by caller"
        assert owner.frame_state(8, initial) == {"score": "at8-last"}


def test_frame_state_before_any_transition_keeps_the_supplied_initial_state(tmp_path):
    initial = {"score_confirmed": False, "point_state": "WAITING_FOR_SERVE"}
    with PointAnalysisSequence(tmp_path, "empty-frames") as owner:
        owner.seal()
        assert owner.frame_state(0, initial) == initial
        assert owner.frame_state(1000, initial) == initial
