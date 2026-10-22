"""Candidate spool preserves frame contracts, lazy production and owned files."""

import json
import math
import sqlite3
from contextlib import closing

import pytest

from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.bbox_utils import BBox
from src.utils import ball_candidate_sequence as module
from src.utils.ball_candidate_sequence import BallCandidateSequence


def observation(index, confidence=.9):
    return BallObservation(index + .125, index + .375, confidence)


def stored_rows(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT count(*) FROM candidate_frames").fetchone()[0]


def test_large_producer_is_consumed_one_frame_at_a_time_and_commits_in_bounded_batches(tmp_path):
    with BallCandidateSequence(tmp_path) as sequence:
        closed = []

        def frames():
            try:
                for index in range(25000):
                    # Materializing the producer before inserting records would
                    # fail on its second yield, even for this large frame count.
                    assert len(sequence) == index
                    yield [] if index % 5 == 0 else [observation(index)]
            finally:
                closed.append(True)

        sequence.extend(frames())
        assert closed == [True]
        assert len(sequence) == 25000
        assert stored_rows(sequence.path) == 25000 // 256 * 256
        assert sequence[0] == []
        assert sequence[24999] == [observation(24999)]
        assert sum(bool(frame) for frame in sequence) == 20000
        owned = sequence.path
    assert not owned.exists()


def test_float_bbox_null_nonfinite_and_candidate_order_round_trip(tmp_path):
    smallest = float.fromhex("0x0.0000000000001p-1022")
    precise = float.fromhex("0x1.0000000000001p+0")
    box = BBox(-0., smallest, precise, float("inf"), confidence=float("nan"), class_id=32, track_id=None)
    candidates = [BallObservation(-0., precise, smallest, box),
                  BallObservation(float("nan"), float("-inf"), None, None)]
    with BallCandidateSequence(tmp_path) as sequence:
        sequence.append(candidates)
        sequence.append([])
        restored = sequence[0]
        assert math.copysign(1., restored[0].x_px) == -1
        assert restored[0].y_px.hex() == precise.hex()
        assert restored[0].confidence.hex() == smallest.hex()
        assert restored[0].bbox is not box
        assert restored[0].bbox.y1.hex() == smallest.hex()
        assert restored[0].bbox.x2.hex() == precise.hex()
        assert math.isinf(restored[0].bbox.y2)
        assert math.isnan(restored[0].bbox.confidence)
        assert restored[0].bbox.class_id == 32
        assert restored[0].bbox.track_id is None
        assert math.isnan(restored[1].x_px)
        assert restored[1].y_px == float("-inf")
        assert restored[1].confidence is None
        assert restored[1].bbox is None
        assert sequence[1] == []


def test_neighbors_backward_slices_and_explicit_updates_preserve_frame_order(tmp_path):
    with BallCandidateSequence(tmp_path) as sequence:
        sequence.extend([[observation(index)] for index in range(9)])
        assert [sequence[index][0].x_px for index in (4, 2, 6, 5, 3, 8, 0)] == [i + .125 for i in (4, 2, 6, 5, 3, 8, 0)]
        assert sequence[-1] == [observation(8)]
        assert sequence[-9] == [observation(0)]
        assert sequence[2:7:2] == [[observation(i)] for i in (2, 4, 6)]
        assert sequence[::-1] == [[observation(i)] for i in range(8, -1, -1)]
        assert sequence[100:200] == []
        snapshot = sequence[3]
        snapshot[0].x_px = 999
        snapshot.append(observation(99))
        assert sequence[3] == [observation(3)]
        sequence[-6] = snapshot
        assert sequence[3] == snapshot
        assert sequence[2] == [observation(2)]
        assert sequence[4] == [observation(4)]
        for index, values in enumerate(sequence):
            sequence[index] = [] if index % 2 == 0 else values
        assert len(sequence) == 9
        assert sequence[::2] == [[]] * 5
        assert list(reversed(sequence)) == sequence[::-1]
        for index in (-10, 9):
            with pytest.raises(IndexError):
                sequence[index]
            with pytest.raises(IndexError):
                sequence[index] = []
        with pytest.raises(TypeError):
            sequence[1.2]
        with pytest.raises(TypeError):
            sequence[1:3] = []


def test_page_cache_journal_and_temp_storage_are_bounded(tmp_path):
    with BallCandidateSequence(tmp_path, transaction_size=2) as sequence:
        connection = sequence._connection
        assert connection.execute("PRAGMA cache_size").fetchone()[0] == -2048
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        sequence.append([])
        assert stored_rows(sequence.path) == 0
        sequence.append([observation(1)])
        assert stored_rows(sequence.path) == 2
        sequence[1] = []
        with closing(sqlite3.connect(sequence.path)) as other:
            before = json.loads(other.execute("SELECT payload FROM candidate_frames WHERE frame_index=1").fetchone()[0])
        assert before[0]["x_px"] == 1.125
        sequence[0] = [observation(0)]
        with closing(sqlite3.connect(sequence.path)) as other:
            assert json.loads(other.execute("SELECT payload FROM candidate_frames WHERE frame_index=1").fetchone()[0]) == []


def test_success_removes_only_owned_database_and_known_sidecars(tmp_path):
    unrelated = tmp_path / "ball_candidates_other.sqlite3"
    unrelated.write_bytes(b"unrelated")
    with BallCandidateSequence(tmp_path) as sequence:
        sequence.append([observation(0)])
        owned = sequence.path
        assert owned.parent == tmp_path.resolve()
        assert owned.exists()
        # Names belong exclusively to this owned DB; journal DELETE never uses
        # these two, but lifecycle cleanup also covers interrupted sidecars.
        for suffix in ("-wal", "-shm"):
            (tmp_path / (owned.name + suffix)).write_bytes(b"owned")
    assert unrelated.read_bytes() == b"unrelated"
    assert list(tmp_path.iterdir()) == [unrelated]
    with pytest.raises(RuntimeError, match="not open"):
        sequence[0]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        sequence.__enter__()


def test_producer_failure_retains_closed_partial_database_and_original_error(tmp_path):
    stopped = []
    sequence = BallCandidateSequence(tmp_path)

    def frames():
        try:
            yield [observation(0)]
            yield []
            raise LookupError("Producer failed")
        finally:
            stopped.append(True)

    with pytest.raises(LookupError, match="Producer failed"):
        with sequence:
            sequence.extend(frames())
    assert stopped == [True]
    assert sequence._connection is None
    assert sequence.path.exists()
    assert stored_rows(sequence.path) == 2
    # Renaming on Windows also checks that no owned DB handle remained open.
    moved = sequence.path.with_suffix(".retained")
    sequence.path.rename(moved)
    assert moved.exists()


@pytest.mark.parametrize("operation", ["append", "replace"])
def test_serialization_failure_leaves_accepted_record_unchanged_and_retains_database(tmp_path, operation):
    sequence = BallCandidateSequence(tmp_path)
    with pytest.raises(TypeError):
        with sequence:
            sequence.append([observation(0)])
            broken = BallObservation(object(), 1., .9)
            if operation == "append":
                sequence.append([broken])
            else:
                sequence[0] = [broken]
    assert sequence._connection is None
    assert stored_rows(sequence.path) == 1
    with closing(sqlite3.connect(sequence.path)) as connection:
        value = json.loads(connection.execute("SELECT payload FROM candidate_frames").fetchone()[0])
    assert value[0]["x_px"] == .125


def test_sqlite_mutation_failure_retains_closed_database_and_original_error(tmp_path):
    sequence = BallCandidateSequence(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="blocked insert"):
        with sequence:
            sequence.append([])
            sequence._connection.execute(
                "CREATE TRIGGER reject_insert BEFORE INSERT ON candidate_frames "
                "BEGIN SELECT RAISE(ABORT,'blocked insert'); END"
            )
            sequence.append([observation(1)])
    assert sequence._connection is None
    assert stored_rows(sequence.path) == 1


def test_cleanup_failure_does_not_mask_original_exception(tmp_path, monkeypatch):
    real_connect = module.sqlite3.connect

    class FailingCommit:
        def __init__(self, path):
            self.connection = real_connect(path)
            self.fail = False
            self.closed = False

        def execute(self, *args):
            return self.connection.execute(*args)

        def commit(self):
            if self.fail:
                raise sqlite3.OperationalError("commit unavailable")
            return self.connection.commit()

        def close(self):
            self.connection.close()
            self.closed = True

    connections = []

    def connect(path):
        value = FailingCommit(path)
        connections.append(value)
        return value

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    sequence = BallCandidateSequence(tmp_path)
    with pytest.raises(LookupError, match="original producer failure") as caught:
        with sequence:
            sequence.append([])
            connections[0].fail = True
            raise LookupError("original producer failure")
    assert connections[0].closed
    assert sequence.path.exists()
    assert any("commit unavailable" in note for note in caught.value.__notes__)


def test_sqlite_setup_failure_retains_owned_file_and_preserves_error(tmp_path, monkeypatch):
    def fail(_path):
        raise sqlite3.OperationalError("cannot connect")

    monkeypatch.setattr(module.sqlite3, "connect", fail)
    sequence = BallCandidateSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match="cannot connect"):
        with sequence:
            pytest.fail("Setup failure must not enter producer")
    assert sequence._connection is None
    assert sequence.path.parent == tmp_path.resolve()
    assert sequence.path.exists()


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "256"])
def test_invalid_transaction_size_creates_no_files(tmp_path, value):
    with pytest.raises(ValueError, match="positive integer"):
        BallCandidateSequence(tmp_path, transaction_size=value)
    assert list(tmp_path.iterdir()) == []


def test_missing_output_directory_is_not_created(tmp_path):
    missing = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError):
        with BallCandidateSequence(missing):
            pass
    assert not missing.exists()
