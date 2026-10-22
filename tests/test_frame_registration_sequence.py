"""Exact, bounded, aligned court registration storage and owned lifecycle."""

from contextlib import closing
from dataclasses import FrozenInstanceError
import math
import sqlite3
import weakref

import numpy as np
import pytest

from src.court.calibration import CourtCalibration
from src.utils import frame_registration_sequence as module
from src.utils.frame_registration_sequence import FrameRegistration, FrameRegistrationSequence


def calibration(index=0, dtype=np.float64):
    matrix = np.array([[1.0000000000000002, 0., index + .12345678901234566],
                       [-0., .5000000000000001, -.125], [0., 0., 1.]], dtype=dtype)
    return CourtCalibration(matrix, (True, False, True), -.0, .12345678901234566,
                            float.fromhex("0x1.0000000000001p-1"), None)


def registration(index=0, dtype=np.float64):
    return FrameRegistration(calibration(index, dtype), np.array(
        [[1., 0., index + .375], [0., 1., -0.], [0., 0., 1.]], dtype=dtype))


def row_count(path):
    with closing(sqlite3.connect(path)) as connection:
        return connection.execute("SELECT COUNT(*) FROM registration_frames").fetchone()[0]


def assert_array_exact(actual, expected):
    assert actual is not expected
    assert actual.dtype == expected.dtype
    assert actual.dtype.str == expected.dtype.str
    assert actual.shape == expected.shape
    assert actual.tobytes(order="C") == expected.tobytes(order="C")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, ">f4", ">f8", "<f4", "<f8"])
def test_round_trip_preserves_fields_matrix_bytes_types_and_exact_projection(tmp_path, dtype):
    original = registration(4, dtype)
    with FrameRegistrationSequence(tmp_path) as sequence:
        sequence.append(original)
        decoded = sequence[0]
        assert decoded is not original and decoded.calibration is not original.calibration
        assert_array_exact(decoded.calibration.image_to_court, original.calibration.image_to_court)
        assert_array_exact(decoded.anchor_to_frame_px, original.anchor_to_frame_px)
        first, second = original.calibration, decoded.calibration
        assert first.inlier_mask == second.inlier_mask == (True, False, True)
        for field in ("reprojection_error_px", "reprojection_error_m", "inlier_reprojection_error_px"):
            assert getattr(first, field).hex() == getattr(second, field).hex()
        assert second.rejection_reason is None and second.is_valid and second.inlier_count == 2
        for point in ((0., 0.), (-.0, 50.12345678901234), (1000.125, 800.375)):
            assert [value.hex() for value in first.project_ground_point(point)] == [
                value.hex() for value in second.project_ground_point(point)]
        with pytest.raises(FrozenInstanceError):
            decoded.anchor_to_frame_px = None
        with pytest.raises(FrozenInstanceError):
            decoded.calibration.rejection_reason = "changed"


def test_arbitrary_numeric_shape_layout_and_nonfinite_bit_patterns_are_not_widened(tmp_path):
    # Distinct NaN payload/sign bits and subnormal values prove a raw-buffer
    # round trip, independently of float equality or public JSON sanitization.
    bits = np.array([0x80000000, 0x00000001, 0x7FC12345, 0xFFC54321, 0x7F800000, 0xFF800000], dtype=np.uint32)
    matrix = bits.view(np.float32).reshape(2, 3)
    transform = np.asfortranarray(np.arange(12, dtype=">f8").reshape(3, 4))[:, ::2]
    record = FrameRegistration(CourtCalibration(matrix, rejection_reason="REJECTED"), transform)
    with FrameRegistrationSequence(tmp_path) as sequence:
        sequence.append(record)
        sequence.append(FrameRegistration(CourtCalibration(np.empty((0, 3), dtype=np.float32)), np.array(.5, dtype=">f4")))
        restored = sequence[0]
        assert_array_exact(restored.calibration.image_to_court, matrix)
        assert_array_exact(restored.anchor_to_frame_px, transform)
        assert not restored.calibration.is_valid
        assert restored.calibration.project_ground_point((1., 2.)) is None
        assert sequence[1].calibration.image_to_court.shape == (0, 3)
        assert sequence[1].anchor_to_frame_px.shape == ()
        assert sequence[1].anchor_to_frame_px.dtype.str == ">f4"


def test_camera_loss_confirmation_and_return_preserve_aligned_nulls_and_reasons(tmp_path):
    valid = registration(0)
    lost = FrameRegistration(CourtCalibration(None, (), None, None, None, "CAMERA_CUT"), None)
    pending = FrameRegistration(CourtCalibration(None, (False, True), .25, .75, .125,
                                                "RETURNING_VIEW_CONFIRMATION_PENDING"), None)
    returned = registration(3)
    # Rejected calibration with a raw matrix is intentionally retained. The
    # public to_dict would hide it; storage must not change original fields.
    raw_rejected = FrameRegistration(CourtCalibration(np.eye(3, dtype=np.float32), (True,),
                                                      float("nan"), float("inf"), -float("inf"), "WEAK_SUPPORT"), None)
    with FrameRegistrationSequence(tmp_path) as sequence:
        sequence.extend(iter([valid, lost, pending, returned, raw_rejected]))
        assert [item.is_valid for item in sequence.calibrations] == [True, False, False, True, False]
        assert [item.rejection_reason for item in sequence.calibrations] == [None, "CAMERA_CUT", "RETURNING_VIEW_CONFIRMATION_PENDING", None, "WEAK_SUPPORT"]
        assert [value is None for value in sequence.transforms] == [False, True, True, False, True]
        for index in (1, 2, 4):
            assert sequence.calibrations[index].project_ground_point((12., 14.)) is None
        fifth = sequence[4].calibration
        assert fifth.image_to_court.dtype == np.float32
        assert math.isnan(fifth.reprojection_error_px)
        assert fifth.reprojection_error_m == float("inf")
        assert fifth.inlier_reprojection_error_px == -float("inf")


def test_neighbors_reverse_negative_slices_views_and_independent_snapshots(tmp_path):
    with FrameRegistrationSequence(tmp_path) as sequence:
        sequence.extend(registration(index) for index in range(9))
        expected = [8, 0, 4, 2, 6, 5, 3]
        assert [sequence[index].anchor_to_frame_px[0, 2] for index in expected] == [index + .375 for index in expected]
        assert sequence[-1].anchor_to_frame_px[0, 2] == 8.375
        assert sequence[-9].anchor_to_frame_px[0, 2] == .375
        assert [item.anchor_to_frame_px[0, 2] for item in sequence[2:7:2]] == [2.375, 4.375, 6.375]
        assert [item[0, 2] for item in sequence.transforms[::-1]] == [i + .375 for i in range(8, -1, -1)]
        assert [item.image_to_court[0, 2] for item in sequence.calibrations[::3]] == [i + .12345678901234566 for i in (0, 3, 6)]
        assert sequence[100:200] == []
        assert len(sequence) == len(sequence.calibrations) == len(sequence.transforms) == 9
        read = sequence[3]
        read.calibration.image_to_court[0, 2] = 999
        read.anchor_to_frame_px[0, 2] = 1000
        assert sequence[3].anchor_to_frame_px[0, 2] == 3.375
        assert sequence.calibrations[3].image_to_court[0, 2] == 3.1234567890123457
        for index in (-10, 9):
            for view in (sequence, sequence.calibrations, sequence.transforms):
                with pytest.raises(IndexError):
                    view[index]
        with pytest.raises(TypeError):
            sequence[1.5]
        with pytest.raises(TypeError):
            sequence[0] = registration(10)


def test_large_lazy_producer_bounded_record_lifetimes_and_commit_batches(tmp_path):
    counters = {"live": 0, "peak": 0, "closed": False}

    def released():
        counters["live"] -= 1

    with FrameRegistrationSequence(tmp_path) as sequence:
        def records():
            try:
                for index in range(25000):
                    assert len(sequence) == index, "Records were materialized before append"
                    record = registration(index)
                    counters["live"] += 1
                    counters["peak"] = max(counters["peak"], counters["live"])
                    weakref.finalize(record, released)
                    yield record
            finally:
                counters["closed"] = True

        sequence.extend(records())
        assert counters["closed"] and counters["live"] == 0 and counters["peak"] < 4
        assert row_count(sequence.path) == 25000 // 256 * 256
        for index in (24999, 15, 1, 16000):
            assert sequence[index].anchor_to_frame_px[0, 2] == index + .375
        decoded = {"live": 0, "peak": 0}

        def decoded_released():
            decoded["live"] -= 1

        for item in sequence:
            decoded["live"] += 1
            decoded["peak"] = max(decoded["peak"], decoded["live"])
            weakref.finalize(item, decoded_released)
        del item
        assert decoded["live"] == 0 and decoded["peak"] < 4


def test_cache_journal_and_transactions_are_bounded(tmp_path):
    with FrameRegistrationSequence(tmp_path, transaction_size=2) as sequence:
        connection = sequence._connection
        assert connection.execute("PRAGMA cache_size").fetchone()[0] == -2048
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 1
        assert connection.execute("PRAGMA mmap_size").fetchone()[0] == 0
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        sequence.append(registration())
        assert row_count(sequence.path) == 0
        sequence.append(registration(1))
        assert row_count(sequence.path) == 2


def test_constant_static_mode_has_no_database_no_frame_list_and_original_calibration(monkeypatch):
    original = calibration()
    sequence = FrameRegistrationSequence.constant(original, 432000)
    real_eye, calls = module.np.eye, []

    def eye(*args, **kwargs):
        calls.append(True)
        return real_eye(*args, **kwargs)

    monkeypatch.setattr(module.np, "eye", eye)
    assert sequence.path is None and sequence.directory is None and sequence._connection is None
    assert len(sequence) == len(sequence.calibrations) == len(sequence.transforms) == 432000
    assert sequence.calibrations[0] is original and sequence.calibrations[-1] is original
    assert calls == [], "Reading calibration should not create an unused identity transform"
    first, last = sequence[0], sequence[-1]
    assert first.calibration is last.calibration is original
    assert len(calls) == 2
    assert first.anchor_to_frame_px.dtype == np.float64
    first.anchor_to_frame_px[0, 0] = 99
    assert sequence.transforms[0][0, 0] == last.anchor_to_frame_px[0, 0] == 1
    assert [item.calibration is original for item in sequence[1:4]] == [True] * 3
    with sequence:
        assert sequence.calibrations[20] is original
    assert sequence.calibrations[20] is original
    with pytest.raises(TypeError, match="read-only"):
        sequence.append(registration())


def test_invalid_static_calibration_and_empty_static_frames_preserve_nulls():
    original = CourtCalibration(None, rejection_reason="INVALID_ANCHOR")
    sequence = FrameRegistrationSequence.constant(original, 4)
    assert [item.calibration is original for item in sequence] == [True] * 4
    assert list(sequence.transforms) == [None] * 4
    assert [item.rejection_reason for item in sequence.calibrations] == ["INVALID_ANCHOR"] * 4
    empty = FrameRegistrationSequence.constant(original, 0)
    assert len(empty) == 0 and list(empty) == [] and empty[::-1] == []
    with pytest.raises(IndexError):
        empty[0]


def test_success_removes_only_exact_owned_database_and_known_sidecars(tmp_path):
    unrelated = tmp_path / "frame_registration_other.sqlite3"
    unrelated.write_bytes(b"other")
    with FrameRegistrationSequence(tmp_path) as sequence:
        sequence.append(registration())
        owned = sequence.path
        for suffix in ("-wal", "-shm"):
            Path = module.Path
            Path(str(owned) + suffix).write_bytes(b"owned")
    assert unrelated.read_bytes() == b"other" and list(tmp_path.iterdir()) == [unrelated]
    assert sequence._connection is None
    with pytest.raises(RuntimeError, match="not open"):
        sequence[0]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        sequence.__enter__()


def test_failed_producer_retains_accepted_rows_closes_handle_and_original_error(tmp_path):
    sequence, closed = FrameRegistrationSequence(tmp_path), []

    def records():
        try:
            yield registration()
            yield FrameRegistration(CourtCalibration(None, rejection_reason="LOST"), None)
            raise LookupError("producer failed")
        finally:
            closed.append(True)

    with pytest.raises(LookupError, match="producer failed"):
        with sequence:
            sequence.extend(records())
    assert closed == [True] and sequence._connection is None
    assert row_count(sequence.path) == 2
    renamed = sequence.path.with_suffix(".retained")
    sequence.path.rename(renamed)
    assert renamed.exists()


@pytest.mark.parametrize("broken", [
    FrameRegistration(CourtCalibration(np.array([[object()]], dtype=object)), None),
    FrameRegistration(CourtCalibration(np.array([(1,)], dtype=[("x", "i4")])), None),
    FrameRegistration(CourtCalibration(None, reprojection_error_px=object()), None),
    FrameRegistration(CourtCalibration(None), [[1, 0], [0, 1]]),
    object(),
])
def test_serialization_failure_preserves_accepted_records_and_retains_closed_database(tmp_path, broken):
    sequence = FrameRegistrationSequence(tmp_path)
    with pytest.raises(TypeError):
        with sequence:
            sequence.append(registration())
            sequence.append(broken)
    assert sequence._connection is None and row_count(sequence.path) == 1


def test_sqlite_mutation_failure_retains_closed_database_and_primary_error(tmp_path):
    sequence = FrameRegistrationSequence(tmp_path)
    with pytest.raises(sqlite3.IntegrityError, match="blocked registration"):
        with sequence:
            sequence.append(registration())
            sequence._connection.execute("CREATE TRIGGER reject_registration BEFORE INSERT ON registration_frames "
                                         "BEGIN SELECT RAISE(ABORT,'blocked registration'); END")
            sequence.append(registration(1))
    assert sequence._connection is None and row_count(sequence.path) == 1


def test_cleanup_failure_closes_connection_and_preserves_primary_exception(tmp_path, monkeypatch):
    real_connect = module.sqlite3.connect
    connections = []

    class FailingCommit:
        def __init__(self, path):
            self.connection = real_connect(path)
            self.fail, self.closed = False, False

        def execute(self, *args):
            return self.connection.execute(*args)

        def commit(self):
            if self.fail:
                raise sqlite3.OperationalError("registration commit failed")
            return self.connection.commit()

        def close(self):
            self.connection.close()
            self.closed = True

    def connect(path):
        connection = FailingCommit(path)
        connections.append(connection)
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    sequence = FrameRegistrationSequence(tmp_path)
    with pytest.raises(LookupError, match="primary producer failure") as caught:
        with sequence:
            sequence.append(registration())
            connections[0].fail = True
            raise LookupError("primary producer failure")
    assert connections[0].closed and sequence.path.exists() and sequence._connection is None
    assert any("registration commit failed" in note for note in caught.value.__notes__)


def test_setup_failure_retains_owned_file_and_original_error(tmp_path, monkeypatch):
    def fail(_path):
        raise sqlite3.OperationalError("registration connect failed")

    monkeypatch.setattr(module.sqlite3, "connect", fail)
    sequence = FrameRegistrationSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match="registration connect failed"):
        with sequence:
            pytest.fail("Failed setup must not enter")
    assert sequence.path.parent == tmp_path.resolve() and sequence.path.exists()
    assert sequence._connection is None


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "256"])
def test_invalid_batch_size_creates_no_file(tmp_path, value):
    with pytest.raises(ValueError):
        FrameRegistrationSequence(tmp_path, transaction_size=value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [True, -1, 1.5, "1"])
def test_invalid_static_count(value):
    with pytest.raises(ValueError):
        FrameRegistrationSequence.constant(calibration(), value)


def test_missing_output_directory_is_not_created(tmp_path):
    missing = tmp_path / "not-created"
    with pytest.raises(FileNotFoundError):
        with FrameRegistrationSequence(missing):
            pass
    assert not missing.exists()
