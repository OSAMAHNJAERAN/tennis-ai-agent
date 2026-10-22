"""Exact observation snapshots, bounded raw export, and owned failure cleanup."""

import ast
from collections.abc import Sequence
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import struct
from typing import Any
import weakref

import numpy as np
import pytest

from src.utils import identity_frame_sequence as module
from src.utils.identity_frame_sequence import IdentityFrameSequence, write_identity_frame_json


@pytest.fixture
def sanitize():
    # Exercise the real converter without importing the inference orchestration
    # or its model libraries into these isolated storage/export regressions.
    path = Path(__file__).resolve().parents[1] / "src/pipeline/phase6_pipeline.py"
    source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in source.body
                    if isinstance(node, ast.FunctionDef) and node.name == "sanitize_json_value")
    namespace = {"np": np, "Any": Any}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["sanitize_json_value"]


def assert_exact(actual, expected):
    """Independent typed/bitwise comparison; equality hides -0 and NumPy loss."""
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype.str == expected.dtype.str
        assert actual.shape == expected.shape
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
    elif isinstance(expected, np.generic):
        assert type(actual) is type(expected)
        assert actual.dtype.str == expected.dtype.str
        assert actual.tobytes() == expected.tobytes()
    else:
        assert type(actual) is type(expected)
        if isinstance(expected, float):
            assert struct.pack("!d", actual) == struct.pack("!d", expected)
        elif isinstance(expected, dict):
            assert len(actual) == len(expected)
            for (actual_key, actual_value), (expected_key, expected_value) in zip(
                    actual.items(), expected.items(), strict=True):
                assert_exact(actual_key, expected_key)
                assert_exact(actual_value, expected_value)
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected)
            for actual_value, expected_value in zip(actual, expected, strict=True):
                assert_exact(actual_value, expected_value)
        else:
            assert actual == expected


def frame(index):
    return {1: {"position_px": [index + .125, -0.], "confidence": .875,
                "joints": {"left_wrist": None, "right_wrist": [index + .25, index + .5]}},
            2: {"state": "MISSING" if index % 3 == 0 else "DETECTED",
                "bbox_xyxy": None if index % 3 == 0 else [index, 10, index + 20, 30],
                "contact_event": None}}


def reference_export(path, metadata, array_key, frames, sanitize):
    complete = {**metadata, array_key: frames}
    with path.open("w", encoding="utf-8") as stream:
        json.dump(sanitize(complete), stream, indent=2, allow_nan=False)
    return path.read_bytes()


def assert_removed(sequence, sentinel):
    assert sequence._connection is None
    assert sequence.path is not None
    for suffix in ("", "-journal", "-wal", "-shm"):
        assert not Path(str(sequence.path) + suffix).exists()
    assert sentinel.read_bytes() == b"unrelated diagnostic"


def sentinel_file(tmp_path):
    path = tmp_path / "identity_frames_unrelated.sqlite3"
    path.write_bytes(b"unrelated diagnostic")
    return path


def test_native_identity_order_types_and_ieee_values_round_trip(tmp_path):
    special_nan = struct.unpack("!d", bytes.fromhex("fff8000000000123"))[0]
    record = {2: None, 1: {"bools": [True, False], "integer": 2 ** 80 + 7,
                          "text": "Tennis \u00e9 \U0001f3be", "zero": -0.,
                          "precise": float.fromhex("0x1.0000000000001p+0"),
                          "smallest": float.fromhex("0x0.0000000000001p-1022"),
                          "nonfinite": [float("inf"), float("-inf"), special_nan],
                          "tuple": (None, 4, -0., [True, "inside tuple"]),
                          "empty": {}}}
    with IdentityFrameSequence(tmp_path) as sequence:
        assert isinstance(sequence, Sequence)
        sequence.append(record)
        sequence.append({})
        first = sequence[0]
        assert first is not record
        assert list(first) == [2, 1]
        assert_exact(first, record)
        assert sequence[1] == {}


@pytest.mark.parametrize("dtype", ["?", "i1", "<i8", ">i8", "<u8", "f2", "f4", "f8", "c8", "c16"])
def test_numpy_scalar_dtype_and_bytes_are_preserved(tmp_path, dtype):
    values = [np.array(1, dtype=dtype)[()], np.array(0, dtype=dtype)[()]]
    if np.dtype(dtype).kind in "fc":
        values.extend([np.array(-0., dtype=dtype)[()], np.array(float("inf"), dtype=dtype)[()]])
    with IdentityFrameSequence(tmp_path) as sequence:
        for value in values:
            record = {1: {"value": value}, 2: None}
            sequence.append(record)
            assert_exact(sequence[-1], record)


def test_numpy_array_shape_endianness_noncontiguous_and_nonfinite_bits(tmp_path):
    bits = np.array([0x80000000, 0x00000001, 0x7FC12345, 0xFFC54321,
                     0x7F800000, 0xFF800000], dtype=np.uint32)
    arrays = [bits.view(np.float32).reshape(2, 3),
              np.asfortranarray(np.arange(24, dtype=">f8").reshape(4, 6))[:, ::2],
              np.empty((0, 3), dtype=np.int16), np.array(-0., dtype=">f8"),
              np.array([1 + 2j, -3 + .125j], dtype=np.complex64)]
    with IdentityFrameSequence(tmp_path) as sequence:
        sequence.append({1: arrays, 2: (np.bool_(False), np.int64(3), np.float64(-0.))})
        decoded = sequence[0]
        assert_exact(decoded, {1: arrays, 2: (np.bool_(False), np.int64(3), np.float64(-0.))})
        for restored, original in zip(decoded[1], arrays, strict=True):
            assert restored is not original
            assert not np.shares_memory(restored, original)


def test_input_and_returned_nested_mutations_never_write_through(tmp_path):
    record = {1: {"list": [1., {"tuple": ([2.],)}], "array": np.array([[3., -0.]])}, 2: None}
    expected = {1: {"list": [1., {"tuple": ([2.],)}], "array": np.array([[3., -0.]])}, 2: None}
    with IdentityFrameSequence(tmp_path) as sequence:
        sequence.append(record)
        record[1]["list"][1]["tuple"][0][0] = 9.
        record[1]["array"][0, 0] = 10.
        record[2] = "changed"
        first, second = sequence[0], sequence[0]
        assert_exact(first, expected)
        assert_exact(second, expected)
        assert first is not second and first[1]["list"] is not second[1]["list"]
        assert not np.shares_memory(first[1]["array"], second[1]["array"])
        first[1]["list"][1]["tuple"][0][0] = 11.
        first[1]["array"][0, 0] = 12.
        assert_exact(sequence[0], expected)


def test_private_tags_do_not_collide_with_user_dictionary_or_list_values(tmp_path):
    record = {1: {"array": ["array", "<f8", [1], "0000000000000000"],
                  "scalar": {"tag": "scalar", "dtype": "O", "payload": "malicious"},
                  "dict": ["dict", [["str", "key"]]],
                  "__identity_frame__": {"float": ["float", "deadbeef"]}}, 2: None}
    with IdentityFrameSequence(tmp_path) as sequence:
        sequence.append(record)
        assert_exact(sequence[0], record)


def test_index_iteration_slices_empty_and_closed_states(tmp_path):
    sequence = IdentityFrameSequence(tmp_path)
    with sequence:
        assert len(sequence) == 0 and not sequence and list(sequence) == []
        with pytest.raises(IndexError):
            sequence[0]
        for index in range(9):
            sequence.append(frame(index))
        for index in (4, 2, 6, 0, 8, -1, -9, np.int64(3)):
            assert_exact(sequence[index], frame(int(index) % 9))
        assert_exact(sequence[2:7:2], [frame(index) for index in (2, 4, 6)])
        assert_exact(sequence[::-1], [frame(index) for index in range(8, -1, -1)])
        assert_exact(list(sequence), [frame(index) for index in range(9)])
        for invalid in (-10, 9):
            with pytest.raises(IndexError):
                sequence[invalid]
        for invalid in (1.5, "0"):
            with pytest.raises(TypeError):
                sequence[invalid]
    with pytest.raises(RuntimeError, match="not open"):
        sequence[0]
    with pytest.raises(RuntimeError, match="cannot be reused"):
        sequence.__enter__()


class CursorProbe:
    def __init__(self, cursor, connection):
        self.cursor, self.connection, self.closed = cursor, connection, False
        connection.active += 1
        connection.peak_active = max(connection.peak_active, connection.active)

    def fetchone(self):
        if self.connection.fail_fetch:
            raise sqlite3.OperationalError("injected fetch failure")
        return self.cursor.fetchone()

    def close(self):
        if not self.closed:
            self.closed = True
            self.connection.active -= 1
            self.cursor.close()

    def __getattr__(self, name):
        return getattr(self.cursor, name)


class ConnectionProbe:
    def __init__(self, connection):
        self.connection = connection
        self.active = self.peak_active = self.selects = self.commits = 0
        self.fail_execute = None
        self.fail_fetch = self.fail_commit = self.fail_close = False
        self.closed = False

    def execute(self, sql, parameters=()):
        if self.fail_execute and sql.lstrip().upper().startswith(self.fail_execute):
            raise sqlite3.OperationalError("injected execute failure")
        if sql.lstrip().upper().startswith("SELECT"):
            self.selects += 1
        return CursorProbe(self.connection.execute(sql, parameters), self)

    def commit(self):
        self.commits += 1
        if self.fail_commit:
            raise sqlite3.OperationalError("injected commit failure")
        return self.connection.commit()

    def close(self):
        self.connection.close()
        self.closed = True
        if self.fail_close:
            raise sqlite3.OperationalError("injected close failure")


@pytest.fixture
def connection_probes(monkeypatch):
    original, probes = module.sqlite3.connect, []

    def connect(*args, **kwargs):
        probe = ConnectionProbe(original(*args, **kwargs))
        probes.append(probe)
        return probe

    monkeypatch.setattr(module.sqlite3, "connect", connect)
    return probes


def test_cache_bounded_commits_and_all_setup_read_write_cursors_close(tmp_path, connection_probes):
    with IdentityFrameSequence(tmp_path, transaction_size=2) as sequence:
        probe = connection_probes[0]
        assert probe.active == 0
        for pragma, expected in (("cache_size", -2048), ("temp_store", 1), ("mmap_size", 0)):
            assert probe.connection.execute(f"PRAGMA {pragma}").fetchone()[0] == expected
        assert probe.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
        first_commit = probe.commits
        sequence.append(frame(0))
        assert probe.commits == first_commit and probe.active == 0
        sequence.append(frame(1))
        assert probe.commits == first_commit + 1 and probe.active == 0
        sequence.append(frame(2))
        assert probe.commits == first_commit + 1
        with closing(probe.connection.__class__(sequence.path)) as reader:
            assert reader.execute("SELECT count(*) FROM identity_frames").fetchone()[0] == 2
        iterator = iter(sequence)
        assert_exact(next(iterator), frame(0))
        assert probe.active == 0, "Indexed iteration must not suspend an open SQLite cursor"
        iterator.close()
        assert probe.active == 0
        assert_exact(sequence[-1], frame(2))
        assert probe.active == 0 and probe.peak_active == 1
    assert probe.closed and probe.active == 0


@pytest.mark.parametrize("count", [0, 1, 64, 512])
def test_complete_raw_export_bytes_match_materialized_native_text_json(tmp_path, sanitize, count):
    metadata = {"schema_version": "2.0", "enabled": bool(count), "unicode": "Tennis \u00e9",
                "flags": {"accuracy_validated": False}, "precise": np.float32(.125),
                "missing": float("nan")}
    expected = reference_export(tmp_path / "reference.json", metadata, "pose_frames",
                                [frame(index) for index in range(count)], sanitize)
    with IdentityFrameSequence(tmp_path) as sequence:
        for index in range(count):
            sequence.append(frame(index))
        write_identity_frame_json(tmp_path / "actual.json", metadata, "pose_frames", sequence, sanitize)
    actual = (tmp_path / "actual.json").read_bytes()
    assert actual == expected
    assert not actual.endswith(b"\n") and not actual.endswith(b"\r")
    parsed = json.loads(actual)
    assert list(parsed)[-1] == "pose_frames"
    assert len(parsed["pose_frames"]) == count
    if not count:
        assert b'"pose_frames": []' in actual


@pytest.mark.parametrize("count", [64, 512])
def test_export_reads_each_frame_once_without_retaining_decoded_arrays(
        tmp_path, sanitize, count, connection_probes):
    metadata = {"schema_version": "1.0", "enabled": True, "accuracy_validated": False}
    expected = reference_export(tmp_path / "reference.json", metadata, "frames",
                                [{1: {"position_px": np.array([index + .125, -0.])}, 2: None}
                                 for index in range(count)], sanitize)
    counters = {"reads": 0, "live": 0, "peak": 0}

    def released():
        counters["live"] -= 1

    class ObservedFrames:
        def __init__(self, sequence):
            self.sequence = sequence

        def __len__(self):
            return len(self.sequence)

        def __iter__(self):
            for index in range(len(self)):
                assert counters["reads"] == index, "Exporter repeated or skipped an ordered frame read"
                observed = self.sequence[index]
                counters["reads"] += 1
                counters["live"] += 1
                counters["peak"] = max(counters["peak"], counters["live"])
                weakref.finalize(observed[1]["position_px"], released)
                yield observed

    with IdentityFrameSequence(tmp_path) as sequence:
        for index in range(count):
            sequence.append({1: {"position_px": np.array([index + .125, -0.])}, 2: None})
        probe = connection_probes[0]
        probe.selects = 0
        write_identity_frame_json(tmp_path / "actual.json", metadata, "frames", ObservedFrames(sequence), sanitize)
        assert counters["reads"] == count
        assert counters["live"] == 0 and counters["peak"] < 4
        assert probe.selects == count and probe.active == 0 and probe.peak_active == 1
    assert (tmp_path / "actual.json").read_bytes() == expected


@pytest.mark.parametrize("record,expected_error", [
    ({1: {"value": (float("nan"),)}, 2: None}, ValueError),
    ({1: {"value": (np.float32(.5),)}, 2: None}, TypeError),
    ({1: {"value": np.array([float("inf")])}, 2: None}, ValueError),
    ({1: {"value": np.array([1 + 2j], dtype=np.complex64)}, 2: None}, TypeError),
])
def test_raw_export_preserves_existing_tuple_and_array_failure_semantics(
        tmp_path, sanitize, record, expected_error):
    metadata = {"schema_version": "1.0", "enabled": True}
    with pytest.raises(expected_error) as original:
        reference_export(tmp_path / "reference.json", metadata, "frames", [record], sanitize)
    sequence = IdentityFrameSequence(tmp_path)
    sentinel = sentinel_file(tmp_path)
    with pytest.raises(expected_error) as observed:
        with sequence:
            sequence.append(record)
            write_identity_frame_json(tmp_path / "actual.json", metadata, "frames", sequence, sanitize)
    assert str(observed.value) == str(original.value)
    assert_removed(sequence, sentinel)


def test_raw_export_sanitizes_direct_nonfinite_scalars_without_boolean_coercion(tmp_path, sanitize):
    record = {1: {"values": [float("nan"), float("inf"), np.float32(float("-inf")),
                             np.bool_(False), True, np.int64(2), -0.]}, 2: None}
    metadata = {"schema_version": "1.0", "enabled": True}
    expected = reference_export(tmp_path / "reference.json", metadata, "frames", [record], sanitize)
    with IdentityFrameSequence(tmp_path) as sequence:
        sequence.append(record)
        write_identity_frame_json(tmp_path / "actual.json", metadata, "frames", sequence, sanitize)
    assert (tmp_path / "actual.json").read_bytes() == expected
    values = json.loads(expected)["frames"][0]["1"]["values"]
    assert values[:6] == [None, None, None, False, True, 2]
    assert type(values[3]) is bool and type(values[4]) is bool and type(values[5]) is int
    assert struct.pack("!d", values[-1]) == struct.pack("!d", -0.)


@pytest.mark.parametrize("array_key", ["frames", None, 1])
def test_invalid_raw_array_key_is_rejected_before_output_creation(tmp_path, sanitize, array_key):
    metadata = {"frames": "already here"}
    with pytest.raises(ValueError):
        write_identity_frame_json(tmp_path / "not-created.json", metadata, array_key, [], sanitize)
    assert not (tmp_path / "not-created.json").exists()


def test_success_cleanup_owns_only_exclusive_database_and_known_sidecars(tmp_path):
    sentinel = sentinel_file(tmp_path)
    first, second = IdentityFrameSequence(tmp_path), IdentityFrameSequence(tmp_path)
    with first, second:
        first.append(frame(0))
        second.append(frame(1))
        assert first.path != second.path
        for sequence in (first, second):
            assert sequence.path.parent == tmp_path.resolve()
            for suffix in ("-wal", "-shm"):
                Path(str(sequence.path) + suffix).write_bytes(b"owned sidecar")
    assert_removed(first, sentinel)
    assert_removed(second, sentinel)
    assert list(tmp_path.iterdir()) == [sentinel]


def test_primary_body_failure_closes_and_removes_partial_owned_storage(tmp_path):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path)
    primary = LookupError("original observation failure")
    with pytest.raises(LookupError) as caught:
        with sequence:
            sequence.append(frame(0))
            raise primary
    assert caught.value is primary
    assert_removed(sequence, sentinel)


@pytest.mark.parametrize("record", [[], None, {True: None}, {1.5: None}, {"1": None},
                                    {1: object()}, {1: np.array([object()], dtype=object)},
                                    {1: np.array([(1.,)], dtype=[("x", "f8")])},
                                    {1: np.void(b"unsafe")}])
def test_bad_identity_or_unsafe_value_does_not_append_and_closes_owned_storage(tmp_path, record):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path)
    with pytest.raises(TypeError):
        with sequence:
            sequence.append(frame(0))
            assert len(sequence) == 1
            sequence.append(record)
    assert len(sequence) == 1
    assert_removed(sequence, sentinel)


def test_setup_failure_removes_exact_private_file_and_preserves_exception(tmp_path, monkeypatch):
    sentinel = sentinel_file(tmp_path)
    primary = sqlite3.OperationalError("injected setup failure")

    def failed_connect(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(module.sqlite3, "connect", failed_connect)
    sequence = IdentityFrameSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError) as caught:
        with sequence:
            pytest.fail("Failed setup must never enter the caller body")
    assert caught.value is primary
    assert_removed(sequence, sentinel)


@pytest.mark.parametrize("failure", ["insert", "commit", "select", "fetch", "decode"])
def test_write_commit_read_and_decode_failures_close_cursors_and_remove_storage(
        tmp_path, connection_probes, failure):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path, transaction_size=1)
    error = sqlite3.OperationalError if failure != "decode" else (ValueError, TypeError)
    with pytest.raises(error):
        with sequence:
            sequence.append(frame(0))
            probe = connection_probes[0]
            if failure == "insert":
                probe.fail_execute = "INSERT"
                sequence.append(frame(1))
            elif failure == "commit":
                probe.fail_commit = True
                sequence.append(frame(1))
            elif failure == "select":
                probe.fail_execute = "SELECT"
                sequence[0]
            elif failure == "fetch":
                probe.fail_fetch = True
                sequence[0]
            else:
                probe.connection.execute("UPDATE identity_frames SET observation=? WHERE frame_index=0",
                                         ('["array","O",[1],"0000000000000000"]',))
                sequence[0]
    assert probe.closed and probe.active == 0
    assert_removed(sequence, sentinel)


def test_cleanup_close_failure_does_not_replace_primary_body_error(tmp_path, connection_probes):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path)
    primary = LookupError("primary caller failure")
    with pytest.raises(LookupError) as caught:
        with sequence:
            sequence.append(frame(0))
            connection_probes[0].fail_close = True
            raise primary
    assert caught.value is primary
    assert any("injected close failure" in note for note in caught.value.__notes__)
    assert connection_probes[0].closed
    assert_removed(sequence, sentinel)


@pytest.mark.parametrize("failure", ["commit", "close"])
def test_clean_exit_reports_cleanup_error_after_retiring_owned_handle_and_files(
        tmp_path, connection_probes, failure):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path)
    with pytest.raises(sqlite3.OperationalError, match=f"injected {failure} failure"):
        with sequence:
            sequence.append(frame(0))
            setattr(connection_probes[0], "fail_" + failure, True)
    assert connection_probes[0].closed and connection_probes[0].active == 0
    assert_removed(sequence, sentinel)


def test_cleanup_unlink_failure_is_a_note_on_original_error_and_preserves_sentinel(tmp_path, monkeypatch):
    sentinel = sentinel_file(tmp_path)
    sequence = IdentityFrameSequence(tmp_path)
    primary = LookupError("primary before unlink failure")
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if sequence.path is not None and path == sequence.path:
            raise PermissionError("injected exact-owned unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(LookupError) as caught:
        with sequence:
            sequence.append(frame(0))
            raise primary
    assert caught.value is primary and sequence._connection is None
    assert any("injected exact-owned unlink failure" in note for note in caught.value.__notes__)
    assert sequence.path.exists() and sentinel.read_bytes() == b"unrelated diagnostic"
    # A Windows rename proves the closed diagnostic is no longer held open.
    retained = sequence.path.with_suffix(".closed-diagnostic")
    sequence.path.rename(retained)
    assert retained.exists()


@pytest.mark.parametrize("transaction_size", [0, -1, True, 1.5, "256"])
def test_invalid_commit_batch_creates_no_private_files(tmp_path, transaction_size):
    with pytest.raises(ValueError):
        IdentityFrameSequence(tmp_path, transaction_size=transaction_size)
    assert list(tmp_path.iterdir()) == []


def test_missing_output_directory_is_never_created(tmp_path):
    missing = tmp_path / "missing-output"
    with pytest.raises(FileNotFoundError):
        with IdentityFrameSequence(missing):
            pytest.fail("Missing directory must not enter storage")
    assert not missing.exists()
