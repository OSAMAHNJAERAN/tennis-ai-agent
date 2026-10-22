"""Real disk result trees, independent JSON oracles and owned failure paths."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3

import pytest

from src.api.result_storage import DiskSeries, ResultStorage


def materialize(value):
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, DiskSeries)):
        return [materialize(item) for item in value]
    return value


def json_file(directory, raw):
    path = directory / "input.json"
    path.write_text(raw, encoding="utf-8")
    return path


@pytest.mark.parametrize("count", [0, 1, 64, 65, 129, 257])
def test_incremental_array_boundary_matches_independent_stdlib_json(count, tmp_path):
    original = {"rows": [{"frame": index, "point": [0, -0., None], "state": "MISSING" if index % 7 == 0 else "DETECTED"}
                          for index in range(count)], "metadata": {"unicode": "Tennis é 🎾", "missing": None}}
    raw = json.dumps(original)
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, raw))
        assert isinstance(parsed["rows"], DiskSeries) is (count > 64)
        assert materialize(parsed) == json.loads(raw)
        output = io.StringIO()
        storage.dump(parsed, output)
        assert output.getvalue() == json.dumps(json.loads(raw), allow_nan=False)
        if count:
            assert math.copysign(1, parsed["rows"][-1]["point"][1]) == -1


def test_nested_spills_duplicate_keys_last_wins_and_private_codec_does_not_collide(tmp_path):
    old = json.dumps(list(range(100)))
    final = {"series": ["series", 1], "object": ["scalar", 0], "__disk_series__": {"id": 17},
             "nested": [list(range(65)), [{"tag": "series", "value": 7}] * 66]}
    raw = '{"rows":' + old + ',"middle":true,"rows":' + json.dumps(final) + '}'
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, raw))
        assert isinstance(parsed["rows"]["nested"][0], DiskSeries)
        assert materialize(parsed) == json.loads(raw)
        assert "".join(storage.json_chunks(parsed)) == json.dumps(json.loads(raw), allow_nan=False)
        storage.set_root(parsed)
        destination = storage.publish(tmp_path / "completed.sqlite3")
    with ResultStorage.open(destination) as reader:
        assert materialize(reader.root()) == json.loads(raw)


@pytest.mark.parametrize("raw", ["null", "true", "false", "-0.0", "1e-300", "123456789012345678901234567890",
                                '"escaped\\n\\t\\r\\b\\f\\\\\\\"\\/\\uD83C\\uDFBE"',
                                '{"NaN":NaN,"positive":Infinity,"negative":-Infinity}'])
def test_scalar_semantics_match_stdlib_artifact_decoder(raw, tmp_path):
    expected = json.loads(raw, parse_constant=lambda _: None)
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, raw))
        assert parsed == expected
        assert "".join(storage.json_chunks(parsed)) == json.dumps(expected, allow_nan=False)


def test_overflow_is_preserved_and_strict_constant_callback_is_not_ignored(tmp_path):
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, '{"unknown":1e999,"array":[-1e999]}'))
        assert parsed["unknown"] == math.inf and parsed["array"] == [-math.inf]
        series = storage.series([parsed])
        assert series[0]["unknown"] == math.inf
        with pytest.raises(ValueError, match="Out of range"):
            storage.dump(series, io.StringIO())

        class StrictConstants:
            def __bool__(self):
                return False

            def __call__(self, token):
                raise ValueError("Forbidden constant " + token)

        with pytest.raises(ValueError, match="Forbidden constant NaN"):
            storage.read_json(json_file(tmp_path, "NaN"), parse_constant=StrictConstants())


@pytest.mark.parametrize("raw", ["", " ", "[] trailing", "{}{}", '{"x":1,}', "[1,]", "[1 2]", "[01]", "[1e]",
                                "[+1]", "[undefined]", '{"x" 1}', "{1:2}", '"unterminated', '"bad\\q"',
                                '"bad\ncontrol"', "[1", '{"x":', '{"x":[' + ','.join(map(str, range(257)))])
def test_all_syntax_and_eof_checked_before_returning_any_section(raw, tmp_path):
    with pytest.raises(ValueError):
        json.loads(raw, parse_constant=lambda _: None)
    with ResultStorage(tmp_path) as storage:
        with pytest.raises(ValueError):
            storage.read_json(json_file(tmp_path, raw))
        assert storage._connection is not None  # Caller can abstain for this entire optional section.
        storage.set_root({"section": None})
        published = storage.publish(tmp_path / "abstained.sqlite3")
    with closing(sqlite3.connect(published)) as database:
        assert database.execute("SELECT COUNT(*) FROM series").fetchone() == (0,)


@pytest.mark.parametrize("prefix", [65530, 65534, 65535, 65536, 131071])
def test_strings_escapes_numbers_and_unicode_cross_read_buffer_boundaries(prefix, tmp_path):
    text = "a" * prefix + '\\"\n🎾' + "b" * 23
    raw = '{"value":' + json.dumps(text, ensure_ascii=False) + ',"number":1.2345678901234567e-299}'
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, raw))
        assert parsed == json.loads(raw)
        assert "".join(storage.json_chunks(parsed)) == json.dumps(json.loads(raw), allow_nan=False)


@pytest.mark.parametrize("escape", ['\\n', '\\"', '\\\\', '\\uD83C\\uDFBE'])
def test_escape_backslash_is_last_character_in_input_buffer(escape, tmp_path):
    raw = '["' + 'a' * 65533 + escape + '"]'
    assert raw[65535] == '\\'
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(json_file(tmp_path, raw))
        assert parsed == json.loads(raw)
        assert "".join(storage.json_chunks(parsed)) == json.dumps(json.loads(raw), allow_nan=False)


def test_number_and_nonfinite_literal_split_across_input_buffers(tmp_path):
    with ResultStorage(tmp_path) as storage:
        for token in ('-1.2345678901234567e-299', '-Infinity', '1e999'):
            raw = ' ' * 65534 + token
            observed = storage.read_json(json_file(tmp_path, raw))
            expected = json.loads(raw, parse_constant=lambda _: None)
            assert observed == expected


def test_parser_only_requests_bounded_reads_and_consumes_complete_input(tmp_path, monkeypatch):
    original = {"frames": [{"frame": index, "text": "é" * 20} for index in range(10000)]}
    path = json_file(tmp_path, json.dumps(original))
    real_open, reads, closed = Path.open, [], []

    class BoundedInput:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()
            closed.append(True)

        def read(self, size=-1):
            assert 0 < size <= 65536, "A whole-file read defeats disk staging"
            value = self.stream.read(size)
            reads.append((size, len(value)))
            return value

    def bounded_open(file, *args, **kwargs):
        stream = real_open(file, *args, **kwargs)
        return BoundedInput(stream) if file == path else stream

    monkeypatch.setattr(Path, "open", bounded_open)
    with ResultStorage(tmp_path) as storage:
        parsed = storage.read_json(path)
        assert isinstance(parsed["frames"], DiskSeries)
        assert len(parsed["frames"]) == 10000
        assert parsed["frames"][4321] == original["frames"][4321]
        assert parsed["frames"][-1] == original["frames"][-1]
    assert len(reads) > 10 and reads[-1] == (65536, 0) and closed == [True]


def test_series_reads_snapshot_order_negative_slice_and_integer_replacement(tmp_path):
    with ResultStorage(tmp_path) as storage:
        original = {"point": [0, -0., 1.25], "nested": {"state": "DETECTED"}}
        series = storage.series([original, None, [], {"frame": 3}])
        original["point"][0] = 100
        snapshot = series[0]
        snapshot["nested"]["state"] = "MISSING"
        assert series[0] == {"point": [0, -0., 1.25], "nested": {"state": "DETECTED"}}
        assert series[-4] == series[0]
        assert series[::-2] == [{"frame": 3}, None]
        assert series[1:1] == []
        replacement = {"frame": 9, "values": [0, None]}
        series[-1] = replacement
        replacement["values"].append(99)
        assert list(series) == [series[0], None, [], {"frame": 9, "values": [0, None]}]
        for index in (4, -5):
            with pytest.raises(IndexError):
                _ = series[index]
        with pytest.raises(TypeError):
            series[1:2] = []
        with pytest.raises(TypeError):
            _ = series[.5]


def test_serialization_exact_key_rules_escape_spacing_and_numeric_types(tmp_path):
    value = {"": [], "nested": {}, 1: "int key", 2.5: "float key", None: "null key", False: "bool key",
             "unicode": "é🎾\ud800", "values": [0, -0., 0., 1e-12, 1e20, True, None, (1, 2)],
             "codec": {"series": 1, "scalar": ["object", [["key", "value"]]]}}
    with ResultStorage(tmp_path) as storage:
        series = storage.series([value])
        assert "".join(storage.json_chunks(series)) == json.dumps([value], allow_nan=False)
        bad = {(1, 2): "invalid key"}
        with pytest.raises(TypeError):
            storage.dump(bad, io.StringIO())
        circular = []
        circular.append(circular)
        with pytest.raises(ValueError, match="Circular"):
            storage.series([circular])
        with pytest.raises(ValueError, match="Circular"):
            storage.dump(circular, io.StringIO())
        self_reference = storage.series()
        self_reference.append(self_reference)
        with pytest.raises(ValueError, match="Circular"):
            storage.dump(self_reference, io.StringIO())


def test_publication_compacts_nested_references_duplicates_and_replacements_on_disk(tmp_path):
    with ResultStorage(tmp_path) as storage:
        abandoned = storage.series(range(300))
        retained = storage.series(range(65))
        replaced_child = storage.series(range(700))
        parent = storage.series([{"child": replaced_child}, {"child": retained}])
        parent[0] = {"child": retained}
        storage.set_root({"unused": abandoned})
        storage.set_root({"rows": parent, "same": retained})
        original_path = storage.path
        destination = storage.publish(tmp_path / "generation.sqlite3")
        assert storage._connection is None and not original_path.exists()
    with closing(sqlite3.connect(destination)) as database:
        assert database.execute("SELECT COUNT(*) FROM series").fetchone() == (2,)
        assert database.execute("SELECT COUNT(*) FROM items").fetchone() == (67,)
        assert database.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    with ResultStorage.open(destination) as reader:
        first = reader.root()
        first["added"] = "does not write through"
        assert "added" not in reader.root()
        assert materialize(reader.root()) == {"rows": [{"child": list(range(65))}] * 2, "same": list(range(65))}
        with pytest.raises(TypeError, match="read-only"):
            first["rows"].append(None)
        reader.close()
        reader.close()
    assert destination.exists()


def test_readonly_reader_supports_sequential_cross_thread_http_iteration(tmp_path):
    with ResultStorage(tmp_path) as storage:
        storage.set_root({"frames": storage.series(range(300))})
        destination = storage.publish(tmp_path / "readable.sqlite3")
    with ResultStorage.open(destination) as reader:
        series = reader.root()["frames"]
        with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
            assert first.submit(lambda: series[11]).result() == 11
            assert second.submit(lambda: series[-1]).result() == 299
            assert second.submit(lambda: "".join(reader.json_chunks({"frames": series[:3]}))).result() == '{"frames": [0, 1, 2]}'


def test_scratch_cleanup_is_exact_and_failure_retains_closed_accepted_rows(tmp_path):
    unrelated = tmp_path / "result_storage_unrelated.sqlite3"
    unrelated.write_bytes(b"preserve")
    with ResultStorage(tmp_path) as storage:
        storage.series(range(3))
        success_path = storage.path
        assert storage._one("PRAGMA cache_size") == (-2048,)
        assert storage._one("PRAGMA mmap_size") == (0,)
        assert storage._one("PRAGMA temp_store") == (1,)
    assert not success_path.exists() and unrelated.read_bytes() == b"preserve"
    with pytest.raises(RuntimeError, match="primary"):
        with ResultStorage(tmp_path) as failed:
            failed.series(range(300))
            failure_path = failed.path
            raise RuntimeError("primary")
    assert failed._connection is None
    with closing(sqlite3.connect(failure_path)) as database:
        assert database.execute("SELECT COUNT(*) FROM items").fetchone() == (300,)


def test_transaction_batch_and_mutation_failure_preserve_current_snapshot(tmp_path, monkeypatch):
    with ResultStorage(tmp_path) as storage:
        series = storage.series()
        for index in range(254):
            series.append(index)
        with closing(sqlite3.connect(storage.path)) as external:
            assert external.execute("SELECT COUNT(*) FROM items").fetchone() == (0,)
        series.append(254)  # One series mutation plus 255 rows commits the 256 batch.
        with closing(sqlite3.connect(storage.path)) as external:
            assert external.execute("SELECT COUNT(*) FROM items").fetchone() == (255,)
        execute = storage._execute

        def fail_refs(sql, parameters=()):
            if sql.startswith("DELETE FROM refs"):
                raise OSError("injected index update")
            return execute(sql, parameters)

        monkeypatch.setattr(storage, "_execute", fail_refs)
        with pytest.raises(OSError, match="injected"):
            series[0] = {"new": True}
        monkeypatch.setattr(storage, "_execute", execute)
        assert len(series) == 255 and series[0] == 0


@pytest.mark.parametrize("failure", ["compact", "close", "rename", "existing"])
def test_publication_failures_keep_private_diagnostics_and_previous_destination(tmp_path, monkeypatch, failure):
    destination = tmp_path / "generation.sqlite3"
    original = b"existing immutable publication"
    with pytest.raises((OSError, FileExistsError)):
        with ResultStorage(tmp_path) as storage:
            storage.set_root({"frames": storage.series(range(65))})
            private = storage.path
            if failure == "existing":
                destination.write_bytes(original)
            elif failure == "compact":
                monkeypatch.setattr(storage, "_compact", lambda: (_ for _ in ()).throw(OSError("compact failed")))
            elif failure == "close":
                real_close = storage.close

                def failing_close():
                    real_close()
                    raise OSError("close failed")

                monkeypatch.setattr(storage, "close", failing_close)
            elif failure == "rename":
                monkeypatch.setattr(Path, "rename", lambda *_args: (_ for _ in ()).throw(OSError("rename failed")))
            storage.publish(destination)
    assert storage._connection is None and private.exists()
    if failure == "existing":
        assert destination.read_bytes() == original
    else:
        assert not destination.exists()


def test_primary_failure_survives_cleanup_failure_and_producer_is_closed(tmp_path, monkeypatch):
    storage = ResultStorage(tmp_path)
    closed = []

    def records():
        try:
            yield {"value": 1}
            raise RuntimeError("producer primary")
        finally:
            closed.append(True)

    with pytest.raises(RuntimeError, match="producer primary") as error:
        with storage:
            real_close = storage.close

            def fail_close():
                real_close()
                raise OSError("cleanup secondary")

            monkeypatch.setattr(storage, "close", fail_close)
            storage.series(records())
    assert closed == [True] and storage._connection is None and storage.path.exists()
    assert any("cleanup secondary" in note for note in error.value.__notes__)


def test_readonly_missing_or_invalid_database_does_not_create_or_delete_source(tmp_path):
    missing = tmp_path / 'missing.sqlite3'
    with pytest.raises(sqlite3.OperationalError):
        with ResultStorage.open(missing):
            pytest.fail('Missing published database must not be created')
    assert not missing.exists()
    corrupt = tmp_path / 'corrupt.sqlite3'
    corrupt.write_bytes(b'not a database')
    with pytest.raises(sqlite3.DatabaseError):
        with ResultStorage.open(corrupt):
            pytest.fail('Invalid database must fail during reader preflight')
    assert corrupt.read_bytes() == b'not a database'


def test_configured_432000_frame_series_streams_without_full_materialization(tmp_path, monkeypatch):
    count = 432000
    with ResultStorage(tmp_path) as storage:
        series = storage.series(range(count))
        assert len(series) == count
        assert series[-1] == count - 1 and series[-count] == 0
        assert series[count - 9:count:3] == [count - 9, count - 6, count - 3]
        series[123456] = None
        storage.set_root({"frames": series})
        reads = []
        read_count = 0
        original_get = DiskSeries.__getitem__

        def tracked_get(self, index):
            nonlocal read_count
            read_count += 1
            if len(reads) < 10:
                reads.append(index)
            return original_get(self, index)

        monkeypatch.setattr(DiskSeries, "__getitem__", tracked_get)
        chunks = storage.json_chunks({"frames": series})
        first = next(chunks)
        assert len(first) <= 65536 and reads == list(range(10))
        assert 10 < read_count < 20000, "First output must precede full-series consumption"
        # An independent arithmetic oracle hashes one scalar at a time. Neither
        # the expected output nor the observed JSON is retained as a full string.
        expected = hashlib.sha256(b'{"frames": [')
        for index in range(count):
            if index:
                expected.update(b", ")
            expected.update(b"null" if index == 123456 else str(index).encode("ascii"))
        expected.update(b"]}")
        observed = hashlib.sha256(first.encode("ascii"))
        for chunk in chunks:
            assert len(chunk) <= 65536
            observed.update(chunk.encode("ascii"))
        assert observed.digest() == expected.digest()
        destination = storage.publish(tmp_path / "capacity.sqlite3")
    with ResultStorage.open(destination) as reader:
        reopened = reader.root()["frames"]
        assert len(reopened) == count and reopened[123456] is None and reopened[-1] == count - 1
