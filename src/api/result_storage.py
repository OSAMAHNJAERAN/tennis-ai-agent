"""Owned result trees whose large arrays live in bounded SQLite storage.

The private tagged codec is outside user objects, so arbitrary evidence keys
cannot be mistaken for references. Memory scales with the current record,
object metadata and nesting depth, rather than the total number of frames.
"""

from collections.abc import Mapping, Sequence
import json
import operator
from pathlib import Path
import re
import sqlite3
import tempfile


class DiskSeries(Sequence):
    """An indexed array; reads decode fresh records and writes are explicit."""

    def __init__(self, storage, series_id):
        self._storage, self._series_id = storage, series_id

    def __len__(self):
        row = self._storage._one("SELECT length FROM series WHERE id=?", (self._series_id,))
        if row is None:
            raise ValueError("Result series is missing")
        return row[0]

    def _index(self, index):
        index = operator.index(index)
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(index)
        return index

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = self._index(index)
        row = self._storage._one("SELECT payload FROM items WHERE series_id=? AND position=?",
                                 (self._series_id, index))
        if row is None:
            raise ValueError("Result series record is missing")
        return self._storage._decode(json.loads(row[0]))

    def __iter__(self):
        # Each indexed read closes its cursor before yielding. A consumer may
        # stop iteration without leaving a database cursor open.
        for index in range(len(self)):
            yield self[index]

    def append(self, value):
        self._storage._write_item(self._series_id, len(self), value, append=True)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError("Result series replacement requires an integer index")
        self._storage._write_item(self._series_id, self._index(index), value, append=False)


class ResultStorage:
    """Exclusive scratch storage, or a sequentially used read-only reader.

    Successful unpublished contexts remove only their exact scratch files.
    Failed contexts close and retain diagnostics. Publication compacts raw
    staging out of the root tree, closes the database, then renames it to a
    fresh destination. The caller owns the authoritative generation manifest.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._path = self._scratch_path = self._connection = None
        self._readonly = self._published = self._retain = self._entered = False
        self._pending = 0

    @classmethod
    def open(cls, path):
        storage = cls(Path(path).parent)
        storage._path = Path(path).resolve()
        storage._readonly = True
        return storage

    @property
    def path(self):
        return self._path

    def _require_open(self, *, write=False):
        if self._connection is None:
            raise RuntimeError("Result storage is not open")
        if write and self._readonly:
            raise TypeError("Published result storage is read-only")
        return self._connection

    def _execute(self, sql, parameters=()):
        cursor = self._require_open().execute(sql, parameters)
        try:
            return cursor.lastrowid
        finally:
            cursor.close()

    def _one(self, sql, parameters=()):
        cursor = self._require_open().execute(sql, parameters)
        try:
            return cursor.fetchone()
        finally:
            cursor.close()

    def __enter__(self):
        if self._entered:
            raise RuntimeError("Result storage cannot be reused")
        self._entered = True
        try:
            if self._readonly:
                self._connection = sqlite3.connect(self._path.as_uri() + "?mode=ro", uri=True,
                                                   check_same_thread=False)
                self._execute("PRAGMA query_only=ON")
            else:
                with tempfile.NamedTemporaryFile(dir=self.directory, prefix=".result_storage_",
                                                  suffix=".sqlite3", delete=False) as temporary:
                    self._path = self._scratch_path = Path(temporary.name).resolve()
                if self._path.parent != self.directory:
                    raise ValueError("Owned result storage escaped its directory")
                self._connection = sqlite3.connect(self._path, check_same_thread=False)
                self._one("PRAGMA journal_mode=DELETE")
            self._execute("PRAGMA cache_size=-2048")
            self._execute("PRAGMA temp_store=FILE")
            self._one("PRAGMA mmap_size=0")
            if self._readonly:
                # A published generation is immutable. Keep validation and all
                # response reads in one snapshot, released when we close, so
                # SQLite need not reacquire a read transaction for every row.
                self._execute("BEGIN")
                if self._one("SELECT version FROM storage_format") != (1,):
                    raise ValueError("Unsupported result storage format")
                self.root()  # Validate the root before an HTTP response starts.
            else:
                for statement in (
                    "CREATE TABLE storage_format(version INTEGER NOT NULL)",
                    "INSERT INTO storage_format VALUES(1)",
                    "CREATE TABLE series(id INTEGER PRIMARY KEY, length INTEGER NOT NULL)",
                    "CREATE TABLE items(series_id INTEGER NOT NULL, position INTEGER NOT NULL, payload TEXT NOT NULL, "
                    "PRIMARY KEY(series_id,position))",
                    "CREATE TABLE refs(parent_series INTEGER NOT NULL, position INTEGER NOT NULL, child_series INTEGER NOT NULL, "
                    "PRIMARY KEY(parent_series,position,child_series))",
                    "CREATE TABLE root_refs(child_series INTEGER PRIMARY KEY)",
                    "CREATE TABLE root(slot INTEGER PRIMARY KEY CHECK(slot=1), payload TEXT NOT NULL)",
                ):
                    self._execute(statement)
                self._connection.commit()
            return self
        except BaseException as error:
            self._retain = True
            self._close_preserving(error)
            raise

    def _encode(self, value, active=None):
        if isinstance(value, DiskSeries):
            if value._storage is not self:
                raise ValueError("Result series belongs to another storage context")
            return ["series", value._series_id]
        if value is None or isinstance(value, (str, bool, int, float)):
            return ["scalar", value]
        if not isinstance(value, (dict, list, tuple)):
            raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ValueError("Circular reference detected")
        active.add(identity)
        try:
            if isinstance(value, dict):
                return ["object", [[self._encode(key, active), self._encode(item, active)]
                                    for key, item in value.items()]]
            return ["tuple" if isinstance(value, tuple) else "array",
                    [self._encode(item, active) for item in value]]
        finally:
            active.remove(identity)

    def _decode(self, value):
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("Malformed private result value")
        tag, payload = value
        if tag == "scalar":
            if payload is not None and not isinstance(payload, (str, bool, int, float)):
                raise ValueError("Malformed result scalar")
            return payload
        if tag == "series":
            if type(payload) is not int or payload < 1:
                raise ValueError("Malformed result series reference")
            return DiskSeries(self, payload)
        if tag == "object":
            return {self._decode(key): self._decode(item) for key, item in payload}
        if tag in {"array", "tuple"}:
            result = [self._decode(item) for item in payload]
            return tuple(result) if tag == "tuple" else result
        raise ValueError("Unknown private result tag")

    def _references(self, encoded):
        tag, payload = encoded
        if tag == "series":
            yield payload
        elif tag == "object":
            for key, value in payload:
                yield from self._references(key)
                yield from self._references(value)
        elif tag in {"array", "tuple"}:
            for value in payload:
                yield from self._references(value)

    def _begin(self):
        connection = self._require_open(write=True)
        if not connection.in_transaction:
            self._execute("BEGIN")
        return connection

    def _changed(self):
        self._pending += 1
        if self._pending >= 256:
            self._connection.commit()
            self._pending = 0

    def series(self, iterable=()):
        self._begin()
        series_id = self._execute("INSERT INTO series(length) VALUES(0)")
        self._changed()
        result = DiskSeries(self, series_id)
        iterator = iter(iterable)
        try:
            for value in iterator:
                result.append(value)
        except BaseException as error:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as cleanup_error:
                error.add_note(f"Result producer cleanup also failed: {cleanup_error}")
            raise
        else:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        return result

    def _write_item(self, series_id, position, value, *, append):
        encoded = self._encode(value)
        payload = json.dumps(encoded, separators=(",", ":"), allow_nan=True)
        self._begin()
        self._execute("SAVEPOINT result_item")
        try:
            self._execute("INSERT OR REPLACE INTO items VALUES(?,?,?)", (series_id, position, payload))
            self._execute("DELETE FROM refs WHERE parent_series=? AND position=?", (series_id, position))
            for child in self._references(encoded):
                self._execute("INSERT OR IGNORE INTO refs VALUES(?,?,?)", (series_id, position, child))
            if append:
                self._execute("UPDATE series SET length=length+1 WHERE id=?", (series_id,))
            self._execute("RELEASE result_item")
        except BaseException as error:
            try:
                self._execute("ROLLBACK TO result_item")
                self._execute("RELEASE result_item")
            except BaseException as cleanup_error:
                error.add_note(f"Result item rollback also failed: {cleanup_error}")
            raise
        self._changed()

    def read_json(self, path, *, parse_constant=None, indexed_root_objects=()):
        """Parse all syntax/EOF; large arrays spill to disk.

        Selected root object values become temporary indexed lookup maps. These
        maps support normalization, not publication as result-tree values.
        """
        self._require_open(write=True)
        if isinstance(indexed_root_objects, str):
            raise TypeError('Indexed root object names require an iterable of names')
        indexed_root_objects = tuple(indexed_root_objects)
        if any(not isinstance(key, str) for key in indexed_root_objects):
            raise TypeError('Indexed root object names must be strings')
        with Path(path).open("r", encoding="utf-8") as stream:
            return _JsonReader(self, stream, parse_constant, indexed_root_objects).read()

    def set_root(self, value):
        encoded = self._encode(value)
        payload = json.dumps(encoded, separators=(",", ":"), allow_nan=True)
        self._begin()
        self._execute("SAVEPOINT result_root")
        try:
            self._execute("INSERT OR REPLACE INTO root VALUES(1,?)", (payload,))
            self._execute("DELETE FROM root_refs")
            for child in self._references(encoded):
                self._execute("INSERT OR IGNORE INTO root_refs VALUES(?)", (child,))
            self._execute("RELEASE result_root")
        except BaseException as error:
            try:
                self._execute("ROLLBACK TO result_root")
                self._execute("RELEASE result_root")
            except BaseException as cleanup_error:
                error.add_note(f"Result root rollback also failed: {cleanup_error}")
            raise
        self._changed()

    def root(self):
        row = self._one("SELECT payload FROM root WHERE slot=1")
        if row is None:
            raise ValueError("Result storage has no completed root")
        return self._decode(json.loads(row[0]))

    def json_chunks(self, value):
        """Exact default json.dumps spacing/escaping, with strict finite floats."""
        encoder = json.JSONEncoder(allow_nan=False)
        active = set()

        def pieces(item):
            if not isinstance(item, (dict, list, tuple, DiskSeries)):
                yield encoder.encode(item)
                return
            identity = (id(item._storage), item._series_id) if isinstance(item, DiskSeries) else id(item)
            if identity in active:
                raise ValueError("Circular reference detected")
            active.add(identity)
            try:
                if isinstance(item, dict):
                    yield "{"
                    for index, (key, child) in enumerate(item.items()):
                        if index:
                            yield ", "
                        # Use the standard encoder's dictionary-key rules,
                        # including numeric/boolean/null keys and strict NaN.
                        yield encoder.encode({key: 0})[1:-4]
                        yield ": "
                        yield from pieces(child)
                    yield "}"
                else:
                    yield "["
                    for index, child in enumerate(item):
                        if index:
                            yield ", "
                        yield from pieces(child)
                    yield "]"
            finally:
                active.remove(identity)

        buffered, size = [], 0
        for piece in pieces(value):
            if buffered and size + len(piece) > 65536:
                yield "".join(buffered)
                buffered, size = [], 0
            if len(piece) > 65536:
                yield piece
            else:
                buffered.append(piece)
                size += len(piece)
        if buffered:
            yield "".join(buffered)

    def dump(self, value, stream):
        for chunk in self.json_chunks(value):
            stream.write(chunk)

    def _compact(self):
        self.root()
        self._execute("CREATE TEMP TABLE reachable(id INTEGER PRIMARY KEY)")
        self._execute("PRAGMA temp.cache_size=-2048")
        self._execute("INSERT INTO reachable WITH RECURSIVE walk(id) AS ("
                      "SELECT child_series FROM root_refs UNION "
                      "SELECT refs.child_series FROM refs JOIN walk ON refs.parent_series=walk.id) SELECT id FROM walk")
        self._execute("DELETE FROM items WHERE series_id NOT IN (SELECT id FROM reachable)")
        self._execute("DELETE FROM refs WHERE parent_series NOT IN (SELECT id FROM reachable)")
        self._execute("DELETE FROM series WHERE id NOT IN (SELECT id FROM reachable)")
        self._execute("DROP TABLE reachable")
        self._connection.commit()
        self._pending = 0
        self._execute("VACUUM")

    def publish(self, destination):
        self._require_open(write=True)
        destination = Path(destination).resolve()
        if not destination.is_relative_to(self.directory) or destination == self._path:
            raise ValueError("Result destination must be a fresh path within its directory")
        self._retain = True
        if destination.exists():
            raise FileExistsError(destination)
        self._compact()
        self.close()
        # Windows rename refuses an existing destination. Caller uses a unique
        # generation name; never replace an authoritative previous database.
        self._path.rename(destination)
        self._path = destination
        self._published = True
        return destination

    def close(self):
        """Idempotently release this context's connection, without deleting it."""
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                if not self._readonly:
                    connection.commit()
            except BaseException as error:
                try:
                    connection.close()
                except BaseException as cleanup_error:
                    error.add_note(f"Result connection close also failed: {cleanup_error}")
                raise
            else:
                connection.close()

    def _close_preserving(self, error):
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f"Result database cleanup also failed: {cleanup_error}")

    def _remove_scratch(self):
        if self._scratch_path is None:
            return
        for suffix in ("-journal", "-wal", "-shm", ""):
            path = Path(str(self._scratch_path) + suffix)
            if path.resolve().parent != self.directory:
                raise ValueError("Owned result cleanup escaped its directory")
            path.unlink(missing_ok=True)

    def __exit__(self, error_type, error, _traceback):
        if error_type is not None:
            self._retain = True
            self._close_preserving(error)
            return False
        self.close()
        if not self._readonly and not self._published and not self._retain:
            self._remove_scratch()
        return False


class IndexedJsonObject(Mapping):
    """Transient, ordered JSON-object lookup backed by this parser's owner.

    Repeated JSON keys replace their value without changing first insertion
    order. Source objects are normalized before result publication; the private
    stored-result format and default parser behavior remain unchanged.
    """
    def __init__(self, storage):
        self._storage = storage
        storage._execute('PRAGMA temp.cache_size=-2048')
        storage._execute('CREATE TEMP TABLE IF NOT EXISTS json_object_keys('
                         'series_id INTEGER,key TEXT COLLATE BINARY,position INTEGER,'
                         'PRIMARY KEY(series_id,key),UNIQUE(series_id,position))')
        self._values = storage.series()

    def _set(self, key, value):
        row = self._storage._one('SELECT position FROM json_object_keys WHERE series_id=? AND key=?',
                                 (self._values._series_id, key))
        if row is not None:
            self._values[row[0]] = value
            return
        position = len(self._values)
        self._values.append(value)
        self._storage._begin()
        self._storage._execute('INSERT INTO json_object_keys VALUES(?,?,?)',
                               (self._values._series_id, key, position))
        self._storage._changed()

    def __len__(self):
        return len(self._values)

    def __getitem__(self, key):
        hash(key)
        if not isinstance(key, str):
            raise KeyError(key)
        row = self._storage._one('SELECT position FROM json_object_keys WHERE series_id=? AND key=?',
                                 (self._values._series_id, key))
        if row is None:
            raise KeyError(key)
        return self._values[row[0]]

    def __iter__(self):
        for position in range(len(self)):
            yield self._storage._one('SELECT key FROM json_object_keys WHERE series_id=? AND position=?',
                                     (self._values._series_id, position))[0]

    def items(self):
        for position, key in enumerate(self):
            yield key, self._values[position]


class _JsonReader:
    """Small grammar around the stdlib scalar decoder; never buffer an array."""

    _special_string = re.compile(r'["\\\x00-\x1f]')

    def __init__(self, storage, stream, parse_constant, indexed_root_objects=()):
        self.storage, self.stream = storage, stream
        # JSONDecoder itself uses truthiness for its callback. Wrap explicitly
        # supplied callable objects so even a falsy strict validator is called.
        constant = (lambda token: parse_constant(token)) if parse_constant is not None else (lambda _: None)
        self.decoder = json.JSONDecoder(parse_constant=constant)
        self.buffer, self.position, self.eof = "", 0, False
        self.indexed_root_objects = frozenset(indexed_root_objects)

    def _fill(self):
        if self.eof:
            return False
        self.buffer = self.stream.read(65536)
        self.position = 0
        self.eof = not self.buffer
        return not self.eof

    def _peek(self):
        if self.position >= len(self.buffer) and not self._fill():
            return ""
        return self.buffer[self.position]

    def _space(self):
        while self._peek() and self.buffer[self.position] in " \t\r\n":
            self.position += 1

    def _string(self):
        fragments, start = [], self.position
        self.position += 1
        escaped = False
        while True:
            if escaped:
                if not self._peek():
                    raise ValueError("Unterminated JSON string")
                self.position += 1
                escaped = False
            match = self._special_string.search(self.buffer, self.position)
            if match is None:
                fragments.append(self.buffer[start:])
                if not self._fill():
                    raise ValueError("Unterminated JSON string")
                start = 0
                continue
            self.position = match.end()
            character = match.group()
            if character == '"':
                fragments.append(self.buffer[start:self.position])
                return self.decoder.decode("".join(fragments))
            if character != "\\":
                raise ValueError("Invalid control character in JSON string")
            if self.position == len(self.buffer):
                fragments.append(self.buffer[start:])
                if not self._fill():
                    raise ValueError("Unterminated JSON string")
                start, escaped = 0, True
            else:
                self.position += 1

    def _scalar(self):
        fragments = []
        while True:
            start = self.position
            while self.position < len(self.buffer) and self.buffer[self.position] not in " \t\r\n,]}":
                self.position += 1
            fragments.append(self.buffer[start:self.position])
            if self.position < len(self.buffer) or not self._fill():
                break
        return self.decoder.decode("".join(fragments))

    def _value(self, *, root=False, indexed_object=False):
        self._space()
        character = self._peek()
        if character == '"':
            return self._string()
        if character == "[":
            self.position += 1
            self._space()
            values, disk = [], None
            if self._peek() == "]":
                self.position += 1
                return values
            while True:
                value = self._value()
                if disk is not None:
                    disk.append(value)
                else:
                    values.append(value)
                    if len(values) > 64:
                        disk = self.storage.series(values)
                        values = []
                self._space()
                separator = self._peek()
                if separator == "]":
                    self.position += 1
                    return disk if disk is not None else values
                if separator != ",":
                    raise ValueError("Expected a JSON array separator")
                self.position += 1
        if character == "{":
            self.position += 1
            self._space()
            value = IndexedJsonObject(self.storage) if indexed_object else {}
            if self._peek() == "}":
                self.position += 1
                return value
            while True:
                self._space()
                if self._peek() != '"':
                    raise ValueError("Expected a JSON object key")
                key = self._string()
                self._space()
                if self._peek() != ":":
                    raise ValueError("Expected a JSON object colon")
                self.position += 1
                item = self._value(indexed_object=root and key in self.indexed_root_objects)
                if isinstance(value, IndexedJsonObject):
                    value._set(key, item)
                else:
                    value[key] = item
                self._space()
                separator = self._peek()
                if separator == "}":
                    self.position += 1
                    return value
                if separator != ",":
                    raise ValueError("Expected a JSON object separator")
                self.position += 1
        if not character:
            raise ValueError("Expected a JSON value")
        return self._scalar()

    def read(self):
        value = self._value(root=True)
        self._space()
        if self._peek():
            raise ValueError("Trailing data after JSON value")
        return value
