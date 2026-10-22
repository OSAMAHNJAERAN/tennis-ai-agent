"""Retained raw-frame commits, not a complete pipeline-resume checkpoint.

The caller must separately seal the normalized source, completed player pass,
selected models and runtime. This journal records only complete raw ball-model
frames. Filters must use another mutable store; accepted rows are immutable.
"""
from collections.abc import Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import re
import sqlite3
import stat

import numpy as np

from src.utils.ball_candidate_sequence import _decode_frame, _encode_frame


JOURNAL_NAME = '.ball_inference.sqlite3'
SCHEMA_VERSION = 1
APPLICATION_ID = 0x42494A31
MAX_IDENTITY_BYTES = 64 * 1024
MAX_FRAME_BYTES = 16 * 1024 * 1024
MAX_CANDIDATES_PER_FRAME = 65536
_MAX_SQLITE_INTEGER = (1 << 63) - 1
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')
_METADATA_SQL = ('CREATE TABLE journal_metadata ('
    'id INTEGER PRIMARY KEY CHECK(id=1), '
    'schema_version INTEGER NOT NULL CHECK(schema_version=1), '
    'identity TEXT NOT NULL, total_frames INTEGER NOT NULL CHECK(total_frames>0), '
    'next_frame INTEGER NOT NULL CHECK(next_frame>=0 AND next_frame<=total_frames))')
_FRAMES_SQL = ('CREATE TABLE raw_frames ('
    'frame_index INTEGER PRIMARY KEY CHECK(frame_index>=0), '
    'payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL CHECK(length(payload_sha256)=64))')


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be an integer frame boundary')
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(name + ' must be an integer frame boundary') from error
    if not (1 if positive else 0) <= value <= _MAX_SQLITE_INTEGER:
        raise ValueError(name + ' is outside the supported frame range')
    return value


def _plain_json(value, active=None):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Journal identity requires finite JSON numbers')
        return
    if type(value) not in (dict, list):
        raise ValueError('Journal identity requires plain JSON values')
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('Journal identity must not be circular')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError('Journal identity requires string dictionary keys')
            values = value.values()
        else:
            values = value
        for item in values:
            _plain_json(item, active)
    finally:
        active.remove(id(value))


def _identity(value):
    if type(value) is not dict or not value:
        raise ValueError('Journal identity must be a nonempty plain JSON object')
    try:
        _plain_json(value)
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False)
    except (RecursionError, OverflowError) as error:
        raise ValueError('Journal identity exceeds supported JSON nesting or numbers') from error
    if len(encoded) > MAX_IDENTITY_BYTES:
        raise ValueError('Journal identity exceeds its size bound')
    return encoded


def _redirected(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _directory(value):
    path = Path(value)
    if '..' in path.parts:
        raise ValueError('Journal directory must not contain parent traversal')
    path = path.absolute()
    for part in (path, *path.parents):
        if _redirected(part):
            raise ValueError('Journal paths must not use symbolic links or junctions')
        try:
            details = part.lstat()
        except FileNotFoundError:
            continue
        if getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0):
            raise ValueError('Journal paths must not traverse reparse points')
    if not path.is_dir():
        raise ValueError('Journal directory does not exist')
    return path.resolve()


def _regular_file(path, *, required):
    if _redirected(path):
        raise ValueError('Journal database and sidecars must not be symbolic links or junctions')
    try:
        details = path.lstat()
    except FileNotFoundError:
        if required:
            raise
        return None
    if (not stat.S_ISREG(details.st_mode)
            or getattr(details, 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)
            or details.st_nlink != 1):
        raise ValueError('Journal database and sidecars require unaliased regular files')
    return details


def _raw_hash(frame_index, payload):
    return hashlib.sha256(str(frame_index).encode('ascii') + b'\0' + payload.encode('utf-8')).hexdigest()


@contextmanager
def _cursor(cursor):
    try:
        yield cursor
    except BaseException as error:
        try:
            cursor.close()
        except BaseException as cleanup_error:
            error.add_note('Journal cursor cleanup also failed: ' + str(cleanup_error))
        raise
    else:
        cursor.close()


def _payload(candidates):
    iterator = iter(candidates)

    def bounded():
        for count, candidate in enumerate(iterator):
            if count >= MAX_CANDIDATES_PER_FRAME:
                raise ValueError('Raw candidate frame exceeds its candidate bound')
            yield candidate

    try:
        encoded = _encode_frame(bounded())
        if len(encoded.encode('utf-8')) > MAX_FRAME_BYTES:
            raise ValueError('Raw candidate frame exceeds its payload bound')
    except BaseException as error:
        try:
            close = getattr(iterator, 'close', None)
            if close is not None:
                close()
        except BaseException as cleanup_error:
            error.add_note('Raw candidate producer cleanup also failed: ' + str(cleanup_error))
        raise
    else:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
        return encoded


class BallInferenceJournal(Sequence):
    """Reopenable append-only raw candidates with one atomic frame frontier.

    Use ``with BallInferenceJournal.create(directory, identity, total_frames)``
    or ``.open(...)``. ``append(global_frame_index, complete_candidates)``
    commits the raw row and next-frame metadata together under synchronous
    FULL. Empty candidate sets are accepted frames. Any transaction failure
    poisons this owner; close and reopen to resolve an uncertain commit.

    At most one raw frame is decoded per read. SQLite has a 2 MiB page cache,
    file-backed temporary storage and no mmap. Explicit slices materialize the
    requested slice. Context exit rolls back uncommitted work and retains the
    database on success and failure. No journal or sidecar is deleted here.
    """

    def __init__(self, directory, identity, total_frames, *, create):
        self._identity = _identity(identity)
        self._total_frames = _integer(total_frames, 'Declared frame count', positive=True)
        self._directory = _directory(directory)
        self._path = self._directory / JOURNAL_NAME
        self._create = create
        self._connection = None
        self._file_id = None
        self._schema_version = None
        self._next_frame = 0
        self._entered = self._poisoned = False

    @classmethod
    def create(cls, directory, identity, total_frames):
        return cls(directory, identity, total_frames, create=True)

    @classmethod
    def open(cls, directory, identity, total_frames):
        return cls(directory, identity, total_frames, create=False)

    @property
    def identity(self):
        return json.loads(self._identity)

    @property
    def directory(self):
        return self._directory

    @property
    def path(self):
        return self._path
    @property
    def total_frames(self):
        return self._total_frames

    @property
    def next_frame(self):
        self._require_open()
        return self._next_frame

    def _safe_paths(self, *, required):
        if self.path != self.directory / JOURNAL_NAME:
            raise ValueError('Journal path must use its fixed owned filename')
        if _directory(self.directory) != self.directory:
            raise ValueError('Journal directory changed')
        details = _regular_file(self.path, required=required)
        if details is not None and self._file_id is not None:
            if (details.st_dev, details.st_ino) != self._file_id:
                raise ValueError('Journal database identity changed')
        for suffix in ('-journal', '-wal', '-shm'):
            sidecar = Path(str(self.path) + suffix)
            current = _regular_file(sidecar, required=False)
            if self._create and self._file_id is None and current is not None:
                raise FileExistsError('Refusing an existing journal sidecar')
        return details

    def _require_open(self):
        if self._poisoned:
            raise RuntimeError('Journal owner is unusable after failure; close and reopen')
        if self._connection is None:
            raise RuntimeError('Ball inference journal is not open')
        return self._connection

    def _execute(self, sql, values=(), *, fetch=False):
        with _cursor(self._require_open().execute(sql, values)) as cursor:
            return cursor.fetchone() if fetch else cursor.rowcount

    def __enter__(self):
        if self._entered:
            raise RuntimeError('Journal context cannot be reused; create another owner')
        self._entered = True
        try:
            details = self._safe_paths(required=not self._create)
            if self._create:
                flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
                descriptor = os.open(self.path, flags, 0o600)
                try:
                    details = os.fstat(descriptor)
                    os.fsync(descriptor)
                except BaseException as error:
                    try:
                        os.close(descriptor)
                    except BaseException as cleanup_error:
                        error.add_note('Journal creation descriptor cleanup also failed: ' + str(cleanup_error))
                    raise
                else:
                    os.close(descriptor)
            self._file_id = (details.st_dev, details.st_ino)
            self._connection = sqlite3.connect(self.path.as_uri() + '?mode=rw',
                                              uri=True, timeout=5., isolation_level=None)
            self._safe_paths(required=True)
            self._execute('PRAGMA trusted_schema=OFF')
            self._execute('PRAGMA cache_size=-2048')
            self._execute('PRAGMA temp_store=FILE')
            self._execute('PRAGMA mmap_size=0', fetch=True)
            self._execute('PRAGMA synchronous=FULL')
            if self._execute('PRAGMA journal_mode=DELETE', fetch=True) != ('delete',):
                raise ValueError('Journal requires SQLite DELETE journaling')
            if self._create:
                self._execute('BEGIN IMMEDIATE')
                self._execute(_METADATA_SQL)
                self._execute(_FRAMES_SQL)
                self._execute('PRAGMA application_id=' + str(APPLICATION_ID))
                self._execute('PRAGMA user_version=' + str(SCHEMA_VERSION))
                self._execute('INSERT INTO journal_metadata VALUES(1,?,?,?,0)',
                              (SCHEMA_VERSION, self._identity, self._total_frames))
                self._execute('COMMIT')
            self._execute('BEGIN')
            next_frame = self._validate()
            schema_version = self._execute('PRAGMA schema_version', fetch=True)
            if (type(schema_version) is not tuple or len(schema_version) != 1
                    or type(schema_version[0]) is not int):
                raise ValueError('Malformed journal schema version')
            self._execute('COMMIT')
            self._next_frame = next_frame
            self._schema_version = schema_version[0]
            return self
        except BaseException as error:
            self._poisoned = True
            self._dispose(error)
            raise

    def _validate(self):
        if self._execute('PRAGMA quick_check(1)', fetch=True) != ('ok',):
            raise ValueError('Malformed ball inference journal')
        if (self._execute('PRAGMA application_id', fetch=True) != (APPLICATION_ID,)
                or self._execute('PRAGMA user_version', fetch=True) != (SCHEMA_VERSION,)):
            raise ValueError('Unknown ball inference journal version')
        public_objects = "lower(substr(name,1,7)) != 'sqlite_'"
        count = self._execute('SELECT count(*) FROM sqlite_master WHERE ' + public_objects, fetch=True)
        expected = [('table', 'journal_metadata', 'journal_metadata', _METADATA_SQL),
                    ('table', 'raw_frames', 'raw_frames', _FRAMES_SQL)]
        if count != (len(expected),):
            raise ValueError('Unexpected ball inference journal schema')
        with _cursor(self._require_open().execute(
                'SELECT type,name,tbl_name,sql FROM sqlite_master WHERE ' + public_objects + ' ORDER BY name')) as cursor:
            for row in expected:
                if cursor.fetchone() != row:
                    raise ValueError('Unexpected ball inference journal schema')
            if cursor.fetchone() is not None:
                raise ValueError('Unexpected ball inference journal schema')
        if self._execute('SELECT count(*) FROM journal_metadata', fetch=True) != (1,):
            raise ValueError('Malformed journal metadata')
        # Bound identity text in SQL before returning it to Python.
        metadata = self._execute(
            "SELECT schema_version,identity,total_frames,next_frame FROM journal_metadata WHERE id=1 "
            "AND typeof(identity)='text' AND length(CAST(identity AS BLOB))<=?",
            (MAX_IDENTITY_BYTES,), fetch=True)
        if metadata is None or type(metadata[0]) is not int or metadata[0] != SCHEMA_VERSION:
            raise ValueError('Malformed journal metadata')
        if metadata[1] != self._identity:
            raise ValueError('Journal identity differs from the admitted identity')
        if type(metadata[2]) is not int or metadata[2] != self._total_frames:
            raise ValueError('Journal declared frame count changed')
        frontier = metadata[3]
        if type(frontier) is not int or not 0 <= frontier <= self._total_frames:
            raise ValueError('Malformed journal frontier')
        rows = self._execute('SELECT count(*),min(frame_index),max(frame_index) FROM raw_frames', fetch=True)
        if rows != ((frontier, 0, frontier - 1) if frontier else (0, None, None)):
            raise ValueError('Journal rows and frontier are not contiguous')
        return frontier

    def append(self, frame_index, candidates):
        self._require_open()
        frame_index = _integer(frame_index, 'Global frame index')
        if frame_index != self._next_frame or frame_index >= self._total_frames:
            raise ValueError('Only the exact next global raw frame can be committed')
        payload = _payload(candidates)
        try:
            self._safe_paths(required=True)
            self._execute('BEGIN IMMEDIATE')
            # The write lock keeps this schema check and following INSERT in
            # one admission frontier. Data-only commits do not change the
            # schema cookie; new triggers must never execute on accepted rows.
            if self._execute('PRAGMA schema_version', fetch=True) != (self._schema_version,):
                raise ValueError('Journal schema changed; reopen before another append')
            metadata = self._execute(
                "SELECT schema_version,identity,total_frames,next_frame FROM journal_metadata WHERE id=1 "
                "AND typeof(identity)='text' AND length(CAST(identity AS BLOB))<=?",
                (MAX_IDENTITY_BYTES,), fetch=True)
            if metadata != (SCHEMA_VERSION, self._identity, self._total_frames, self._next_frame):
                raise ValueError('Journal identity or frontier changed; reopen before another append')
            self._execute('INSERT INTO raw_frames(frame_index,payload,payload_sha256) VALUES(?,?,?)',
                          (frame_index, payload, _raw_hash(frame_index, payload)))
            changed = self._execute('UPDATE journal_metadata SET next_frame=? WHERE id=1 AND next_frame=?',
                                    (frame_index + 1, frame_index))
            if changed != 1:
                raise ValueError('Journal frontier did not advance atomically')
            self._execute('COMMIT')
        except BaseException as error:
            # COMMIT can succeed before its acknowledgement is interrupted.
            # Never reuse this owner's cached frontier, even after rollback.
            self._poisoned = True
            self._dispose(error)
            raise
        self._next_frame = frame_index + 1

    def __len__(self):
        return self.next_frame

    def __getitem__(self, index):
        self._require_open()
        if isinstance(index, slice):
            if any(isinstance(value, (bool, np.bool_)) for value in (index.start, index.stop, index.step)):
                raise ValueError('Raw frame slices must not use Boolean boundaries')
            return [self[position] for position in range(*index.indices(self._next_frame))]
        index = _integer(index, 'Raw frame index')
        if index >= self._next_frame:
            raise IndexError(index)
        try:
            self._safe_paths(required=True)
            # These predicates prevent an oversized/corrupt text value from
            # being fetched into Python before its size/type is checked.
            row = self._execute(
                "SELECT payload,payload_sha256 FROM raw_frames WHERE frame_index=? "
                "AND typeof(payload)='text' AND length(CAST(payload AS BLOB))<=? "
                "AND typeof(payload_sha256)='text' AND length(CAST(payload_sha256 AS BLOB))=64",
                (index, MAX_FRAME_BYTES), fetch=True)
            if row is None or _DIGEST.fullmatch(row[1]) is None or _raw_hash(index, row[0]) != row[1]:
                raise ValueError('Raw journal frame is missing, oversized or corrupt')
            decoded = _decode_frame(row[0])
            if len(decoded) > MAX_CANDIDATES_PER_FRAME:
                raise ValueError('Raw journal frame exceeds its candidate bound')
            if _encode_frame(decoded) != row[0]:
                raise ValueError('Raw journal frame differs from its canonical candidate codec')
            return decoded
        except BaseException:
            self._poisoned = True
            raise

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _dispose(self, primary=None):
        connection, self._connection = self._connection, None
        failures = []
        if connection is not None:
            try:
                if connection.in_transaction:
                    connection.rollback()
            except BaseException as error:
                failures.append(('rollback', error))
            finally:
                try:
                    connection.close()
                except BaseException as error:
                    failures.append(('close', error))
        if primary is None and failures:
            _, primary = failures.pop(0)
            for stage, error in failures:
                primary.add_note('Journal ' + stage + ' also failed: ' + str(error))
            raise primary
        if primary is not None:
            for stage, error in failures:
                primary.add_note('Journal ' + stage + ' also failed: ' + str(error))

    def close(self):
        self._dispose()

    def __exit__(self, error_type, error, _traceback):
        self._dispose(error)
        return False
