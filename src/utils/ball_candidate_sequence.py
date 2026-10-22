"""Owned disk-backed candidate frames for multipass offline ball analysis."""

from collections.abc import Sequence
import json
import operator
from pathlib import Path
import sqlite3
import tempfile

from src.tracking.temporal_ball_tracker import BallObservation
from src.utils.bbox_utils import BBox


def _encode_frame(candidates):
    records = []
    for candidate in candidates:
        if not isinstance(candidate, BallObservation):
            raise TypeError("Candidate frames must contain BallObservation records")
        box = candidate.bbox
        records.append({
            "x_px": candidate.x_px, "y_px": candidate.y_px, "confidence": candidate.confidence,
            "bbox": None if box is None else {
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "confidence": box.confidence, "class_id": box.class_id, "track_id": box.track_id,
            },
        })
    # Nonfinite values are preserved for the existing downstream predicates.
    # This is private scratch JSON read only by json.loads, never executable data.
    return json.dumps(records, separators=(",", ":"), allow_nan=True)


def _decode_frame(payload):
    return [BallObservation(
        x_px=record["x_px"], y_px=record["y_px"], confidence=record["confidence"],
        bbox=BBox(**record["bbox"]) if record["bbox"] is not None else None,
    ) for record in json.loads(payload)]


class BallCandidateSequence(Sequence):
    """Mutable indexed frame storage with bounded SQLite cache/transactions.

    A read returns an independent snapshot. Persist changes with explicit
    indexed assignment; mutating a returned candidate or list does not write
    through. Iteration reads one frame at a time, including empty frames.

    Only this context's exclusively created database and known SQLite sidecars
    are removed on success. On an exception, accepted records are committed
    when possible and the closed partial database is retained for diagnosis.
    """

    def __init__(self, directory, *, transaction_size=256):
        if isinstance(transaction_size, bool):
            raise ValueError("Transaction size must be a positive integer")
        try:
            transaction_size = operator.index(transaction_size)
        except TypeError as error:
            raise ValueError("Transaction size must be a positive integer") from error
        if transaction_size < 1:
            raise ValueError("Transaction size must be a positive integer")
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self._path = None
        self._connection = None
        self._length = 0
        self._pending_mutations = 0

    @property
    def path(self):
        """The exclusive owned database path; retained after the context closes."""
        return self._path

    def __enter__(self):
        if self._path is not None:
            raise RuntimeError("Ball candidate storage cannot be reused")
        try:
            # The supplied output directory must already exist. No parent or
            # shared storage is created or changed by this context.
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix="ball_candidates_", suffix=".sqlite3", delete=False,
            ) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError("Owned candidate database escaped the output directory")
            self._connection = sqlite3.connect(self._path)
            self._connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            self._connection.execute("PRAGMA cache_size=-2048")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute("PRAGMA mmap_size=0").fetchone()
            self._connection.execute(
                "CREATE TABLE candidate_frames (frame_index INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
            )
            self._connection.commit()
            return self
        except BaseException as error:
            self._close_preserving(error)
            raise

    def _require_open(self):
        if self._connection is None:
            raise RuntimeError("Ball candidate storage is not open")
        return self._connection

    def __len__(self):
        return self._length

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return index

    def __getitem__(self, index):
        connection = self._require_open()
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._length))]
        index = self._index(index)
        cursor = connection.execute("SELECT payload FROM candidate_frames WHERE frame_index=?", (index,))
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError(f"Candidate frame {index} is missing from owned storage")
        return _decode_frame(row[0])

    def __iter__(self):
        # An explicit range avoids retaining a cursor or decoded frame cache
        # while a consumer replaces the current record during the same pass.
        for index in range(self._length):
            yield self[index]

    def _record_mutation(self):
        self._pending_mutations += 1
        if self._pending_mutations >= self.transaction_size:
            self._require_open().commit()
            self._pending_mutations = 0

    def append(self, candidates):
        connection = self._require_open()
        payload = _encode_frame(candidates)
        cursor = connection.execute(
            "INSERT INTO candidate_frames(frame_index,payload) VALUES(?,?)", (self._length, payload)
        )
        cursor.close()
        self._length += 1
        self._record_mutation()

    def extend(self, frames):
        iterator = iter(frames)
        try:
            for candidates in iterator:
                self.append(candidates)
        except BaseException as error:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as close_error:
                error.add_note(f"Candidate producer cleanup also failed: {close_error}")
            raise
        else:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def __setitem__(self, index, candidates):
        connection = self._require_open()
        index = self._index(index)
        payload = _encode_frame(candidates)
        cursor = connection.execute("UPDATE candidate_frames SET payload=? WHERE frame_index=?", (payload, index))
        try:
            if cursor.rowcount != 1:
                raise RuntimeError(f"Candidate frame {index} is missing from owned storage")
        finally:
            cursor.close()
        self._record_mutation()

    def _close_preserving(self, error):
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            try:
                connection.commit()
            finally:
                connection.close()
        except BaseException as close_error:
            error.add_note(f"Candidate database cleanup also failed: {close_error}")

    def __exit__(self, error_type, error, _traceback):
        if error_type is not None:
            self._close_preserving(error)
            return False
        try:
            self._require_open().commit()
            self._connection.close()
            self._connection = None
            # Exact sidecar names belong to this exclusive database, never a
            # glob or a path derived from another artifact's contents.
            for suffix in ("-journal", "-wal", "-shm", ""):
                owned = Path(str(self._path) + suffix)
                if owned.resolve().parent != self.directory:
                    raise ValueError("Owned candidate cleanup escaped the output directory")
                owned.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            self._close_preserving(cleanup_error)
            raise
        return False
