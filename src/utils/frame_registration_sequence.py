"""Owned indexed court registration records for multipass offline analysis."""

from collections.abc import Sequence
from dataclasses import dataclass
import json
import operator
from pathlib import Path
import sqlite3
import tempfile
from typing import Optional

import numpy as np

from src.court.calibration import CourtCalibration


@dataclass(frozen=True)
class FrameRegistration:
    """One frame's observed calibration and nullable anchor transform.

    The record is immutable. Dynamic storage reads produce independent array
    snapshots; mutating a snapshot never changes stored records. Callers treat
    calibration and transform matrices as read-only observations.
    """

    calibration: CourtCalibration
    anchor_to_frame_px: Optional[np.ndarray]


def _array_parts(array):
    if array is None:
        return None, None
    if not isinstance(array, np.ndarray):
        raise TypeError("Registration matrices must be NumPy arrays or None")
    # Raw numeric buffers have no executable Python object payload. dtype.str
    # fully describes these simple types, including precision and byte order.
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.kind not in "biufc":
        raise TypeError("Registration matrices require a simple numeric dtype")
    metadata = {"dtype": array.dtype.str, "shape": list(array.shape)}
    return metadata, array.tobytes(order="C")


def _array_from_parts(metadata, payload):
    if metadata is None:
        if payload is not None:
            raise ValueError("Null registration matrix has a buffer")
        return None
    if not isinstance(metadata, dict) or set(metadata) != {"dtype", "shape"}:
        raise ValueError("Malformed registration matrix metadata")
    dtype = np.dtype(metadata["dtype"])
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        raise ValueError("Unsafe registration matrix dtype")
    shape = metadata["shape"]
    if (not isinstance(shape, list) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                                          for value in shape)):
        raise ValueError("Malformed registration matrix shape")
    if payload is None:
        raise ValueError("Registration matrix is missing its buffer")
    expected_bytes = dtype.itemsize
    for dimension in shape:
        expected_bytes *= dimension
    if expected_bytes != len(payload):
        raise ValueError("Registration matrix buffer size differs from its shape")
    # Copy only this requested record's arrays. The underlying SQLite buffer is
    # never retained, and no precision/byte-order conversion is performed.
    return np.frombuffer(payload, dtype=dtype).reshape(tuple(shape)).copy(order="C")


def _encode_record(record):
    if not isinstance(record, FrameRegistration) or not isinstance(record.calibration, CourtCalibration):
        raise TypeError("Registration storage requires FrameRegistration/CourtCalibration records")
    calibration = record.calibration
    image_metadata, image_bytes = _array_parts(calibration.image_to_court)
    transform_metadata, transform_bytes = _array_parts(record.anchor_to_frame_px)
    metadata = {
        "image_to_court": image_metadata,
        "inlier_mask": calibration.inlier_mask,
        "reprojection_error_px": calibration.reprojection_error_px,
        "reprojection_error_m": calibration.reprojection_error_m,
        "inlier_reprojection_error_px": calibration.inlier_reprojection_error_px,
        "rejection_reason": calibration.rejection_reason,
        "anchor_to_frame_px": transform_metadata,
    }
    # Preserve raw fields rather than CourtCalibration.to_dict(): the public
    # dict hides a matrix when rejection_reason is present. JSON does not round
    # Python floats and retains nonfinite private diagnostic scalar values.
    return json.dumps(metadata, separators=(",", ":"), allow_nan=True), image_bytes, transform_bytes


def _decode_record(row):
    metadata = json.loads(row[0])
    calibration = CourtCalibration(
        image_to_court=_array_from_parts(metadata["image_to_court"], row[1]),
        inlier_mask=tuple(metadata["inlier_mask"]),
        reprojection_error_px=metadata["reprojection_error_px"],
        reprojection_error_m=metadata["reprojection_error_m"],
        inlier_reprojection_error_px=metadata["inlier_reprojection_error_px"],
        rejection_reason=metadata["rejection_reason"],
    )
    return FrameRegistration(calibration, _array_from_parts(metadata["anchor_to_frame_px"], row[2]))


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be an integer " + ("at least zero" if allow_zero else "greater than zero"))
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(name + " must be an integer " + ("at least zero" if allow_zero else "greater than zero")) from error
    if value < (0 if allow_zero else 1):
        raise ValueError(name + " must be an integer " + ("at least zero" if allow_zero else "greater than zero"))
    return value


class _RegistrationView(Sequence):
    """Aligned read-only view; never materialize the source into a full list."""

    def __init__(self, source, field):
        self._source, self._field = source, field

    def __len__(self):
        return len(self._source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if self._source._constant_calibration is not None:
            self._source._index(index)
            if self._field == "calibration":
                return self._source._constant_calibration
            return np.eye(3) if self._source._constant_calibration.is_valid else None
        return getattr(self._source[index], self._field)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class FrameRegistrationSequence(Sequence):
    """Append-only SQLite sequence or constant static-camera record view.

    Dynamic mode owns one exclusively created scratch database, with a 2 MiB
    page cache, no mmap, and bounded commit batches. Reads retain only their
    requested record. Successful exit closes/removes exact owned database and
    sidecars; failure closes and retains accepted rows for diagnosis.

    ``constant(calibration, frame_count)`` needs no directory or database. It
    shares the original immutable calibration reference and creates only the
    requested identity transform snapshot when that calibration is valid.
    """

    def __init__(self, directory, *, transaction_size=256):
        self.directory = Path(directory).resolve()
        self.transaction_size = _positive_integer(transaction_size, "Transaction size")
        self._path = self._connection = None
        self._length = self._pending_mutations = 0
        self._constant_calibration = None
        self.calibrations = _RegistrationView(self, "calibration")
        self.transforms = _RegistrationView(self, "anchor_to_frame_px")

    @classmethod
    def constant(cls, calibration, frame_count):
        if not isinstance(calibration, CourtCalibration):
            raise TypeError("Constant registration requires CourtCalibration")
        frame_count = _positive_integer(frame_count, "Frame count", allow_zero=True)
        sequence = cls.__new__(cls)
        sequence.directory = sequence.transaction_size = None
        sequence._path = sequence._connection = None
        sequence._length, sequence._pending_mutations = frame_count, 0
        sequence._constant_calibration = calibration
        sequence.calibrations = _RegistrationView(sequence, "calibration")
        sequence.transforms = _RegistrationView(sequence, "anchor_to_frame_px")
        return sequence

    @property
    def path(self):
        """The exact owned dynamic database path, retained after close."""
        return self._path

    def __len__(self):
        return self._length

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        return index

    def _require_open(self):
        if self._connection is None:
            raise RuntimeError("Frame registration storage is not open")
        return self._connection

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self._length))]
        index = self._index(index)
        if self._constant_calibration is not None:
            return FrameRegistration(self._constant_calibration,
                                     np.eye(3) if self._constant_calibration.is_valid else None)
        connection = self._require_open()
        cursor = connection.execute(
            "SELECT metadata,image_matrix,anchor_transform FROM registration_frames WHERE frame_index=?", (index,)
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError(f"Frame registration {index} is missing from owned storage")
        return _decode_record(row)

    def __iter__(self):
        for index in range(self._length):
            yield self[index]

    def __enter__(self):
        if self._constant_calibration is not None:
            return self
        if self._path is not None:
            raise RuntimeError("Frame registration storage cannot be reused")
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix="frame_registration_", suffix=".sqlite3", delete=False,
            ) as temporary:
                self._path = Path(temporary.name).resolve()
            if self._path.parent != self.directory:
                raise ValueError("Owned registration database escaped the output directory")
            self._connection = sqlite3.connect(self._path)
            self._connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            self._connection.execute("PRAGMA cache_size=-2048")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute("PRAGMA mmap_size=0").fetchone()
            self._connection.execute(
                "CREATE TABLE registration_frames (frame_index INTEGER PRIMARY KEY, metadata TEXT NOT NULL, "
                "image_matrix BLOB, anchor_transform BLOB)"
            )
            self._connection.commit()
            return self
        except BaseException as error:
            self._close_preserving(error)
            raise

    def append(self, record):
        if self._constant_calibration is not None:
            raise TypeError("Constant frame registration is read-only")
        connection = self._require_open()
        metadata, image_matrix, anchor_transform = _encode_record(record)
        cursor = connection.execute(
            "INSERT INTO registration_frames(frame_index,metadata,image_matrix,anchor_transform) VALUES(?,?,?,?)",
            (self._length, metadata, image_matrix, anchor_transform),
        )
        cursor.close()
        self._length += 1
        self._pending_mutations += 1
        if self._pending_mutations >= self.transaction_size:
            connection.commit()
            self._pending_mutations = 0

    def extend(self, records):
        iterator = iter(records)
        try:
            for record in iterator:
                self.append(record)
        except BaseException as error:
            try:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            except BaseException as close_error:
                error.add_note(f"Registration producer cleanup also failed: {close_error}")
            raise
        else:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

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
            error.add_note(f"Registration database cleanup also failed: {close_error}")

    def __exit__(self, error_type, error, _traceback):
        if self._constant_calibration is not None:
            return False
        if error_type is not None:
            self._close_preserving(error)
            return False
        try:
            self._require_open().commit()
            self._connection.close()
            self._connection = None
            for suffix in ("-journal", "-wal", "-shm", ""):
                owned = Path(str(self._path) + suffix)
                if owned.resolve().parent != self.directory:
                    raise ValueError("Owned registration cleanup escaped the output directory")
                owned.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            self._close_preserving(cleanup_error)
            raise
        return False
