"""Atomic JSON artifacts whose frame records are serialized one at a time."""

import json
import os
from pathlib import Path
import tempfile


class JsonArrayArtifact:
    """Append an array to a small metadata object, publishing on clean exit.

    Only the current record and the text encoder's chunks are retained. The
    same-directory temporary file remains private until every record has been
    written and the context exits successfully. Errors preserve any previous
    completed destination and close/remove this writer's temporary file.
    """

    def __init__(self, destination, metadata, array_key="frames", *, allow_nan=True):
        if not isinstance(array_key, str) or array_key in metadata:
            raise ValueError("Array key must be a string absent from metadata")
        self.destination = Path(destination)
        self.metadata = metadata
        self.array_key = array_key
        self.allow_nan = allow_nan
        self._stream = None
        self._temporary = None
        self._count = 0

    def __enter__(self):
        if self._stream is not None:
            raise RuntimeError("JSON array writer cannot be reused")
        try:
            self._stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", delete=False,
                dir=self.destination.parent, prefix=f".{self.destination.name}.", suffix=".tmp",
            )
            self._temporary = Path(self._stream.name)
            self._stream.write("{\n")
            for key, value in self.metadata.items():
                json.dump(key, self._stream)
                self._stream.write(": ")
                json.dump(value, self._stream, allow_nan=self.allow_nan)
                self._stream.write(",\n")
            json.dump(self.array_key, self._stream)
            self._stream.write(": [\n")
            return self
        except BaseException as error:
            self._discard_preserving(error)
            raise

    def append(self, record):
        if self._stream is None or self._stream.closed:
            raise RuntimeError("JSON array writer is not open")
        try:
            if self._count:
                self._stream.write(",\n")
            json.dump(record, self._stream, allow_nan=self.allow_nan, separators=(",", ":"))
            self._count += 1
        except BaseException as error:
            self._discard_preserving(error)
            raise

    def _discard_preserving(self, error):
        try:
            try:
                if self._stream is not None and not self._stream.closed:
                    self._stream.close()
            finally:
                if self._temporary is not None:
                    self._temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            error.add_note(f"JSON temporary-file cleanup also failed: {cleanup_error}")

    def __exit__(self, error_type, error, _traceback):
        if error_type is not None:
            self._discard_preserving(error)
            return False
        try:
            if self._stream is None or self._stream.closed:
                raise RuntimeError("JSON array writer closed before publication")
            self._stream.write("\n]}\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            os.replace(self._temporary, self.destination)
        except BaseException as publication_error:
            self._discard_preserving(publication_error)
            raise
        return False


def write_json_array_artifact(destination, metadata, array_key, records, *, allow_nan=True):
    """Consume a lazy record producer and close it if serialization fails."""
    iterator = iter(records)
    try:
        with JsonArrayArtifact(destination, metadata, array_key, allow_nan=allow_nan) as artifact:
            for record in iterator:
                artifact.append(record)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
