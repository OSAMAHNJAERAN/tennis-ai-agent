"""Bounded disk selection of the legacy offline near/far court roles.

This is the existing whole-segment presence prior, not causal match-player A/B
identity. Two candidate passes preserve ``CourtPlayerTracker.select`` decisions
while keeping frame observations, growing track counts and selected snapshots
on disk. Source proposal indices remain available for exact person annotations.
"""

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
import math
import operator
from pathlib import Path
import sqlite3
import tempfile

from src.court.court_geometry import TennisCourtGeometry as Court
from src.utils.player_detection_sequence import PlayerDetectionSequence


class _SelectionLedger:
    """Owned presence counts and aligned proposal indices, without a row cache."""

    def __init__(self, directory, transaction_size=256):
        self.directory = Path(directory).resolve()
        self.transaction_size = transaction_size
        self.path = self._connection = None
        self._length = self._pending = 0

    def __enter__(self):
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix="court_player_selection_", suffix=".sqlite3", delete=False,
            ) as temporary:
                self.path = Path(temporary.name).resolve()
            if self.path.parent != self.directory:
                raise ValueError("Owned selection database escaped the output directory")
            self._connection = sqlite3.connect(self.path)
            self._connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            self._connection.execute("PRAGMA cache_size=-2048")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._connection.execute("PRAGMA mmap_size=0").fetchone()
            self._connection.execute(
                "CREATE TABLE presence (role INTEGER NOT NULL, track_key TEXT NOT NULL, "
                "count INTEGER NOT NULL, PRIMARY KEY(role,track_key)) WITHOUT ROWID"
            )
            self._connection.execute(
                "CREATE TABLE selected_indices (frame_index INTEGER PRIMARY KEY, "
                "role_1 INTEGER, role_2 INTEGER)"
            )
            self._connection.commit()
            return self
        except BaseException as error:
            self._close_preserving(error)
            raise

    def _require_open(self):
        if self._connection is None:
            raise RuntimeError("Court player selection is not open")
        return self._connection

    def _mutated(self):
        self._pending += 1
        if self._pending >= self.transaction_size:
            self._require_open().commit()
            self._pending = 0

    def add_presence(self, role, track_id):
        cursor = self._require_open().execute(
            "INSERT INTO presence(role,track_key,count) VALUES(?,?,1) "
            "ON CONFLICT(role,track_key) DO UPDATE SET count=count+1", (role, _track_key(track_id)),
        )
        cursor.close()
        self._mutated()

    def presence(self, role, track_id):
        if track_id is None:
            return 0
        cursor = self._require_open().execute(
            "SELECT count FROM presence WHERE role=? AND track_key=?", (role, _track_key(track_id)),
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        return 0 if row is None else row[0]

    def append(self, role_1, role_2):
        cursor = self._require_open().execute(
            "INSERT INTO selected_indices(frame_index,role_1,role_2) VALUES(?,?,?)",
            (self._length, role_1, role_2),
        )
        cursor.close()
        self._length += 1
        self._mutated()

    def indices(self, frame_index):
        frame_index = operator.index(frame_index)
        if frame_index < 0:
            frame_index += self._length
        if not 0 <= frame_index < self._length:
            raise IndexError(frame_index)
        cursor = self._require_open().execute(
            "SELECT role_1,role_2 FROM selected_indices WHERE frame_index=?", (frame_index,),
        )
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        if row is None:
            raise RuntimeError(f"Selection frame {frame_index} is missing from owned storage")
        return {1: row[0], 2: row[1]}

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
            error.add_note(f"Selection database cleanup also failed: {close_error}")

    def __exit__(self, error_type, error, _traceback):
        if error_type is not None:
            self._close_preserving(error)
            return False
        try:
            self._require_open().commit()
            self._connection.close()
            self._connection = None
            for suffix in ("-journal", "-wal", "-shm", ""):
                owned = Path(str(self.path) + suffix)
                if owned.resolve().parent != self.directory:
                    raise ValueError("Owned selection cleanup escaped the output directory")
                owned.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            self._close_preserving(cleanup_error)
            raise
        return False


def _track_key(track_id):
    # Actual detector IDs are integral. TEXT preserves integers wider than
    # SQLite INTEGER, and merges equal Python/NumPy integral scalar keys.
    return str(operator.index(track_id))


class _SelectedIndices(Sequence):
    def __init__(self, ledger):
        self._ledger = ledger

    def __len__(self):
        return self._ledger._length

    def __getitem__(self, frame_index):
        if isinstance(frame_index, slice):
            return [self[index] for index in range(*frame_index.indices(len(self)))]
        return self._ledger.indices(frame_index)

    def __iter__(self):
        for frame_index in range(len(self)):
            yield self[frame_index]


class _SelectedRole(Sequence):
    def __init__(self, selection, role):
        self._selection, self._role = selection, role

    def __len__(self):
        return len(self._selection._selected)

    def __getitem__(self, frame_index):
        if isinstance(frame_index, slice):
            return [self[index] for index in range(*frame_index.indices(len(self)))]
        indices = self._selection.selected_detection_indices[frame_index]
        if indices[self._role] is None:
            return None
        # Snapshot rows contain only actual selected boxes, in role order.
        offset = int(self._role == 2 and indices[1] is not None)
        return self._selection._selected[frame_index][offset]

    def __iter__(self):
        for frame_index in range(len(self)):
            yield self[frame_index]


class DiskCourtPlayerSelection(Mapping):
    """Read-only role sequences and exact source indices, valid in the context.

    Reads return independent BBox/PoseBBox snapshots. Mutating a read cannot
    change persisted selection. Slices explicitly materialize the requested
    slice; ordinary iteration/indexing retains only the current record.
    """

    def __init__(self, selected, ledger, input_path=None):
        self._selected, self._ledger = selected, ledger
        self.selected_detection_indices = _SelectedIndices(ledger)
        self._roles = {role: _SelectedRole(self, role) for role in (1, 2)}
        self.storage_paths = tuple(path for path in (input_path, selected.path, ledger.path) if path is not None)

    def __len__(self):
        return 2

    def __iter__(self):
        return iter((1, 2))

    def __getitem__(self, role):
        return self._roles[role]

    def index(self, role, frame_index):
        if role not in (1, 2):
            raise KeyError(role)
        return self.selected_detection_indices[frame_index][role]


def _candidates(tracker, boxes, calibration):
    # Keep the exact source position, including duplicate IDs and equal boxes.
    return [(source_index, value) for source_index, box in enumerate(boxes)
            if (value := tracker._candidate(box, calibration)) is not None]


def _select(tracker, detections, fps, calibrations, selected, ledger):
    count = len(detections)
    if calibrations is not None and len(calibrations) != count:
        raise ValueError("Per-frame calibrations must align with detections")
    valid_geometry = (tracker.calibration.is_valid if calibrations is None
                      else any(calibration.is_valid for calibration in calibrations))
    if not valid_geometry:
        for _ in range(count):
            selected.append([])
            ledger.append(None, None)
        return

    # Whole-segment presence includes every candidate occurrence, including
    # duplicates. Its cardinality grows on disk rather than in a Counter.
    for index, boxes in enumerate(detections):
        calibration = tracker.calibration if calibrations is None else calibrations[index]
        for _, (role, _, box) in _candidates(tracker, boxes, calibration):
            if box.track_id is not None:
                ledger.add_presence(role, box.track_id)

    memory = {1: None, 2: None}
    for index, boxes in enumerate(detections):
        calibration = tracker.calibration if calibrations is None else calibrations[index]
        if not calibration.is_valid:
            memory = {1: None, 2: None}
            selected.append([])
            ledger.append(None, None)
            continue
        frame = _candidates(tracker, boxes, calibration)
        used_ids, accepted, indices = set(), [], {1: None, 2: None}
        for role in (1, 2):
            previous = memory[role]
            if previous is not None and (index - previous[0]) / fps > tracker.memory_seconds:
                previous = memory[role] = None
            best = None
            for source_index, (candidate_role, point, box) in frame:
                if candidate_role != role or (box.track_id is not None and box.track_id in used_ids):
                    continue
                if previous is not None:
                    last_frame, last_point, last_id = previous
                    distance = math.dist(point, last_point)
                    allowed = tracker.localization_slack_m + tracker.max_speed_mps * (index - last_frame) / fps
                    if distance > allowed:
                        continue
                    score = distance - (.5 if box.track_id is not None and box.track_id == last_id else 0.)
                else:
                    baseline_y = Court.COURT_LENGTH if role == 1 else 0.
                    court_cost = abs(point[0] - Court.COURT_WIDTH_DOUBLES / 2) + .15 * abs(point[1] - baseline_y)
                    score = court_cost - 3 * ledger.presence(role, box.track_id) / max(1, count)
                score -= .2 * float(box.confidence)
                # Strict comparison preserves min(..., key=score)'s first tie.
                if best is None or score < best[0]:
                    best = score, source_index, point, box
            if best is not None:
                _, source_index, point, box = best
                accepted.append(box)
                indices[role] = source_index
                memory[role] = (index, point, box.track_id)
                if box.track_id is not None:
                    used_ids.add(box.track_id)
        selected.append(accepted)
        ledger.append(indices[1], indices[2])


@contextmanager
def select_court_players_disk(tracker, detections, fps, directory, calibrations=None):
    """Select exact legacy near/far choices with bounded resident frame storage.

    ``detections`` may be an open PlayerDetectionSequence or a one-shot iterable.
    Other inputs are copied to owned typed disk snapshots before the two
    candidate passes. Per-frame calibrations must be a replayable aligned
    indexed sequence. The output directory must already exist.

    On successful exit only owned scratch databases/sidecars are removed. Any
    producer, selection or consumer failure closes and retains accepted records
    for diagnosis. An existing input store remains owned by its caller.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite")
    with ExitStack() as stack:
        if isinstance(detections, PlayerDetectionSequence):
            source, input_path = detections, None
        else:
            source = stack.enter_context(PlayerDetectionSequence(directory))
            source.extend(detections)
            input_path = source.path
        ledger = stack.enter_context(_SelectionLedger(directory))
        selected = stack.enter_context(PlayerDetectionSequence(directory))
        _select(tracker, source, fps, calibrations, selected, ledger)
        yield DiskCourtPlayerSelection(selected, ledger, input_path)
