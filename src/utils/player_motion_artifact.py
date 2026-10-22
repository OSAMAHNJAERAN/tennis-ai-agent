"""Bounded candidate motion samples and atomic nested-array JSON export."""
from collections.abc import Sequence
import json
import operator
import os
from pathlib import Path
import tempfile

from src.tracking.streaming_player_motion import GroundMotionAccumulator


class UnknownGroundPositions(Sequence):
    """Unqualified geometry supplies unknowns without a match-long null list."""

    def __init__(self, length):
        if type(length) is not int or length < 0:
            raise ValueError('Frame length must be a nonnegative integer')
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [None for _ in range(*index.indices(self.length))]
        index = operator.index(index)
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(index)
        return None


class FrameTimes(UnknownGroundPositions):
    """Normalized replay clock; this alone cannot authorize physical motion."""

    def __init__(self, length, fps):
        import math
        super().__init__(length)
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Normalized FPS must be finite and positive')
        self.fps = fps

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self.length))]
        super().__getitem__(index)
        if index < 0:
            index += self.length
        return index / self.fps


class MotionSampleView(Sequence):
    def __init__(self, frames, identity):
        self.frames, self.identity = frames, identity

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        return self.frames[index][self.identity]


def accumulate_motion_frames(positions_1, positions_2, timestamps, destination, *, source_frames=None):
    """Append one pair at a time; source provenance never upgrades null geometry."""
    length = len(timestamps)
    if len(positions_1) != length or len(positions_2) != length or (source_frames is not None and len(source_frames) != length):
        raise ValueError('Motion positions and source timing must align')
    accumulators = {1: GroundMotionAccumulator(), 2: GroundMotionAccumulator()}
    for index in range(length):
        interval = source_frames[index][0] if source_frames is not None else None
        pair = {}
        for identity, positions in ((1, positions_1), (2, positions_2)):
            point = positions[index]
            if interval is not None and point is not None:
                raise ValueError('Candidate source motion requires qualified ground-contact integration before measurements')
            record = accumulators[identity].observe(timestamps[index], point,
                         transition_allowed=bool(interval['measurement_transition_allowed']) if interval is not None else True)
            if interval is not None:
                record.update(source_timestamp_s=interval['source_timestamp_seconds'],
                              source_pts=interval['source_pts'], source_time_base=interval['source_time_base'],
                              camera_segment_id=interval['segment_id'],
                              metric_measurements_available=False)
            pair[identity] = record
        destination.append(pair)
    return {identity: {**accumulator.summary(), 'samples': MotionSampleView(destination, identity)}
            for identity, accumulator in accumulators.items()}


class _FrameArray(list):
    def __init__(self, frames, sanitize):
        self.frames, self.sanitize = frames, sanitize

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for frame in self.frames:
            yield self.sanitize(frame)


def write_player_motion_json(destination, metadata, pose_frames, sanitize):
    """Publish the established nested contract without materializing samples."""
    output = {key: sanitize(value) for key, value in metadata.items() if key not in ('player_1', 'player_2')}
    for identity in (1, 2):
        player = metadata[f'player_{identity}']
        output[f'player_{identity}'] = {key: sanitize(value) for key, value in player.items() if key != 'samples'}
        output[f'player_{identity}']['samples'] = _FrameArray(player['samples'], sanitize)
    output['pose_frames'] = _FrameArray(pose_frames, sanitize)
    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.' + destination.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(output, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException as primary:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup:
                primary.add_note(f'Motion artifact cleanup also failed: {cleanup}')
        raise
