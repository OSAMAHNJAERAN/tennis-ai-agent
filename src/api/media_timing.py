"""Stream FFmpeg's actual decoder-to-normalized-frame provenance to JSONL."""

from fractions import Fraction
import json
import math


def frame_mapping(stats_path, destination, fps, source_frame_count=None, source_start_time=0):
    """Capture exact PTS tokens; gate window measurements across changed timing.

    Input format is FFmpeg stats_enc_pre: n ni ptsi tbi pts tb. The raw
    source timestamps are retained through -copyts, and normalized output is
    explicitly rebased by setpts=PTS-STARTPTS. This is timing provenance, not
    independent camera, position or physical-speed qualification.
    """
    rate = Fraction(str(fps))
    if rate <= 0:
        raise ValueError('Normalized frame rate must be positive.')
    start = Fraction(str(source_start_time or 0))
    unique_sources = duplicates = 0
    last_source_index = None
    intervals = []
    interval_start = None
    previous = None
    first_source_time = None
    incomplete = False
    mismatch_max = 0.0
    count = 0
    # Decoder indices are monotonic through FFmpeg's fps filter. Keep counts
    # and the previous index only; frame records are streamed to disk.
    with stats_path.open(encoding='utf-8') as source, destination.open('w', encoding='utf-8') as output:
        for line in source:
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 6:
                raise ValueError('Malformed FFmpeg frame provenance.')
            index, source_index, source_pts, input_tb, output_pts, output_tb = fields
            index, source_index, source_pts, output_pts = map(int, (index, source_index, source_pts, output_pts))
            if index != count:
                raise ValueError('FFmpeg provenance frame order is incomplete.')
            input_base, output_base = Fraction(input_tb), Fraction(output_tb)
            normalized_time = output_pts * output_base
            if abs(float(normalized_time - Fraction(index, 1) / rate)) > 1e-6:
                raise ValueError('Normalized presentation time does not match the analysis clock.')
            available = source_index >= 0 and source_pts != 9223372036854775807 and input_base > 0
            source_time = source_pts * input_base if available else None
            if source_time is not None and not math.isfinite(float(source_time)):
                available, source_time = False, None
            if source_time is not None and first_source_time is None:
                first_source_time = source_time
            interval_valid = False
            mismatch = None
            # Eligible transitions require distinct adjacent decoder frames and
            # cadence agreement within one source tick (at least 1 microsecond).
            # A speed fit must require every transition in its window to pass.
            if available and previous is not None and previous['available']:
                delta = source_time - previous['time']
                mismatch = abs(float(delta - 1 / rate))
                mismatch_max = max(mismatch_max, mismatch)
                tolerance = max(float(input_base), float(previous['time_base']), 1e-6)
                interval_valid = (source_index == previous['index'] + 1 and delta > 0 and mismatch <= tolerance + 1e-9)
            if available:
                if last_source_index is not None and source_index < last_source_index:
                    raise ValueError('Source decoder-frame order moved backward.')
                if source_index == last_source_index:
                    duplicates += 1
                else:
                    unique_sources += 1
                last_source_index = source_index
            else:
                incomplete = True
            if interval_valid and interval_start is None:
                interval_start = index
            if not interval_valid and interval_start is not None:
                intervals.append({'startFrame': interval_start, 'endFrameExclusive': index})
                interval_start = None
            record = {'frame_index': index, 'timestamp_seconds': float(normalized_time),
                      'normalized_pts': str(output_pts), 'normalized_time_base': output_tb,
                      'source_frame_index': source_index if available else None,
                      'source_pts': str(source_pts) if available else None,
                      'source_time_base': input_tb if available else None,
                      'source_timestamp_seconds': float(source_time) if source_time is not None else None,
                      'source_elapsed_seconds': float(source_time - start) if source_time is not None else None,
                      'validMeasurementTransitionFromPrevious': interval_valid,
                      'cadence_mismatch_seconds': mismatch}
            output.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
            previous = {'available': available, 'index': source_index, 'time': source_time, 'time_base': input_base}
            count += 1
    if interval_start is not None:
        intervals.append({'startFrame': interval_start, 'endFrameExclusive': count})
    if not count:
        raise ValueError('No normalized frame provenance was emitted.')
    known_count = source_frame_count if isinstance(source_frame_count, int) and source_frame_count >= 0 else None
    if known_count is not None and last_source_index is not None and last_source_index >= known_count:
        raise ValueError('Source frame indices exceed the declared decoded frame count.')
    return {'status': 'PARTIAL_FFMPEG_FRAME_MAP' if incomplete else 'CAPTURED_FFMPEG_FRAME_MAP',
            'frameMapping': None, 'frameMappingArtifact': destination.name, 'frames': count,
            'sourceStartTimeSeconds': float(start),
            'firstMappedSourceTimestampSeconds': float(first_source_time) if first_source_time is not None else None,
            'sourceFramesDeclared': known_count, 'uniqueMappedSourceFrames': unique_sources,
            'duplicatedOutputFrames': duplicates,
            'droppedSourceFrames': known_count - unique_sources if known_count is not None else None,
            'maximumCadenceMismatchSeconds': mismatch_max,
            'validMeasurementTransitionIntervals': intervals,
            'measurementCadenceTolerance': 'One source time-base tick, at least one microsecond.',
            'semantics': 'JSONL records normalized presentation time and original decoder frame index/PTS. '
                         'Movement windows exclude duplicated/dropped/missing or cadence-invalid transitions. '
                         'Captured timing does not establish camera or physical-speed accuracy.'}
