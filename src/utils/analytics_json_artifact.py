"""Atomic analytics exports with incremental nested shot memberships.

Only the native caller-owned path uses lazy views. Completed JSON retains the
legacy arrays; the returned native summary references the published heatmap.
"""
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
import json
import os
from pathlib import Path
import tempfile

from src.utils.json_array_artifact import write_json_array_artifact


@contextmanager
def _destination(destination):
    destination = Path(destination)
    temporary = stream = None
    try:
        stream = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', delete=False,
            dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp')
        temporary = Path(stream.name)
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        os.replace(temporary, destination)
    except BaseException as error:
        try:
            try:
                if stream is not None and not stream.closed:
                    stream.close()
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        except BaseException as cleanup_error:
            error.add_note(f'Analytics temporary-file cleanup also failed: {cleanup_error}')
        raise


def _close(iterator):
    close = getattr(iterator, 'close', None)
    if close is not None:
        close()


def write_membership_analytics(destination, array_key, records, sanitize, *, schema_version='1.0'):
    """Write point/rally records without copying their nested ``shot_ids``."""
    iterator = iter(records)
    try:
        with _destination(destination) as stream:
            stream.write('{"schema_version":')
            json.dump(schema_version, stream)
            stream.write(',')
            json.dump(array_key, stream)
            stream.write(':[')
            first_record = True
            for record in iterator:
                if not first_record:
                    stream.write(',')
                first_record = False
                # asdict deep-copies arbitrary values, including SQLite owners.
                values = ((item.name, getattr(record, item.name)) for item in fields(record)) if is_dataclass(record) else record.items()
                stream.write('{')
                first_field = True
                for key, value in values:
                    if not first_field:
                        stream.write(',')
                    first_field = False
                    json.dump(key, stream)
                    stream.write(':')
                    if key != 'shot_ids' or value is None:
                        json.dump(sanitize(value), stream, allow_nan=False, separators=(',', ':'))
                        continue
                    members = iter(value)
                    try:
                        stream.write('[')
                        first_member = True
                        for member in members:
                            if not first_member:
                                stream.write(',')
                            first_member = False
                            json.dump(sanitize(member), stream, allow_nan=False, separators=(',', ':'))
                        stream.write(']')
                    finally:
                        _close(members)
                stream.write('}')
            stream.write(']}\n')
    finally:
        _close(iterator)


def write_match_analytics(destination, analytics, sanitize):
    metadata = {'schema_version': '1.0', **{key: value for key, value in analytics.items()
                                          if key != 'ball_landing_heatmap'}}
    def records():
        iterator = iter(analytics['ball_landing_heatmap'])
        try:
            for row in iterator:
                yield sanitize(row)
        finally:
            _close(iterator)
    write_json_array_artifact(destination, sanitize(metadata), 'ball_landing_heatmap', records(), allow_nan=False)


def native_match_analytics_summary(analytics):
    """No temporary SQLite owner or lazy view may outlive the pipeline scope."""
    summary = {key: value for key, value in analytics.items() if key != 'ball_landing_heatmap'}
    summary['ball_landing_heatmap_artifact'] = {
        'path': 'match_analytics.json',
        'array_key': 'ball_landing_heatmap',
        'row_count': len(analytics['ball_landing_heatmap']),
    }
    return summary
