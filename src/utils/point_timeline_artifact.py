"""Hash-bound publication of one native rule-timeline artifact bundle.

This manifest seals these eight files only. It is not an atomic publication of
the full analysis, video, or a later correction revision.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


POINT_TIMELINE_FILES = (
    'point_analytics.json', 'rallies.json', 'match_analytics.json',
    'serve_executions.json', 'shot_play_scope.json', 'scoring_events.json',
    'score_history.json', 'match_state.json',
)
MANIFEST_NAME = 'point_timeline_manifest.json'
SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')


def artifact_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def publish_point_timeline_manifest(output_dir, *, analysis_id, analysis_revision,
                                    source_sha256, scoring_revision,
                                    authoritative_events_enabled):
    """Publish only after all eight native exports have closed successfully."""
    if (not isinstance(analysis_id, str) or not analysis_id
            or type(analysis_revision) is not int or analysis_revision < 0
            or type(scoring_revision) is not int or scoring_revision != 0
            or type(authoritative_events_enabled) is not bool
            or not isinstance(source_sha256, str)
            or SHA256_PATTERN.fullmatch(source_sha256) is None):
        raise ValueError('Invalid native point-timeline publication identity')
    directory = Path(output_dir).resolve()
    artifacts = {}
    for name in POINT_TIMELINE_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Native point-timeline export is missing: '+name)
        artifacts[name] = {'sha256': artifact_sha256(path), 'bytes': path.stat().st_size}
    manifest = {
        'schema_version': '1.0', 'analysis_id': analysis_id,
        'analysis_revision': analysis_revision, 'source_sha256': source_sha256,
        'scoring_revision': scoring_revision,
        'authoritative_events_enabled': authoritative_events_enabled,
        'independent_accuracy_validated': False, 'artifacts': artifacts,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=directory, prefix='.'+MANIFEST_NAME+'.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / MANIFEST_NAME)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup_error:
                error.add_note('Point manifest cleanup also failed: '+str(cleanup_error))
        raise
    return manifest
