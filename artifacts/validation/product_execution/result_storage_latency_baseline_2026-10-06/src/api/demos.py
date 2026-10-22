"""Publish only explicitly curated, completed analyses with matching evidence."""

import json
import sqlite3
from pathlib import Path

from src.api.jobs import REPO_ROOT
from src.api.media_source import ANALYSIS_ID, ATTRIBUTION_FIELDS, validate_media_source
from src.api.results import file_hash
from src.api.media_revision import presented_media_source, revisioned_url

DEMO_MANIFEST = REPO_ROOT / 'data/demos/sources.json'


def _read_object(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda _: None)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def curated_demos(store, manifest_path: Path | None = None):
    manifest = _read_object(manifest_path if manifest_path is not None else DEMO_MANIFEST)
    sources = manifest.get('sources')
    if manifest.get('schemaVersion') != 1 or not isinstance(sources, list):
        return {'schemaVersion': 1, 'demos': []}
    demos, seen_ids, seen_analyses = [], set(), set()
    for entry in sources[:100]:
        source = validate_media_source(entry)
        if source is None or entry.get('status') != 'COMPLETED':
            continue
        analysis_id = entry.get('analysisId')
        if not isinstance(analysis_id, str) or ANALYSIS_ID.fullmatch(analysis_id) is None:
            continue
        if source['id'] in seen_ids or analysis_id in seen_analyses:
            continue
        job = store.get(analysis_id)
        # Curated acquisition may preserve WebM or Matroska source cadence.
        # This list does not change ordinary upload validation in app.py.
        if job is None or job['status'] != 'COMPLETED' or job['extension'] not in {'.mp4', '.mov', '.avi', '.webm', '.mkv'}:
            continue
        directory = store.output_dir(analysis_id)
        from src.api.result_document import open_result
        try:
            with open_result(directory, analysis_id) as result:
                summary = result.get('summary')
                result_source = validate_media_source(result.get('mediaSource'))
        except (OSError, ValueError, sqlite3.Error):
            continue
        if not isinstance(summary, dict) or summary.get('id') != analysis_id or summary.get('status') != 'COMPLETED':
            continue
        persisted_source = validate_media_source(_read_object(directory / 'media_source.json'))
        if persisted_source != source or result_source != source:
            continue
        original = store.upload_dir(analysis_id) / ('original' + job['extension'])
        if file_hash(original) != source['inputSha256'] or not (directory / 'annotated_web.mp4').is_file():
            continue
        presented_source = presented_media_source(source, directory, analysis_id)
        seen_ids.add(source['id'])
        seen_analyses.add(analysis_id)
        demos.append({'id': source['id'], 'title': source['title'], 'analysisId': analysis_id,
                      'previewUrl': revisioned_url(f'/api/analyses/{analysis_id}/media/annotated', directory / 'annotated_web.mp4'),
                      'attribution': {key: presented_source[key] for key in ATTRIBUTION_FIELDS},
                      **({'excerpt': source['excerpt']} if 'excerpt' in source else {}),
                      'qualificationEvidence': False})
    return {'schemaVersion': 1, 'demos': demos}
