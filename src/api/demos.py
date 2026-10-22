"""Publish only explicitly curated, completed analyses with matching evidence."""

import json
import re
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
        selected_generation = entry.get('generation')
        if selected_generation is not None and (not isinstance(selected_generation, str)
                or re.fullmatch('[0-9a-f]{64}', selected_generation) is None):
            continue
        job = store.get(analysis_id)
        # Curated acquisition may preserve WebM or Matroska source cadence.
        # This list does not change ordinary upload validation in app.py.
        if job is None or job['status'] != 'COMPLETED' or job['extension'] not in {'.mp4', '.mov', '.avi', '.webm', '.mkv'}:
            continue
        directory = store.output_dir(analysis_id)
        from src.api.result_document import ResultDocument
        try:
            with ResultDocument(directory, analysis_id, generation=selected_generation) as document:
                if document.generation is not None and selected_generation != document.generation:
                    # Selecting a source is not permission to publish every
                    # later correction/profile revision under that analysis ID.
                    continue
                result = document.value
                summary = result.get('summary')
                result_source = validate_media_source(result.get('mediaSource'))
                if (not isinstance(summary, dict) or summary.get('id') != analysis_id
                        or summary.get('status') != 'COMPLETED'):
                    continue
                source_path = document.asset_path('media_source.json')
                persisted_source = validate_media_source(_read_object(source_path)) if source_path is not None else None
                if persisted_source != source or result_source != source:
                    continue
                annotated = document.asset_path('annotated_web.mp4')
                if annotated is None or not annotated.is_file():
                    continue
                if document.snapshot is not None:
                    original_name = 'original' + job['extension']
                    if document.snapshot.file_sha256(original_name) != source['inputSha256']:
                        continue
                    presented = document.presented(analysis_id)
                    presented_source = presented['mediaSource']
                    preview = presented.get('annotatedVideoUrl')
                    if not isinstance(preview, str):
                        continue
                else:
                    original = store.upload_dir(analysis_id) / ('original' + job['extension'])
                    if file_hash(original) != source['inputSha256']:
                        continue
                    presented_source = presented_media_source(source, directory, analysis_id)
                    preview = revisioned_url(f'/api/analyses/{analysis_id}/media/annotated', annotated)
        except (OSError, ValueError, sqlite3.Error, TypeError, KeyError):
            continue
        seen_ids.add(source['id'])
        seen_analyses.add(analysis_id)
        demos.append({'id': source['id'], 'title': source['title'], 'analysisId': analysis_id,
                      'previewUrl': preview,
                      'attribution': {key: presented_source[key] for key in ATTRIBUTION_FIELDS},
                      **({'excerpt': source['excerpt']} if 'excerpt' in source else {}),
                      'qualificationEvidence': False})
    return {'schemaVersion': 1, 'demos': demos}
