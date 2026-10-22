"""Present current media URLs and hash-bound derivative credit in response copies."""
import json
from pathlib import Path
import unicodedata

from src.api.media_source import ANALYSIS_ID, SHA256, plain_text, validate_media_source
from src.api.results import file_hash


def _read_object(path: Path, maximum_bytes: int = 8192) -> dict:
    try:
        if path.stat().st_size > maximum_bytes:
            return {}
        value = json.loads(path.read_text(encoding='utf-8'), parse_constant=lambda _: None)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def presented_media_source(source, directory: Path, analysis_id: str):
    """Change only derivative modifications when the sibling credit matches.

    The original source credit remains authoritative for authorship, licensing,
    URLs, source/input hashes and excerpt. Saved evidence is never rewritten.
    """
    if not isinstance(analysis_id, str) or ANALYSIS_ID.fullmatch(analysis_id) is None:
        return source
    original = validate_media_source(source)
    persisted = validate_media_source(_read_object(directory / 'media_source.json'))
    if original is None or persisted != original:
        return source
    sidecar = _read_object(directory.parent / f'{analysis_id}.annotated_web_credit.json')
    if (set(sidecar) != {'schemaVersion', 'analysisId', 'inputSha256', 'sourceSha256',
                        'annotatedSha256', 'modifications'} or
            type(sidecar.get('schemaVersion')) is not int or sidecar['schemaVersion'] != 1 or
            sidecar.get('analysisId') != analysis_id):
        return source
    for field in ('inputSha256', 'sourceSha256', 'annotatedSha256'):
        value = sidecar.get(field)
        if not isinstance(value, str) or SHA256.fullmatch(value) is None:
            return source
    if (sidecar['inputSha256'] != original['inputSha256'] or
            sidecar['sourceSha256'] != original['sourceSha256']):
        return source
    modifications = plain_text(sidecar.get('modifications'))
    if (modifications is None or any(character in '<>' or unicodedata.category(character).startswith('C')
                                     for character in modifications) or
            modifications.startswith(('=', '+', '-', '@'))):
        return source
    if file_hash(directory / 'annotated_web.mp4') != sidecar['annotatedSha256']:
        return source
    return {**source, 'modifications': modifications}


def revisioned_url(url: str, path: Path) -> str:
    stat = path.stat()
    return f'{url}?v={stat.st_mtime_ns:x}-{stat.st_size:x}'


def presentation_urls(result: dict, directory: Path, analysis_id: str) -> dict:
    """Decorate response URLs only; never rewrite the persisted measurements."""
    result = {**result}
    base = '/api/analyses/' + analysis_id
    for field, kind, filename in (
            ('videoUrl', 'source', 'source.mp4'),
            ('annotatedVideoUrl', 'annotated', 'annotated_web.mp4')):
        path = directory / filename
        if result.get(field) == base + '/media/' + kind and path.is_file():
            result[field] = revisioned_url(result[field], path)
    exports = result.get('exports')
    path = directory / 'annotated_web.mp4'
    if isinstance(exports, dict) and exports.get('annotatedVideo') == base + '/download/annotated' and path.is_file():
        result['exports'] = {**exports, 'annotatedVideo': revisioned_url(exports['annotatedVideo'], path)}
    if 'mediaSource' in result:
        result['mediaSource'] = presented_media_source(result['mediaSource'], directory, analysis_id)
    return result
