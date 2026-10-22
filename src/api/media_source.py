"""Bounded attribution for deliberately curated, licensed demonstration media."""

import math
import re
from urllib.parse import urlsplit

SOURCE_HOSTS = frozenset({'commons.wikimedia.org'})
LICENSE_HOSTS = frozenset({'creativecommons.org'})
DEMO_ID = re.compile(r'[a-z0-9][a-z0-9_-]{0,79}')
ANALYSIS_ID = re.compile(r'match_[0-9a-f]{32}')
SHA256 = re.compile(r'[0-9a-f]{64}')
ATTRIBUTION_FIELDS = ('author', 'sourcePage', 'license', 'licenseUrl', 'derivativeLicense', 'modifications')


def plain_text(value, maximum=1000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        return None
    return value.strip()


def safe_https_url(value, allowed_hosts):
    value = plain_text(value, 2048)
    if value is None or any(c.isspace() for c in value) or '\\' in value:
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or parsed.hostname not in allowed_hosts or
                parsed.username is not None or parsed.password is not None or parsed.port not in {None, 443}):
            return None
    except ValueError:
        return None
    return value


def _number(value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return value if not integer or int(value) == value else None


def validate_media_source(value):
    """Return only public credit fields; a malformed optional section stays null.

    The current curated collection is Wikimedia Commons with Creative Commons
    license pages. New source domains require a deliberate whitelist change.
    Validation records attribution; it does not independently certify rights.
    """
    if not isinstance(value, dict):
        return None
    identity = value.get('id')
    title = plain_text(value.get('title'), 240)
    if not isinstance(identity, str) or DEMO_ID.fullmatch(identity) is None or title is None:
        return None
    source = {'id': identity, 'title': title}
    for key in ATTRIBUTION_FIELDS:
        field = (safe_https_url(value.get(key), SOURCE_HOSTS) if key == 'sourcePage' else
                 safe_https_url(value.get(key), LICENSE_HOSTS) if key == 'licenseUrl' else
                 plain_text(value.get(key)))
        if field is None:
            return None
        source[key] = field
    for key in ('sourceSha256', 'inputSha256'):
        field = value.get(key)
        if not isinstance(field, str) or SHA256.fullmatch(field) is None:
            return None
        source[key] = field
    excerpt = value.get('excerpt')
    if excerpt is not None:
        if not isinstance(excerpt, dict):
            return None
        normalized = {}
        for key in ('startSeconds', 'endSeconds', 'startFrame', 'endFrameExclusive'):
            field = _number(excerpt.get(key), integer=key in {'startFrame', 'endFrameExclusive'})
            if field is None:
                return None
            normalized[key] = field
        if normalized['endSeconds'] <= normalized['startSeconds'] or normalized['endFrameExclusive'] <= normalized['startFrame']:
            return None
        for key in ('sourceTimeBase', 'firstSourcePts', 'lastSourcePts'):
            field = plain_text(excerpt.get(key), 100)
            if field is None:
                return None
            normalized[key] = field
        source['excerpt'] = normalized
    return source
