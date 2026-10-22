"""One original source's public text metadata; never fetch the video."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ART = Path(__file__).resolve().parent
URL = 'https://www.youtube.com/oembed?url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DMG-RjlqyaJI&format=json'


def main():
    if set(path.name for path in ART.iterdir()) != {'acquire_oembed.py'}:
        raise RuntimeError('Preserve the prior text attempt')
    receipt = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'requested_url': URL,
        'source_id': 'MG-RjlqyaJI', 'scope': 'SINGLE_SOURCE_PUBLIC_OEMBED_TEXT_ONLY',
        'media_or_targets_acquired': False, 'source_specific_research_grant_established': False,
        'independent_test_admitted': False, 'models_training_or_resource_probes': False}
    try:
        request = Request(URL, headers={'User-Agent': 'TennisResearchMetadata/1.0', 'Accept': 'application/json'})
        with urlopen(request, timeout=20) as response:
            assert response.headers.get_content_type() == 'application/json'
            body = response.read(65537)
            if len(body) > 65536:
                raise ValueError('oEmbed response exceeds 64 KiB')
            receipt.update(status=response.status, final_url=response.url)
        value = json.loads(body)
        assert isinstance(value, dict) and isinstance(value.get('title'), str)
        with (ART / 'source_oembed.json').open('xb') as stream:
            stream.write(body)
        receipt.update(result='TEXT_METADATA_ACQUIRED_NO_DATA_GRANT', bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            metadata={key: value.get(key) for key in ('title', 'author_name', 'author_url', 'provider_name', 'type')},
            metadata_keys=sorted(value))
    except Exception as error:
        receipt.update(result='ACCESS_OR_VALIDATION_FAILURE', error_type=type(error).__name__, error=str(error))
    with (ART / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
