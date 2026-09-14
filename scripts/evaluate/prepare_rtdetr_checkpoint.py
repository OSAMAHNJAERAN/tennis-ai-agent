"""Download one official RT-DETR asset and verify the release digest before inspection."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT/'artifacts/models/person/rtdetr_research'
EXPECTED = '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f'


def main():
    metadata_url = 'https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0'
    with urllib.request.urlopen(urllib.request.Request(metadata_url, headers={'User-Agent': 'tennis-vision-research'}), timeout=30) as response:
        release = json.load(response)
    asset = next(a for a in release['assets'] if a['name'] == 'rtdetr-l.pt')
    if asset['digest'] != 'sha256:'+EXPECTED or asset['size'] != 66511432:
        raise ValueError('Official asset changed')
    FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = FOLDER/asset['name']
    if not checkpoint.exists():
        temporary = FOLDER/'rtdetr-l.pt.partial'
        if temporary.exists():
            raise FileExistsError('Prior incomplete download requires inspection')
        with urllib.request.urlopen(urllib.request.Request(asset['browser_download_url'], headers={'User-Agent': 'tennis-vision-research'}), timeout=60) as response, temporary.open('xb') as target:
            while block := response.read(1024*1024):
                target.write(block)
        if temporary.stat().st_size != asset['size'] or hashlib.sha256(temporary.read_bytes()).hexdigest() != EXPECTED:
            raise ValueError('Downloaded asset does not match official size/digest')
        temporary.rename(checkpoint)
    if checkpoint.stat().st_size != asset['size'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Existing checkpoint differs')
    report = {'complete': True, 'metadata_url': metadata_url, 'asset': {k: asset[k] for k in
        ['id', 'name', 'size', 'digest', 'browser_download_url', 'created_at', 'updated_at']},
        'checkpoint_sha256': EXPECTED, 'checkpoint': str(checkpoint.relative_to(ROOT)),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Official Ultralytics converted checkpoint; local research only; no accuracy claim'}
    (FOLDER/'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
