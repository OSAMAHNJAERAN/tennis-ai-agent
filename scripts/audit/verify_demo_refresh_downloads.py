"""Check the browser-downloaded refresh against immutable presentation receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'artifacts/validation/product_execution'
ANALYSIS = 'match_02d494b88e99400d8cf7243528302f9b'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = json.loads((BASE / 'public_demo_presentation_refresh_2026-10-01/receipt.json').read_text())
    record = next(item for item in receipt['demos'] if item['analysisId'] == ANALYSIS)
    downloads = Path('C:/Users/ac-98/Downloads')
    video = downloads / (ANALYSIS + '_annotated (1).mp4')
    report = downloads / (ANALYSIS + '_readable-report (1).html')
    assert digest(video) == record['staged']['sha256']
    html = report.read_text(encoding='utf-8')
    assert ANALYSIS in html and 'Oleg Yunakov' in html and 'CC BY-SA 4.0' in html
    assert 'data:image/jpeg;base64,' in html and 'share-alike' in html.lower()
    (BASE / 'demo_refresh_browser_download_integrity_2026-10-01.json').write_text(json.dumps({
        'verification': 'BROWSER_DOWNLOADED_VIDEO_HASH_AND_REPORT_ID_CREDITS; NOT_MODEL_ACCURACY',
        'analysisId': ANALYSIS, 'video': str(video), 'videoSha256': digest(video),
        'downloadEqualsReviewedPublishedPresentation': True,
        'report': str(report), 'reportSha256': digest(report), 'reportBytes': report.stat().st_size,
        'reportMatchingIdAuthorLicenseAndEmbeddedImages': True}, indent=2), encoding='utf-8')
    print('Browser-downloaded video matches the reviewed refresh; report identity, credits and embedded evidence verified.')


if __name__ == '__main__':
    main()
