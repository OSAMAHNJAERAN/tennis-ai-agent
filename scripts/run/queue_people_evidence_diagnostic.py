"""Queue one bounded diagnostic re-run without replacing public demos.

This is artifact verification and development material, never qualification.
The input is the previously authorized, lossless Commons rally excerpt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.api.jobs import JobStore, Settings

RECORD = ROOT / 'artifacts/validation/product_execution/people_evidence_run.json'


def main():
    settings = Settings.from_env()
    store = JobStore(settings.root)
    if RECORD.exists():
        record = json.loads(RECORD.read_text(encoding='utf-8'))
        job = store.get(record['analysisId'])
        if not job:
            raise RuntimeError('Diagnostic record exists without its persistent job; inspect before retrying.')
        print(json.dumps({'analysisId':job['id'], 'status':job['status']}))
        return
    manifest = json.loads((ROOT / 'data/demos/sources.json').read_text(encoding='utf-8'))
    source = next(item for item in manifest['sources'] if item['id'] == 'dimitrov_tiafoe_2024')
    selected = (ROOT / source['inputPath']).resolve(strict=True)
    selected.relative_to((ROOT / 'artifacts/demos/sources').resolve(strict=True))
    with selected.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != source['inputSha256']:
            raise ValueError('Diagnostic input changed.')
    analysis_id = store.new_id()
    upload, output = store.upload_dir(analysis_id), store.output_dir(analysis_id)
    upload.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(selected, upload / ('original' + selected.suffix))
    attribution = {key:source[key] for key in ('id','title','author','sourcePage','license','licenseUrl','derivativeLicense','modifications','sourceSha256','inputSha256','excerpt')}
    (output / 'media_source.json').write_text(json.dumps(attribution, indent=2), encoding='utf-8')
    record = {'analysisId':analysis_id,'parentAnalysisId':source['analysisId'],'sourceSha256':source['sourceSha256'],
              'inputSha256':source['inputSha256'],'inputPath':source['inputPath'],'purpose':'RAW_PERSON_EXPORT_AND_RENDERING_DIAGNOSTIC',
              'qualificationEvidence':False,'publicCatalogReplaced':False}
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=2), encoding='utf-8')
    store.create(analysis_id, 'Dimitrov-Tiafoe person-evidence diagnostic.mkv', selected.suffix, selected.stat().st_size, settings.config)
    print(json.dumps({'analysisId':analysis_id,'status':'UPLOADED','qualificationEvidence':False}))


if __name__ == '__main__':
    main()
