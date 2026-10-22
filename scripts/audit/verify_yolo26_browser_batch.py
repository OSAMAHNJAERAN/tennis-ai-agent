"""Bind an observed browser upload to durable server bytes; no inference claim."""
import argparse
import hashlib
import json
import time

from verify_yolo26_private_upload import ROOT, SOURCES, login, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    owner = login('local_browser_test')
    batch = request(owner, 'GET', f'/api/batches/{args.batch}', 200)
    assert batch['mode'] == 'independent' and len(batch['uploads']) == 2
    files = []
    for source, upload in zip(SOURCES, batch['uploads']):
        assert upload['filename'] == source.name and upload['status'] == 'COMPLETED'
        assert upload['offset'] == upload['size_bytes'] == source.stat().st_size
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert upload['checksum'] == digest and upload['job']['status'] == 'UPLOADED'
        job = request(owner, 'GET', f"/api/analyses/{upload['analysis_id']}", 200)
        assert job['cancel_requested'] is False
        files.append({'name': source.name, 'bytes': upload['offset'], 'sha256': digest,
                      'upload_id': upload['id'], 'analysis_id': job['id']})
    out = ROOT / 'artifacts/validation/yolo26_execution_2026-10-08/browser_batch_verified.json'
    receipt = {'status': 'PASS', 'checked_unix': time.time(), 'browser': 'Codex In-app Browser',
               'browser_file_chooser_observed': True, 'browser_multiple_selection_observed': True,
               'browser_analyze_click_observed': True, 'browser_both_saved_analysis_links_observed': True,
               'observation_source': 'CUA accessibility state, actual user-interface interaction',
               'batch_id': args.batch, 'files': files, 'worker_enabled': False,
               'vision_qualification': False, 'frontend_build': 'frontend_build_v4',
               'earlier_chrome_file_permission_limitation': 'No Chrome extension permission changed. Later in-app browser upload succeeded.',
               'earlier_contract_failure': 'SQLite cancel_requested integer failed frontend boolean schema; repaired in public_job.',
               'script_sha256': hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'receipt': str(out), 'files': len(files)}))


if __name__ == '__main__':
    main()
