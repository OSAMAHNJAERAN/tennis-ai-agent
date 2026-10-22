"""Read pinned author loader text; verify Git blobs, never execute it."""
import hashlib
import json
from pathlib import Path
from acquire_text import ART, REPO, fetch


def main():
    receipt_path = ART / 'loader_acquisition_receipt.json'
    if receipt_path.exists():
        raise FileExistsError('Retain the prior loader receipt')
    receipt = json.loads((ART / 'acquisition_receipt.json').read_text(encoding='utf-8'))
    tree = json.loads((ART / 'benchmark_tree.json').read_text(encoding='utf-8'))
    assert tree['sha'] == receipt['pinned_commit'] and tree.get('truncated') is False
    index = {value['path']: value for value in tree['tree']}
    records = []
    for name in ('benchmark/benchmark_loader.py', 'benchmark/loaders_helpers.py'):
        expected = index[name]
        assert expected['type'] == 'blob' and expected['size'] < 16384
        target = 'publisher_' + Path(name).name + '.txt'
        record = fetch('https://raw.githubusercontent.com/' + REPO + '/' + receipt['pinned_commit'] + '/' + name, target)
        body = (ART / target).read_bytes()
        git_sha = hashlib.sha1(b'blob ' + str(len(body)).encode('ascii') + b'\0' + body).hexdigest()
        assert len(body) == expected['size'] and git_sha == expected['sha']
        records.append({**record, 'publisher_path': name, 'verified_git_blob_sha1': git_sha})
    value = {'status': 'PINNED_LOADER_TEXT_ACQUIRED_NOT_EXECUTED', 'pinned_commit': receipt['pinned_commit'],
             'records': records, 'media_targets_or_archives_acquired': False,
             'models_or_resource_probes': False, 'metric_speed_reference_qualified': False}
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': value['status'], 'files': len(records),
                      'bytes': sum(record['bytes'] for record in records)}, indent=2))


if __name__ == '__main__':
    main()
