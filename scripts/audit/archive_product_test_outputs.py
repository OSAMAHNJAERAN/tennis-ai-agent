"""Preserve and consolidate only five audited historical pytest output trees.

Every byte is verified in a recoverable ZIP before originals are removed. The
manifest records relative aliases separately, avoiding traversal of links.
Persistent outputs/product, user media and scientific datasets are excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2].resolve(strict=True)
BASELINE = ROOT / 'outputs/execution_baseline/2026-09-30'
INVENTORY = BASELINE / 'cleanup_contents_2026-10-01_v2.json'
ARCHIVE = BASELINE / 'cleanup_test_outputs_2026-10-01.zip'
RECEIPT = BASELINE / 'cleanup_completed_2026-10-01.json'
ALLOWED = {f'outputs/{name}' for name in ('product_regression_tmp','product_regression_tmp2','product_review_tests','product_test_final','product_test_tmp')}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(relative):
    raw = Path(relative)
    if raw.is_absolute() or '..' in raw.parts:
        raise ValueError('Only safe relative paths may be archived.')
    path = ROOT / raw
    path.resolve(strict=True).relative_to(ROOT)
    return path


def main():
    if ARCHIVE.exists() or RECEIPT.exists():
        raise FileExistsError('Preserve the first archive/receipt; inspect it before another run.')
    inventory = json.loads(INVENTORY.read_text(encoding='utf-8'))
    if {row['path'] for row in inventory} != ALLOWED:
        raise ValueError('Inventory does not match the five explicitly authorized temporary trees.')
    before_bytes = before_files = 0
    targets = []
    for row in inventory:
        target = safe_path(row['path'])
        if target.is_symlink() or target.is_junction():
            raise ValueError('Refuse a linked target directory.')
        expected = {file['path'] for file in row['entries']}
        expected_links = {link['path'] for link in row['links']}
        actual_files, actual_links = set(), set()
        for path in target.rglob('*'):
            relative = path.relative_to(target).as_posix()
            if path.is_symlink():
                path.resolve(strict=True).relative_to(target)
                actual_links.add(relative)
            elif path.is_junction() or path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f'Refuse unrecorded reparse point {path}')
            elif path.is_file():
                actual_files.add(relative)
        if actual_files != expected or actual_links != expected_links:
            raise ValueError(f"Temporary contents differ at {row['path']}: "
                             f"added_files={sorted(actual_files-expected)[:8]}, "
                             f"missing_files={sorted(expected-actual_files)[:8]}, "
                             f"added_aliases={sorted(actual_links-expected_links)[:8]}, "
                             f"missing_aliases={sorted(expected_links-actual_links)[:8]}; re-audit first.")
        for file in row['entries']:
            path = safe_path(f"{row['path']}/{file['path']}")
            if path.stat().st_size != file['bytes'] or digest(path) != file['sha256']:
                raise ValueError('An audited file changed; preserve and re-audit first.')
            before_bytes += file['bytes']
            before_files += 1
        for link in row['links']:
            path = target / link['path']
            if path.resolve(strict=True) != (target / link['target']).resolve(strict=True):
                raise ValueError('Recorded alias changed.')
        targets.append(target)
    with ZipFile(ARCHIVE, 'x', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr('manifest.json', json.dumps(inventory, indent=2))
        for row in inventory:
            for file in row['entries']:
                relative = f"{row['path']}/{file['path']}"
                archive.write(safe_path(relative), relative)
    with ZipFile(ARCHIVE) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC verification failed; preserve all originals.')
        for row in inventory:
            for file in row['entries']:
                data = archive.read(f"{row['path']}/{file['path']}")
                if len(data) != file['bytes'] or hashlib.sha256(data).hexdigest() != file['sha256']:
                    raise ValueError('Archive byte verification failed; preserve all originals.')
    receipt = {'action':'ARCHIVE_VERIFIED_TEMPORARY_PYTEST_OUTPUTS_THEN_REMOVE_ORIGINALS',
               'paths':sorted(ALLOWED),'beforeBytes':before_bytes,'beforeFiles':before_files,
               'archivePath':ARCHIVE.relative_to(ROOT).as_posix(),'archiveBytes':ARCHIVE.stat().st_size,
               'archiveSha256':digest(ARCHIVE),'everyFileHashVerified':True,
               'recovery':'ZIP contains each original workspace-relative file; manifest.json records local pytest aliases. Extract files under the workspace root to recover payloads; aliases are optional fixture conveniences.',
               'persistentProductStorageChanged':False,'scientificDataChanged':False}
    # All aliases were independently resolved inside their literal tree above.
    # Remove links first so recursive removal cannot walk an alias target.
    for row, target in zip(inventory, targets):
        for link in row['links']:
            (target / link['path']).unlink()
        target.resolve(strict=True).relative_to(ROOT / 'outputs')
        shutil.rmtree(target)
    receipt['afterTemporaryBytes'] = sum(sum(path.stat().st_size for path in target.rglob('*') if path.is_file()) for target in targets)
    receipt['allFiveTargetsAbsent'] = all(not target.exists() for target in targets)
    receipt['netLogicalBytesSavedBeforeReceipt'] = before_bytes - receipt['archiveBytes']
    RECEIPT.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
