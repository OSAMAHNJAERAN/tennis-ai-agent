"""Preserve prior documentation and state the historical-format detection limit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).parent
BACKUP = BASE / 'result_storage_reader_gate_document_baseline_2026-10-06'
RECEIPT = Path(__file__).with_suffix('.json')
assert not BACKUP.exists() and not RECEIPT.exists()
names = ('docs/plans/TENNIS_EXECUTION_PROGRESS.md', 'docs/deployment/PRODUCT_INSTALLATION.md',
         'docs/deployment/PRODUCT_RESTORE_MANIFEST.json')
before = {}
for name in names:
    data = (ROOT / name).read_bytes()
    target = BACKUP / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(data)
    before[name] = hashlib.sha256(data).hexdigest()

progress = ROOT / names[0]
text = progress.read_text(encoding='utf-8')
old = 'a missing/corrupt committed manifest or database is unavailable, never silently replaced with compatibility JSON.'
new = ('a missing/corrupt manifest or database within an existing `.result` directory is unavailable. '
       'An entirely omitted `.result` directory is indistinguishable from historical JSON and follows bounded legacy parsing; '
       'preserving complete hidden directories is required.')
assert text.count(old) == 1
progress.write_text(text.replace(old, new), encoding='utf-8')
installation = ROOT / names[1]
text = installation.read_text(encoding='utf-8')
old = 'Historical JSON analyses remain readable without migration; owned scratch is temporary reader storage.'
new = ('Historical JSON analyses remain readable without migration; owned scratch is temporary reader storage. '
       'The no-fallback gate applies when the `.result` directory exists. Omitting that entire directory is indistinguishable '
       'from historical JSON and takes the bounded legacy reader path, so retain the complete directory during backup/restoration.')
assert text.count(old) == 1
installation.write_text(text.replace(old, new), encoding='utf-8')
manifest_path = ROOT / names[2]
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
old = 'a missing or changed new-generation manifest/database is unavailable.'
new = ('a missing or changed manifest/database within an existing .result directory is unavailable. '
       'Entire-directory omission is indistinguishable from historical JSON and takes bounded legacy parsing; '
       'preserve the complete hidden generation directory.')
assert manifest['persistentState']['backupPolicy'].count(old) == 1
manifest['persistentState']['backupPolicy'] = manifest['persistentState']['backupPolicy'].replace(old, new)
manifest['evidenceUpdatedAt'] = datetime.now(timezone.utc).isoformat()
with manifest_path.open('w', encoding='utf-8') as stream:
    json.dump(manifest, stream, indent=2, allow_nan=False)
    stream.write('\n')
with RECEIPT.open('x', encoding='utf-8') as stream:
    json.dump({'verifiedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
               'correction': 'Qualify no-fallback guarantee to existing generation directory; whole-directory omission follows historical-format reader.',
               'productionCodeChanged': False, 'previousDocumentBytesPreserved': True, 'previousHashes': before,
               'currentHashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names},
               'priorHostAndDocumentationSealsHistorical': True}, stream, indent=2)
print(json.dumps({'corrected': True, 'productionCodeChanged': False, 'previousDocumentBytesPreserved': True}))
