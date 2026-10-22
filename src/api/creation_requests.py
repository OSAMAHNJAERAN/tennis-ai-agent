"""Creation acknowledgement identities committed with their upload sessions."""
import hashlib
import json

from fastapi import HTTPException

from src.api.jobs import utc_now
from src.api.retry_intents import KEY


class CreationRequests:
    def __init__(self, jobs):
        with jobs.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS upload_creation_requests (
                owner_id TEXT NOT NULL, intent_key TEXT NOT NULL, kind TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, descriptor_json TEXT NOT NULL,
                created_at TEXT NOT NULL, PRIMARY KEY(owner_id,intent_key))''')

    @staticmethod
    def key(request):
        keys = request.headers.getlist('idempotency-key')
        if len(keys) != 1 or KEY.fullmatch(keys[0]) is None:
            raise HTTPException(422, 'Send one stable Idempotency-Key of 16–128 letters, digits, underscores or hyphens.')
        return keys[0]

    @staticmethod
    def fingerprint(spec):
        payload = json.dumps(spec.model_dump(mode='json'), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @staticmethod
    def find(conn, owner_id, key, kind, fingerprint):
        if not conn.in_transaction:
            raise RuntimeError('Creation admission requires the caller write transaction')
        row = conn.execute('SELECT * FROM upload_creation_requests WHERE owner_id=? AND intent_key=?',
                           (owner_id, key)).fetchone()
        if row is None:
            return None
        if row['kind'] != kind or row['payload_sha256'] != fingerprint:
            raise HTTPException(409, 'This Idempotency-Key already belongs to a different upload creation request.')
        try:
            encoded = row['descriptor_json']
            if len(encoded) > 4096:
                raise ValueError('Oversized creation descriptor')
            result = json.loads(encoded)
            expected = {'id'} if kind == 'upload' else {'id', 'mode', 'upload_ids'}
            if not isinstance(result, dict) or set(result) != expected:
                raise ValueError('Invalid creation descriptor')
            prefix = 'upload_' if kind == 'upload' else 'batch_'
            if not isinstance(result['id'], str) or len(result['id']) != len(prefix)+32 or not result['id'].startswith(prefix):
                raise ValueError('Invalid creation identity')
            int(result['id'][len(prefix):], 16)
            if kind == 'batch':
                ids = result['upload_ids']
                if (result['mode'] not in {'independent', 'same_match'} or not isinstance(ids, list)
                        or not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids)):
                    raise ValueError('Invalid saved batch')
                for ident in ids:
                    if not isinstance(ident, str) or not ident.startswith('upload_') or len(ident) != 39:
                        raise ValueError('Invalid saved upload identity')
                    int(ident[7:], 16)
            return result
        except (ValueError, TypeError, KeyError, OverflowError):
            raise HTTPException(503, 'The saved upload creation acknowledgement could not be verified.') from None

    @staticmethod
    def save(conn, owner_id, key, kind, fingerprint, descriptor):
        if not conn.in_transaction:
            raise RuntimeError('Creation admission requires the caller write transaction')
        conn.execute('INSERT INTO upload_creation_requests VALUES (?,?,?,?,?,?)',
                     (owner_id, key, kind, fingerprint,
                      json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False), utc_now()))
