"""SQLite queue flags must satisfy the browser's boolean job contract."""
from pathlib import Path

from src.api.jobs import JobStore, public_job


def test_saved_job_cancel_flag_is_boolean_before_and_after_cancellation(tmp_path):
    store = JobStore(tmp_path)
    job = store.create('match_contract', 'test.mp4', '.mp4', 20, Path('candidate.yaml'), owner_id='test_owner')
    public = public_job(store.get(job['id']))
    assert public['cancel_requested'] is False and public['cancelled'] is False
    cancelled = public_job(store.cancel(job['id'], 'test_owner'))
    assert cancelled['cancel_requested'] is True and cancelled['cancelled'] is True
    assert cancelled['status'] == 'FAILED'
    assert not {'owner_id', 'config_path', 'extension'} & public.keys()
