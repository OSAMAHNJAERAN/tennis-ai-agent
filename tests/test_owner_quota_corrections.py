"""Quota admission and publication on retained synthetic correction sources."""
from dataclasses import replace
import copy
import hashlib

from fastapi import HTTPException
from filelock import FileLock
import pytest

from src.api import correction_worker
from src.api.corrections import CorrectionStore
from src.api.resource_quotas import ResourceQuotas
from tests.test_correction_worker import queued_correction, assert_original_unchanged
from tests.test_corrected_shot_rebuild import no_model_imports


def test_correction_idempotence_precedes_full_pending_quota_and_new_source_parsing(queued_correction, monkeypatch):
    case = queued_correction
    settings = replace(case.settings, max_owner_pending_jobs=1)
    quotas = ResourceQuotas(case.jobs, settings)
    commands = CorrectionStore(case.jobs, quotas=quotas)
    monkeypatch.setattr(commands, '_validate_source_command', lambda *args: pytest.fail('full queue expanded source'))
    owner = case.fixture.job['owner_id']
    existing = commands.submit('fixture', owner, case.payload, authorization_method='AUTHENTICATED_OWNER')
    assert existing['id'] == case.claimed['id']
    changed = copy.deepcopy(case.payload)
    changed['correction_id'] = 'a-different-owner-command'
    with pytest.raises(HTTPException) as rejected:
        commands.submit('fixture', owner, changed, authorization_method='AUTHENTICATED_OWNER')
    assert rejected.value.status_code == 429
    assert len(commands.list('fixture', owner)) == 1


def test_review_source_expansion_is_owned_counted_and_serialized_even_when_pending_full(queued_correction):
    case = queued_correction
    owner = case.fixture.job['owner_id']
    quotas = ResourceQuotas(case.jobs, replace(case.settings, max_owner_pending_jobs=1))
    commands = CorrectionStore(case.jobs, quotas=quotas)
    with case.jobs.connect() as conn:
        before = quotas.snapshot(conn, owner)
    with commands._source(case.fixture.snapshot) as source:
        assert len(source.source_frames) == case.fixture.snapshot.frames
        with case.jobs.connect() as conn:
            expanded = quotas.snapshot(conn, owner)
        assert expanded['stored_bytes'] > before['stored_bytes']
        assert expanded['pending_jobs'] == 1
        assert list((case.fixture.directory/'.correction-admission').glob('admission-*'))
        with pytest.raises(HTTPException) as blocked:
            with commands._source(case.fixture.snapshot):
                pytest.fail('concurrent source parser admitted')
        assert blocked.value.status_code == 429
    with case.jobs.connect() as conn:
        assert quotas.snapshot(conn, owner)['stored_bytes'] == before['stored_bytes']


def test_storage_growth_at_final_correction_commit_keeps_old_generation(queued_correction, monkeypatch):
    case = queued_correction
    from src.scoring import corrected_publication
    settings = replace(case.settings, max_owner_storage_bytes=64*1024**2)
    # The human has not released host resource probing. Inject this exact
    # admission boundary; reconstruction, codecs and publication are real.
    monkeypatch.setattr(correction_worker, '_resources', lambda *args: None)
    original_publish = corrected_publication.publish_corrected_result
    callbacks = []

    def grow_before_commit(*args, **kwargs):
        original_check = kwargs['before_commit']

        def grow_and_check():
            callbacks.append(True)
            retained = case.fixture.directory/'quota_growth_diagnostic.bin'
            with retained.open('wb') as stream:
                stream.truncate(settings.max_owner_storage_bytes)
            original_check()

        kwargs['before_commit'] = grow_and_check
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(corrected_publication, 'publish_corrected_result', grow_before_commit)
    correction_worker.process_correction(case.jobs, case.claimed, settings)
    final = case.commands.get(case.claimed['id'], internal=True)
    assert callbacks == [True]
    assert final['status'] == 'FAILED' and final['error'] == 'Your storage limit has been reached.'
    assert_original_unchanged(case)
    quotas = ResourceQuotas(case.jobs, settings)
    with case.jobs.connect() as conn:
        usage = quotas.snapshot(conn, case.fixture.job['owner_id'])
    assert usage['stored_bytes'] > settings.max_owner_storage_bytes
    assert usage['pending_corrections'] == 0
