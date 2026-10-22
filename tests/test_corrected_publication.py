"""Actual owner-authorized complete CAS after private correction owners close.

All source images, observations and authorizations are explicit local test
fixtures. Passing publication checks establishes no model or physical accuracy.
"""
from dataclasses import replace
import copy
import hashlib
import json

import pytest

from tests.test_corrected_preparation import build_complete_base, complete_inputs, prepare
from tests.test_corrected_shot_rebuild import no_model_imports, advance_equivalent_generation
from tests.test_corrected_rule_rebuild import unchanged
from tests.test_corrected_exports import read, digests


OBSERVATIONS = []


def ready(tmp_path, *, optional=False):
    from src.scoring.corrected_publication import describe_prepared_correction
    fixture = build_complete_base(tmp_path, with_optional=optional)
    fixture.request['reviewer_id'] = fixture.job['owner_id']
    fixture.request['reason'] = 'Reviewed target <script>bounded</script>'
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            descriptor = describe_prepared_correction(prepared)
            fixture.private_hashes = digests(prepared.directory)
        assert prepared.storage._connection is None
    unchanged(fixture)
    return fixture, descriptor


def authorization(fixture, descriptor, method='AUTHENTICATED_OWNER'):
    return {'schema_version': 1, 'analysis_id': 'fixture', 'owner_id': fixture.job['owner_id'],
        'reviewer_id': fixture.job['owner_id'], 'correction_id': fixture.request['correction_id'],
        'request_sha256': descriptor.request_sha256, 'authorized_at': '2026-10-18T04:00:00+00:00',
        'authorization_method': method}


@pytest.mark.parametrize('method,optional', [('AUTHENTICATED_OWNER', False),
    ('EXPLICIT_LOCAL_OWNER', False), ('AUTHENTICATED_OWNER', True)])
def test_actual_complete_authorized_cas_preserves_old_generation_and_delivers_consistent_reports(tmp_path, method, optional):
    from src.api.analysis_generation import resolve_generation
    from src.api.result_document import ResultDocument
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path, optional=optional)
    receipt = authorization(fixture, descriptor, method)
    callbacks = []
    token = publish_corrected_result(descriptor, fixture.directory, authorization=receipt,
                                    before_commit=lambda: callbacks.append('owner checked'))
    assert callbacks == ['owner checked']
    snapshot = resolve_generation(fixture.directory)
    assert snapshot.generation == token and snapshot.analysis_revision == 1
    assert snapshot.file_sha256('source.mp4') == fixture.source_sha
    assert snapshot.resolve_file('corrected_preparation_manifest.json') is None
    assert snapshot.resolve_file('corrected_delivery_manifest.json') is not None
    with ResultDocument(fixture.directory, 'fixture', generation=token) as document:
        result = document.value
        correction = result['analysisCorrection']
        assert correction['status'] == 'OWNER_REVIEWED_RESULT'
        assert correction['authorizationStatus'] == ('OWNER_AUTHORIZED' if method == 'AUTHENTICATED_OWNER' else method)
        assert correction['authorizationReceipt'] == receipt
        assert correction['publicationPolicy'] == 'ATOMIC_COMPLETE_GENERATION'
        assert correction['privatePreparationManifestSha256'] == descriptor.manifest_sha256
        assert 'published' not in correction and correction['independentAccuracyValidated'] is False
        assert correction['review']['reason'] == fixture.request['reason']
        assert result['pointAnalysis']['summary']['totalPointsPlayed'] == 1
        assert result['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 1
        assert result['nativeForecast']['current']['sample_count'] == 1
        assert result['nativeForecast']['current']['serve_estimates'][0]['probability'] == .4
        assert result['analysisRevision'] == 1
        execution = read(snapshot.resolve_file('correction_execution.json'))
        assert correction['execution'] == execution
        assert execution['new_vision_inference'] is False and execution['pose_provider_calls'] == 0
        assert execution['pose_prediction_calls'] == 0
        assert execution['selected_player_checkpoint_sha256'] is None
        assert read(snapshot.resolve_file('correction_authorization.json')) == receipt
        assert read(snapshot.resolve_file('report.json'))['analysisCorrection'] == correction
        html = snapshot.resolve_file('readable_report.html').read_text(encoding='utf-8')
        assert 'Owner-authorized correction review' in html
        assert ('authenticated analysis owner' in html) == (method == 'AUTHENTICATED_OWNER')
        assert ('explicit local owner' in html) == (method == 'EXPLICIT_LOCAL_OWNER')
        assert '&lt;script&gt;bounded&lt;/script&gt;' in html and '<script>bounded</script>' not in html
        assert 'Private correction review' not in html
        if optional:
            credit = read(snapshot.resolve_file('annotated_web_credit.json'))
            assert credit == read(descriptor.directory/'annotated_web_credit.json')
            assert result['mediaSource']['modifications'] == credit['modifications']
            assert 'STALE_OLD' not in html
        assert len(snapshot.resolve_file('telemetry.csv').read_text(encoding='utf-8').splitlines()) == 13
        presented = document.presented('fixture')
        assert presented['analysisGeneration'] == token
        assert presented['annotatedVideoUrl'].endswith('?generation='+token)
        assert all('?generation='+token in url for url in presented['exports'].values())
    with ResultDocument(fixture.directory, 'fixture', generation=fixture.snapshot.generation) as old:
        assert old.value['analysisRevision'] == 0
        assert old.value['pointAnalysis']['summary']['totalPointsPlayed'] == 0
        assert old.value['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 0
    assert digests(fixture.snapshot.files_directory) == fixture.manifest
    assert digests(descriptor.directory) == fixture.private_hashes
    seal = read(snapshot.resolve_file('corrected_delivery_manifest.json'))
    assert seal['private_preparation_manifest_sha256'] == descriptor.manifest_sha256
    assert seal['request_sha256'] == descriptor.request_sha256
    assert 'published' not in seal
    for name, record in seal['artifacts'].items():
        path = snapshot.resolve_file(name)
        assert record == {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    OBSERVATIONS.append({'actual_authorized_cas': {'method': method, 'new_revision': 1,
        'old_revision': 0, 'corrected_double_faults': 1, 'private_bundle_unchanged': True,
        'independent_accuracy_validated': False}})


@pytest.mark.parametrize('damage', ['missing', 'foreign_owner', 'foreign_request', 'foreign_analysis', 'bool_schema', 'unknown_method'])
def test_owner_receipt_must_match_exact_original_reviewed_command(tmp_path, damage):
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path)
    receipt = authorization(fixture, descriptor)
    if damage == 'missing':
        receipt.pop('authorized_at')
    elif damage == 'foreign_owner':
        receipt['owner_id'] = receipt['reviewer_id'] = 'different-owner'
    elif damage == 'foreign_request':
        receipt['request_sha256'] = '0'*64
    elif damage == 'foreign_analysis':
        receipt['analysis_id'] = 'other-analysis'
    elif damage == 'bool_schema':
        receipt['schema_version'] = True
    else:
        receipt['authorization_method'] = 'CLAIMED_IN_PRIVATE_REQUEST'
    with pytest.raises(ValueError, match='authorization|owner|schema|receipt'):
        publish_corrected_result(descriptor, fixture.directory, authorization=receipt)
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes


@pytest.mark.parametrize('name', ['metrics.json', 'result.json', 'summary.json', 'DATABASE'])
def test_final_authorization_callback_cannot_change_the_complete_candidate(tmp_path, name):
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path)
    changed = []
    def corrupt_candidate():
        candidates = [path for path in (fixture.directory/'.result').glob('*.files')
                      if (path/'corrected_delivery_manifest.json').is_file()]
        assert len(candidates) == 1
        path = candidates[0].with_suffix('.sqlite3') if name == 'DATABASE' else candidates[0]/name
        path.write_bytes(path.read_bytes()+b'SYNTHETIC_FINAL_AUTH_CALLBACK_MUTATION')
        changed.append(path.name)
    with pytest.raises(ValueError, match='candidate.*differs|candidate database changed'):
        publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor), before_commit=corrupt_candidate)
    assert len(changed) == 1
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes
    assert not list(fixture.scratch.rglob('corrected_delivery_manifest.json'))


@pytest.mark.parametrize('writer', ['readable_report_chunks', 'telemetry_csv_chunks'])
def test_delivery_report_failure_preserves_complete_old_result_and_private_evidence(tmp_path, monkeypatch, writer):
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = getattr(module, writer)
    def partial(*args, **kwargs):
        chunks = iter(native(*args, **kwargs))
        try:
            yield next(chunks)
            raise OSError('SYNTHETIC_DELIVERY_REPORT_FAILURE')
        finally:
            close = getattr(chunks, 'close', None)
            if close is not None:
                close()
    monkeypatch.setattr(module, writer, partial)
    with pytest.raises(OSError, match='SYNTHETIC_DELIVERY_REPORT_FAILURE'):
        module.publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor))
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes
    assert not list(fixture.scratch.rglob('corrected_delivery_manifest.json'))


@pytest.mark.parametrize('name', ['report.json', 'native_rule_replay.json', 'source.mp4', 'corrected_preparation_manifest.json'])
def test_changed_prepared_bytes_cannot_be_published_even_after_native_owners_close(tmp_path, name):
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path)
    path = descriptor.directory/name
    original = path.read_bytes()
    try:
        path.write_bytes(b'SYNTHETIC_CHANGED_PRIVATE_PREPARATION')
        with pytest.raises((ValueError, json.JSONDecodeError), match='prepared|preparation|manifest|artifact|changed|Expecting'):
            publish_corrected_result(descriptor, fixture.directory, authorization=authorization(fixture, descriptor))
    finally:
        path.write_bytes(original)
    unchanged(fixture)


def test_failed_final_cancel_check_preserves_old_pointer_and_private_artifacts(tmp_path):
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path)
    def cancelled():
        raise RuntimeError('SYNTHETIC_DURABLE_OWNER_CANCEL')
    with pytest.raises(RuntimeError, match='SYNTHETIC_DURABLE_OWNER_CANCEL'):
        publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor), before_commit=cancelled)
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes
    assert not list(fixture.scratch.rglob('corrected_delivery_manifest.json'))


def test_actual_new_base_generation_prevents_stale_correction_publication(tmp_path):
    from src.api.analysis_generation import resolve_generation
    from src.scoring.corrected_publication import publish_corrected_result
    fixture, descriptor = ready(tmp_path)
    advanced = advance_equivalent_generation(fixture)
    with pytest.raises(ValueError, match='current base|generation'):
        publish_corrected_result(descriptor, fixture.directory, authorization=authorization(fixture, descriptor))
    assert resolve_generation(fixture.directory).generation == advanced
    assert digests(fixture.snapshot.files_directory) == fixture.manifest


def test_lost_acknowledgement_is_reconciled_from_actual_current_generation_and_retry_is_idempotent(tmp_path, monkeypatch):
    from src.api.analysis_generation import resolve_generation
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    receipt = authorization(fixture, descriptor)
    native = module.publish_result
    committed = []
    def lost(*args, **kwargs):
        token = native(*args, **kwargs)
        committed.append(token)
        raise OSError('SYNTHETIC_ACK_LOST_AFTER_CURRENT_POINTER_COMMIT')
    monkeypatch.setattr(module, 'publish_result', lost)
    token = module.publish_corrected_result(descriptor, fixture.directory, authorization=receipt)
    assert committed == [token] and resolve_generation(fixture.directory).generation == token
    inventory = sorted(path.name for path in (fixture.directory/'.result').iterdir())
    assert module.publish_corrected_result(descriptor, fixture.directory, authorization=receipt) == token
    assert sorted(path.name for path in (fixture.directory/'.result').iterdir()) == inventory
    assert len(committed) == 1


def test_transient_delivery_copy_corruption_in_real_generation_writer_rejects_candidate(tmp_path, monkeypatch):
    from src.api import analysis_generation
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = analysis_generation._copy
    changed = []
    def copying(source, destination, **kwargs):
        if source.name == 'metrics.json' and source.parent.name.startswith('corrected-delivery-'):
            original = source.read_bytes()
            source.write_bytes(b'SYNTHETIC_TRANSIENT_DELIVERY_COPY')
            try:
                result = native(source, destination, **kwargs)
                changed.append(True)
                return result
            finally:
                source.write_bytes(original)
        return native(source, destination, **kwargs)
    monkeypatch.setattr(analysis_generation, '_copy', copying)
    with pytest.raises(ValueError, match='candidate copy differs'):
        module.publish_corrected_result(descriptor, fixture.directory, authorization=authorization(fixture, descriptor))
    assert changed == [True]
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes


def test_descriptor_cannot_be_issued_from_a_closed_prepared_owner(tmp_path):
    from src.scoring.corrected_publication import describe_prepared_correction
    fixture = build_complete_base(tmp_path)
    with complete_inputs(fixture) as (rebuilt, shots, forecast, exports, rendered):
        with prepare(rebuilt, shots, forecast, exports, rendered, fixture.scratch, fixture.job) as prepared:
            pass
        with pytest.raises(RuntimeError, match='open|closed'):
            describe_prepared_correction(prepared)
    unchanged(fixture)


@pytest.mark.parametrize('damage', ['foreign_source', 'unused_with_model', 'wrong_selected_checkpoint',
    'wrong_selected_checkpoint_without_prediction', 'false_inference', 'unreported_inference',
    'prediction_without_provider', 'boolean_prediction_counter'])
def test_delivery_execution_cannot_claim_a_different_source_or_selected_model(tmp_path, damage):
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    _, manifest, _ = module._prepared(descriptor)
    execution = module._execution(descriptor, manifest, None)
    if damage == 'foreign_source':
        execution['source_sha256'] = '0'*64
    elif damage == 'unused_with_model':
        execution['selected_player_checkpoint_sha256'] = 'a'*64
    elif damage.startswith('wrong_selected_checkpoint'):
        predicted = damage == 'wrong_selected_checkpoint'
        execution.update(new_vision_inference=predicted, pose_provider_calls=1, pose_prediction_calls=int(predicted),
            selected_player_checkpoint_sha256='f'*64,
            selected_player_checkpoint_filename='different-player.pt',
            runtime_versions={'fixture': 'EXPLICIT_SYNTHETIC_FORGED_EXECUTION'},
            input_spec={'keypoints': 17}, architecture='EXPLICIT_SYNTHETIC_WRONG_CHECKPOINT')
    elif damage == 'false_inference':
        execution['new_vision_inference'] = True
    elif damage == 'unreported_inference':
        execution['pose_provider_calls'] = execution['pose_prediction_calls'] = 1
    elif damage == 'prediction_without_provider':
        execution.update(new_vision_inference=True, pose_prediction_calls=1)
    else:
        execution['pose_prediction_calls'] = True
    with pytest.raises(ValueError, match='execution|unused|checkpoint'):
        module.publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor), execution=execution)
    unchanged(fixture)


@pytest.mark.parametrize('predictions', [0, 2])
def test_delivery_distinguishes_provider_invocation_from_actual_prediction_calls(tmp_path, predictions):
    from src.api.result_document import ResultDocument
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    _, manifest, _ = module._prepared(descriptor)
    execution = module._execution(descriptor, manifest, None)
    selected = fixture.inference_provenance['checkpointHashes']['players']
    # This synthetic caller receipt exercises metadata delivery. No checkpoint
    # is loaded and these test counters are not vision execution evidence.
    execution.update(pose_provider_calls=1, pose_prediction_calls=predictions,
        new_vision_inference=predictions > 0,
        selected_player_checkpoint_sha256=selected['sha256'],
        selected_player_checkpoint_filename=selected['filename'],
        runtime_versions={'fixture': 'EXPLICIT_SYNTHETIC_CALLER_RECEIPT_ONLY'},
        input_spec={'fixture': 'EXPLICIT_SYNTHETIC_CALLER_RECEIPT_ONLY'},
        architecture={'fixture': 'EXPLICIT_SYNTHETIC_CALLER_RECEIPT_ONLY'})
    token = module.publish_corrected_result(descriptor, fixture.directory,
        authorization=authorization(fixture, descriptor), execution=execution)
    with ResultDocument(fixture.directory, 'fixture', generation=token) as document:
        delivered = document.value['analysisCorrection']['execution']
        assert delivered == execution
        assert delivered['new_vision_inference'] is (predictions > 0)
        assert delivered['pose_provider_calls'] == 1
        assert delivered['pose_prediction_calls'] == predictions
        assert document.value['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 1
        assert read(document.snapshot.resolve_file('correction_execution.json')) == execution
    assert digests(fixture.snapshot.files_directory) == fixture.manifest
    assert digests(descriptor.directory) == fixture.private_hashes


@pytest.mark.parametrize('damage', ['database', 'generation_manifest'])
def test_pre_callback_native_seal_tampering_cannot_commit_an_unreadable_current_result(tmp_path, monkeypatch, damage):
    from src.api import analysis_generation
    from src.api.result_document import ResultDocument
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = analysis_generation._copy
    changed = []
    def copying(source, destination, **kwargs):
        result = native(source, destination, **kwargs)
        if (source.name == 'result.json' and source.parent.suffix == '.files'
                and destination.parent == fixture.directory):
            if damage == 'database':
                path = source.parent.with_suffix('.sqlite3')
                with path.open('ab') as stream:
                    stream.write(b'EXPLICIT_SYNTHETIC_PRE_CALLBACK_DATABASE_DAMAGE')
            else:
                manifests = [path for path in source.parent.parent.glob('*.generation')
                             if read(path)['filesDirectory'] == source.parent.name]
                assert len(manifests) == 1
                path = manifests[0]
                payload = read(path)
                payload['schemaVersion'] = 'EXPLICIT_SYNTHETIC_PRE_CALLBACK_MANIFEST_DAMAGE'
                path.write_text(json.dumps(payload), encoding='utf-8')
            changed.append(path.name)
        return result
    monkeypatch.setattr(analysis_generation, '_copy', copying)
    with pytest.raises(ValueError, match='candidate database changed|candidate generation manifest differs'):
        module.publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor))
    assert len(changed) == 1
    unchanged(fixture)
    with ResultDocument(fixture.directory, 'fixture') as old:
        assert old.value['analysisRevision'] == 0
    assert digests(descriptor.directory) == fixture.private_hashes
    assert not list(fixture.scratch.rglob('corrected_delivery_manifest.json'))


def test_valid_sqlite_root_tampering_before_initial_database_hash_is_rejected_against_report(tmp_path, monkeypatch):
    import sqlite3
    from src.api.result_storage import ResultStorage
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = ResultStorage.publish
    changed = []
    def publishing(owner, destination):
        path = native(owner, destination)
        # Preserve a valid SQLite database and its series references while
        # changing a normalized scalar before the first retained DB hash.
        with sqlite3.connect(path) as connection:
            payload = connection.execute('SELECT payload FROM root WHERE slot=1').fetchone()[0]
            marker = 'EXPLICIT_SYNTHETIC_CAPTURED_INFERENCE'
            assert payload.count(marker) == 1
            payload = payload.replace(marker, 'EXPLICIT_SYNTHETIC_TAMPERED_CLOSED_ROOT')
            connection.execute('UPDATE root SET payload=? WHERE slot=1', (payload,))
            connection.commit()
        changed.append(path.name)
        return path
    monkeypatch.setattr(ResultStorage, 'publish', publishing)
    with pytest.raises(ValueError, match='closed database root differs from delivery report'):
        module.publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor))
    assert len(changed) == 1
    unchanged(fixture)
    assert digests(descriptor.directory) == fixture.private_hashes


@pytest.mark.parametrize('when', ['native_write', 'authorization_callback'])
def test_temporary_pointer_is_bound_to_exact_candidate_before_and_after_authorization(tmp_path, monkeypatch, when):
    from src.api import analysis_generation
    from src.api.result_document import ResultDocument
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = analysis_generation._write_bytes
    changed = []

    def corrupt(path):
        path.write_bytes(b'{"schemaVersion":2,"generation":"'+b'f'*64+b'"}\n')
        changed.append(path.name)

    def writing(path, payload, **kwargs):
        value = native(path, payload, **kwargs)
        if when == 'native_write' and path.name.endswith('.store.tmp'):
            corrupt(path)
        return value

    def authorization_callback():
        if when == 'authorization_callback':
            pointers = list((fixture.directory/'.result').glob('*.store.tmp'))
            assert len(pointers) == 1
            corrupt(pointers[0])

    monkeypatch.setattr(analysis_generation, '_write_bytes', writing)
    with pytest.raises(ValueError, match='candidate pointer differs from trusted generation'):
        module.publish_corrected_result(descriptor, fixture.directory,
            authorization=authorization(fixture, descriptor), before_commit=authorization_callback)
    assert len(changed) == 1
    unchanged(fixture)
    with ResultDocument(fixture.directory, 'fixture') as previous:
        assert previous.value['analysisRevision'] == 0
    assert digests(descriptor.directory) == fixture.private_hashes
    assert not list(fixture.scratch.rglob('corrected_delivery_manifest.json'))


@pytest.mark.parametrize('interruption', [KeyboardInterrupt, SystemExit])
def test_critical_post_commit_interruption_propagates_while_actual_commit_stays_readable(tmp_path, monkeypatch, interruption):
    from src.api.analysis_generation import resolve_generation
    from src.api.result_document import ResultDocument
    from src.scoring import corrected_publication as module
    fixture, descriptor = ready(tmp_path)
    native = module.publish_result
    committed = []
    def interrupted(*args, **kwargs):
        token = native(*args, **kwargs)
        committed.append(token)
        raise interruption('EXPLICIT_SYNTHETIC_CRITICAL_POST_COMMIT_INTERRUPTION')
    monkeypatch.setattr(module, 'publish_result', interrupted)
    receipt = authorization(fixture, descriptor)
    with pytest.raises(interruption, match='EXPLICIT_SYNTHETIC_CRITICAL_POST_COMMIT_INTERRUPTION'):
        module.publish_corrected_result(descriptor, fixture.directory, authorization=receipt)
    assert len(committed) == 1
    token = resolve_generation(fixture.directory).generation
    assert token == committed[0]
    with ResultDocument(fixture.directory, 'fixture', generation=token) as current:
        assert current.value['analysisRevision'] == 1
        assert current.value['pointAnalysis']['serveByPlayer']['player_1']['doubleFaults'] == 1
    assert module.publish_corrected_result(descriptor, fixture.directory, authorization=receipt) == token
    assert len(committed) == 1
    assert list(fixture.scratch.rglob('corrected_delivery_manifest.json'))
    assert digests(fixture.snapshot.files_directory) == fixture.manifest
    assert digests(descriptor.directory) == fixture.private_hashes
