"""Only committed, revision-consistent, hash-bound generations are readable.

All files and rules are synthetic. The tests use actual generation storage and
ASGI routes; no worker, video decoder or model runs.
"""
import json

import pytest

from tests.test_generation_http_contract import service, publish, pinned


def candidate(service, *, label, raw_revision=1, point_revision=None,
              before_commit=None, omit_annotated=False):
    from src.api.analysis_generation import publish_analysis_generation
    from src.api.result_storage import ResultStorage
    from src.api.results import build_result
    directory = service.store.output_dir(service.analysis_id)
    with ResultStorage(directory / '.result') as storage:
        value = build_result(directory, {**service.job, 'filename': label}, _storage=storage)
        value['analysisRevision'] = raw_revision
        if point_revision is not None:
            value['pointAnalysis'] = {'analysisRevision': point_revision}
        files = {path.name: path for path in directory.iterdir()
                 if path.is_file() and path.name not in {'result.json', 'summary.json'}
                 and path.suffix in {'.json', '.jsonl', '.yaml', '.mp4'}
                 and not (omit_annotated and path.name == 'annotated_web.mp4')}
        files['original.mp4'] = service.store.upload_dir(service.analysis_id) / 'original.mp4'
        return publish_analysis_generation(storage, value, directory, files=files,
            analysis_revision=1, before_commit=before_commit)


def test_final_verification_rejection_never_creates_readable_committed_history(service):
    first = publish(service, 'Synthetic committed baseline')
    generations = first.directory / '.result'
    old_pointer = (generations / 'current.store').read_bytes()
    old_manifests = {path.stem for path in generations.glob('*.generation')}
    old_report = service.alice.get(service.base + '/download/report').content
    rejected = []

    def reject_before_commit():
        # Backend diagnostics may know a sealed token without its publication
        # ever succeeding. Random token secrecy is not commit-state evidence.
        rejected.extend(path.stem for path in generations.glob('*.generation')
                        if path.stem not in old_manifests)
        raise OSError('Synthetic final verification rejected publication')

    with pytest.raises(OSError, match='final verification rejected'):
        candidate(service, label='UNCOMMITTED synthetic candidate', before_commit=reject_before_commit)
    assert len(rejected) == 1
    assert (generations / 'current.store').read_bytes() == old_pointer
    assert service.store.get(service.analysis_id)['status'] == 'COMPLETED'
    assert service.alice.get(service.base + '/download/report').content == old_report
    assert service.alice.get(service.base + '/result').json()['analysisGeneration'] == first.token
    for suffix in ('/result', '/media/annotated', '/download/report', '/download/original'):
        response = service.alice.get(pinned(service, suffix, rejected[0]))
        assert response.status_code in {404, 503}
        assert b'UNCOMMITTED synthetic candidate' not in response.content


@pytest.mark.parametrize('raw_revision,point_revision', [
    (0, None), (True, None), ('1', None), (1, 0), (1, True), (1, '1'),
])
def test_present_raw_and_point_revisions_must_match_publication_revision(service, raw_revision, point_revision):
    first = publish(service, 'Synthetic consistent baseline')
    pointer = first.directory / '.result/current.store'
    previous_pointer = pointer.read_bytes()
    previous_report = service.alice.get(service.base + '/download/report').content
    with pytest.raises(ValueError):
        candidate(service, label='Inconsistent synthetic revision', raw_revision=raw_revision,
                  point_revision=point_revision)
    assert pointer.read_bytes() == previous_pointer
    assert service.alice.get(service.base + '/download/report').content == previous_report
    assert service.alice.get(service.base + '/result').json()['analysisGeneration'] == first.token


@pytest.mark.parametrize('case', ['corrupt_declared', 'untracked_undeclared'])
def test_html_images_validate_declared_generation_asset_and_ignore_untracked_file(service, monkeypatch, case):
    from src.api import reporting
    from src.api.analysis_generation import resolve_generation
    from src.api.result_document import ResultDocument
    first = publish(service, 'Synthetic hash-bound HTML generation')
    if case == 'corrupt_declared':
        token = first.token
        video = first.assets / 'annotated_web.mp4'
        video.write_bytes(bytes([first.annotated[0] ^ 1]) + first.annotated[1:])
    else:
        token = candidate(service, label='Synthetic generation without declared annotation',
                          omit_annotated=True)
        snapshot = resolve_generation(first.directory, token)
        assert snapshot.resolve_file('annotated_web.mp4') is None
        (snapshot.files_directory / 'annotated_web.mp4').write_bytes(b'UNTRACKED synthetic annotation')
    calls = []
    def images(directory, result):
        calls.append(directory)
        assert directory is None, 'An undeclared video must never reach image extraction'
        return ''
    monkeypatch.setattr(reporting, 'evidence_images', images)
    closed = []
    actual_close = ResultDocument.close
    def close(document):
        actual_close(document)
        closed.append(document)
    monkeypatch.setattr(ResultDocument, 'close', close)
    response = service.alice.get(pinned(service, '/download/readable-report', token))
    if case == 'corrupt_declared':
        assert response.status_code == 503 and calls == []
    else:
        assert response.status_code == 200 and calls == [None]
        assert 'UNTRACKED synthetic annotation' not in response.text
    assert closed and all(document.closed for document in closed)
    assert all(document.storage is None or document.storage._connection is None for document in closed)
