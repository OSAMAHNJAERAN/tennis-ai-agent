import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.jobs import JobStore, Settings
from src.api.product import ProductStore, ScoreCommand, replay_score


@pytest.fixture
def service(tmp_path):
    app = create_app(Settings(tmp_path, Path('config.yaml')), start_worker=False)
    with TestClient(app) as client:
        yield client, app.state.store


def manual(client, **kwargs):
    response = client.post('/api/matches', json={'source': 'manual', 'initial_server': 1, **kwargs})
    assert response.status_code == 201, response.text
    return response.json()


def command(client, match, action='point', **kwargs):
    response = client.post(f'/api/matches/{match["id"]}/score-commands', json={
        'command_id': f'command-{match["revision"]}-{action}', 'expected_revision': match['revision'],
        'action': action, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_links_draft_atomically_and_preserves_metadata(service):
    client, store = service
    metadata = {'participants': [{'name': 'A'}, {'name': 'B'}], 'tournament': 'Club final', 'match_date': '2026-09-19'}
    match = client.post('/api/matches', json={'metadata': metadata}).json()
    assert client.get('/api/matches').json() == []
    response = client.post('/api/analyses', params={'filename': 'match.mp4', 'match_id': match['id']}, content=b'\0\0\0\x18ftypisom')
    assert response.status_code == 202
    job = response.json()
    linked = client.get(f'/api/matches/by-analysis/{job["id"]}').json()
    assert linked['id'] == match['id'] and linked['metadata']['tournament'] == 'Club final'
    duplicate = client.post('/api/analyses', params={'filename': 'match.mp4', 'match_id': match['id']}, content=b'\0\0\0\x18ftypisom')
    assert duplicate.status_code == 409 and len(store.list()) == 1
    assert len(list(store.root.glob('uploads/*/original.mp4'))) == 1


def test_migration_backfills_existing_uploads_without_rewriting_results(tmp_path):
    store = JobStore(tmp_path)
    store.create('match_legacy', 'old.mp4', '.mp4', 12, 'config.yaml')
    output = store.output_dir('match_legacy')
    output.mkdir(parents=True)
    (output / 'result.json').write_text('{"unaltered":true}')
    with store.connect() as conn:
        conn.execute('DROP TABLE matches')
        conn.execute('DROP TABLE players')
        conn.execute('DROP TABLE score_commands')
        conn.execute('PRAGMA user_version=1')
    recovered = JobStore(tmp_path)
    match = ProductStore(recovered).get('match_legacy')
    assert match['metadata']['participants'][0]['name'] is None
    assert (output / 'result.json').read_text() == '{"unaltered":true}'
    assert len(ProductStore(JobStore(tmp_path)).list()) == 1


def test_manual_score_is_idempotent_and_rejects_conflicts(service):
    client, _ = service
    match = manual(client)
    url = f'/api/matches/{match["id"]}/score-commands'
    payload = {'command_id': 'point-unique', 'expected_revision': 0, 'action': 'point', 'winner': 1}
    first = client.post(url, json=payload).json()
    second = client.post(url, json=payload).json()
    assert first['score']['display_points'] == second['score']['display_points'] == ['15', '0']
    assert second['revision'] == 1
    assert client.post(url, json={**payload, 'winner': 2}).status_code == 409
    assert client.post(url, json={**payload, 'command_id': 'different-id'}).status_code == 409
    assert client.post(url, json={**payload, 'expected_revision': 1, 'command_id': 'bad-winner', 'winner': 3}).status_code == 422


def test_manual_deuce_undo_correction_pause_and_restart(service):
    client, jobs = service
    match = manual(client)
    for winner in [1, 2, 1, 2, 1, 2, 1]:
        match = command(client, match, winner=winner)
    assert match['score']['display_points'] == ['AD', '40']
    match = command(client, match, 'undo')
    assert match['score']['display_points'] == ['40', '40']
    match = command(client, match, 'correct', point_index=5, winner=1)
    assert match['score']['state']['games_p1'] == 1
    match = command(client, match, 'pause')
    assert match['status'] == 'PAUSED'
    denied = client.post(f'/api/matches/{match["id"]}/score-commands', json={'command_id': 'paused-point', 'expected_revision': match['revision'], 'action': 'point', 'winner': 1})
    assert denied.status_code == 409
    match = command(client, match, 'resume')
    restored = ProductStore(JobStore(jobs.root)).get(match['id'])
    assert restored['score'] == match['score'] and restored['revision'] == match['revision']
    assert sum(p['points_won'] for p in match['score']['statistics']) == 6


def test_tiebreak_and_match_completion_use_existing_rules(service):
    client, jobs = service
    match = manual(client)
    store = ProductStore(jobs)
    # Twelve held games, then a 7-0 tiebreak; another 6-0 set completes best-of-three.
    winners = [player for game in range(12) for player in [1 if game % 2 == 0 else 2] * 4] + [1]*7 + [1]*24
    for i, winner in enumerate(winners):
        match = store.command(match['id'], ScoreCommand(command_id=f'point-id-{i}', expected_revision=match['revision'], action='point', winner=winner))
    assert match['status'] == 'COMPLETED'
    assert match['score']['state']['winner_id'] == 1
    assert match['score']['state']['completed_sets'][0]['tie_break_p1'] == 7
    rejected = client.post(f'/api/matches/{match["id"]}/score-commands', json={'command_id': 'after-complete', 'expected_revision': match['revision'], 'action': 'point', 'winner': 2})
    assert rejected.status_code == 422
    assert store.command(match['id'], ScoreCommand(command_id='undo-final', expected_revision=match['revision'], action='undo'))['status'] == 'LIVE'


def test_best_of_five_requires_three_sets(service):
    client, jobs = service
    match = manual(client, best_of=5)
    with jobs.connect() as conn:
        row = ProductStore(jobs).raw(conn, match['id'])
    assert not replay_score(row, [{'winner': 1}]*48)['state']['match_complete']
    assert replay_score(row, [{'winner': 1}]*72)['state']['match_complete']


def test_manual_prediction_prefixes_ids_and_correction_revision(service):
    client, jobs = service
    match = manual(client)
    assert match['predictions']['status'] == 'prior_only'
    assert match['predictions']['probabilities']['match']['player_1'] == pytest.approx(.5)
    match = command(client, match, winner=1)
    original_id = match['score']['history'][0]['point_id']
    assert match['predictions']['sample_count'] == 1
    assert match['predictions']['serve_estimates'][0]['probability'] == pytest.approx(.6)
    assert match['predictions']['history'][0]['time_base'] == 'point_sequence'
    assert match['predictions']['revision'] == match['revision']
    revised = command(client, match, 'correct', point_index=0, winner=2)
    assert revised['score']['history'][0]['point_id'] == original_id
    assert revised['score']['history'][0]['provenance'] == 'human_review'
    assert revised['score']['history'][0]['corrections'][0]['previous_winner'] == 1
    assert revised['predictions']['serve_estimates'][0]['probability'] == pytest.approx(.4)
    assert revised['predictions']['revision'] == revised['revision']
    assert revised['predictions']['calibration_status'] == 'not_calibrated'
    listed = next(row for row in client.get('/api/matches').json() if row['id'] == revised['id'])
    assert listed['predictions']['history'] == []
    restored = ProductStore(JobStore(jobs.root)).get(revised['id'])
    assert restored['predictions'] == revised['predictions']


def test_profiles_do_not_merge_names_and_unconfirmed_motion_is_excluded(service):
    client, jobs = service
    first = client.post('/api/players', json={'name': 'Alex'}).json()
    other = client.post('/api/players', json={'name': 'Alex'}).json()
    assert first['id'] != other['id']
    metadata = {'participants': [{'profile_id': first['id']}, {'profile_id': other['id']}]}
    match = client.post('/api/matches', json={'metadata': metadata}).json()
    jobs.create('match_test', 'clip.mp4', '.mp4', 12, 'config.yaml', match['id'])
    jobs.finish('match_test')
    output = jobs.output_dir('match_test'); output.mkdir(parents=True)
    saved = json.dumps({'summary': {'id': 'match_test', 'durationSeconds': 7},
                        'playerMetrics': {'player_1': {'total_distance_m': 10}}})
    (output/'result.json').write_text(saved)
    assert client.get(f'/api/players/{first["id"]}').json()['movement'] == []
    updated = client.patch(f'/api/matches/{match["id"]}/metadata', json={'expected_revision': 0, 'metadata': {**match['metadata'], 'role_assignment': 'same'}}).json()
    assert len(client.get(f'/api/players/{first["id"]}').json()['movement']) == 1
    updated['metadata']['participants'][0] = {'name': 'Someone else', 'profile_id': None}
    corrected = client.patch(f'/api/matches/{match["id"]}/metadata', json={'expected_revision': updated['revision'], 'metadata': updated['metadata']}).json()
    assert corrected['metadata']['role_assignment'] == 'unconfirmed'
    assert client.get(f'/api/players/{first["id"]}').json()['movement'] == []
    assert (output/'result.json').read_text() == saved


@pytest.mark.parametrize('saved_analysis_id', [None, 'another-analysis'])
def test_confirmed_profile_role_cannot_admit_missing_or_wrong_result_identity(service, saved_analysis_id):
    from src.api.result_document import open_result
    client, jobs = service
    player = client.post('/api/players', json={'name': 'Alex'}).json()
    match = client.post('/api/matches', json={'metadata': {
        'participants': [{'profile_id': player['id']}, {'name': 'Other participant'}],
    }}).json()
    jobs.create('match_test', 'clip.mp4', '.mp4', 12, 'config.yaml', match['id'])
    jobs.finish('match_test')
    output = jobs.output_dir('match_test')
    output.mkdir(parents=True)
    summary = {'durationSeconds': 7}
    if saved_analysis_id is not None:
        summary['id'] = saved_analysis_id
    saved = json.dumps({'summary': summary,
                        'playerMetrics': {'player_1': {'total_distance_m': 10}}})
    (output / 'result.json').write_text(saved)
    response = client.patch(f'/api/matches/{match["id"]}/metadata', json={
        'expected_revision': 0,
        'metadata': {**match['metadata'], 'role_assignment': 'same'},
    })
    assert response.status_code == 200
    assert response.json()['metadata']['role_assignment'] == 'same'
    detail = client.get(f'/api/players/{player["id"]}').json()
    assert len(detail['matches']) == 1
    assert detail['movement'] == []
    with pytest.raises(ValueError, match='different analysis'):
        with open_result(output, 'match_test'):
            pytest.fail('A role confirmation admitted unbound or foreign result metrics')
    assert (output / 'result.json').read_text() == saved


def test_notes_and_metadata_revision_and_prediction_availability(service):
    client, _ = service
    match = manual(client)
    updated = client.put(f'/api/matches/{match["id"]}/notes', json={'expected_revision': 0, 'notes': 'Review recovery depth.'}).json()
    assert updated['notes'] == 'Review recovery depth.'
    assert updated['predictions']['status'] == 'prior_only'
    assert updated['predictions']['calibration_status'] == 'not_calibrated'
    assert client.put(f'/api/matches/{match["id"]}/notes', json={'expected_revision': 0, 'notes': 'stale'}).status_code == 409
    assert client.post('/api/matches', json={'source': 'manual'}).status_code == 422
    assert client.post('/api/matches', json={'metadata': {'match_type': 'doubles'}}).status_code == 422
