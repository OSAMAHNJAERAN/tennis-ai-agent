"""Product records and manual scoring, separate from immutable inference evidence."""
import json
import sqlite3
import uuid
from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field, ConfigDict, model_validator

from src.api.jobs import utc_now, LOCAL_OWNER
from src.api.auth import current_owner
from src.scoring.match_state import MatchState, MatchFormat
from src.scoring.scoring_rules import TennisScoringRules
from src.prediction.tennis import replay_predictions


def migrate_product(conn):
    version = conn.execute('PRAGMA user_version').fetchone()[0]
    if version > 4:
        raise RuntimeError('This database requires a newer product server.')
    if version < 2:
        conn.execute('''CREATE TABLE IF NOT EXISTS players (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS matches (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, analysis_id TEXT UNIQUE,
            metadata TEXT NOT NULL DEFAULT '{}', notes TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 0, live_config TEXT,
            points TEXT NOT NULL DEFAULT '[]', paused INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS score_commands (
            match_id TEXT NOT NULL, command_id TEXT NOT NULL, payload TEXT NOT NULL,
            created_at TEXT NOT NULL, PRIMARY KEY(match_id, command_id))''')
        conn.execute('''INSERT OR IGNORE INTO matches(id, source, analysis_id, created_at, updated_at)
            SELECT id, 'upload', id, created_at, COALESCE(completed_at, created_at) FROM analyses''')
        conn.execute('PRAGMA user_version=2')


def attach_analysis(conn, analysis_id, match_id=None, owner_id=LOCAL_OWNER):
    now = utc_now()
    if match_id:
        result = conn.execute("UPDATE matches SET analysis_id=?,updated_at=? WHERE id=? AND owner_id=? AND source='upload' AND analysis_id IS NULL",
                              (analysis_id, now, match_id, owner_id))
        if result.rowcount != 1:
            raise HTTPException(409, 'This upload draft is already used or unavailable. Refresh before retrying.')
    else:
        conn.execute("INSERT INTO matches(id,source,analysis_id,created_at,updated_at,owner_id) VALUES (?,'upload',?,?,?,?)",
                     (analysis_id, analysis_id, now, now, owner_id))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class Participant(StrictModel):
    name: str | None = Field(default=None, max_length=120)
    profile_id: str | None = None


class MatchMetadata(StrictModel):
    participants: list[Participant] = Field(default_factory=lambda: [Participant(), Participant()], min_length=2, max_length=2)
    tournament: str | None = Field(default=None, max_length=160)
    match_date: date | None = None
    match_type: Literal['singles'] = 'singles'
    role_assignment: Literal['unconfirmed', 'same', 'swapped'] = 'unconfirmed'
    result: str | None = Field(default=None, max_length=120)


class CreateMatch(StrictModel):
    source: Literal['upload', 'manual'] = 'upload'
    metadata: MatchMetadata = Field(default_factory=MatchMetadata)
    best_of: Literal[3, 5] = 3
    initial_server: Literal[1, 2] | None = None

    @model_validator(mode='after')
    def manual_server(self):
        if self.source == 'manual' and self.initial_server is None:
            raise ValueError('Select the initial server for manual scoring.')
        return self


class MetadataEdit(StrictModel):
    expected_revision: int = Field(ge=0)
    metadata: MatchMetadata


class NoteEdit(StrictModel):
    expected_revision: int = Field(ge=0)
    notes: str = Field(max_length=10000)


class CreatePlayer(StrictModel):
    name: str = Field(min_length=1, max_length=120)


class ScoreCommand(StrictModel):
    command_id: str = Field(min_length=8, max_length=100)
    expected_revision: int = Field(ge=0)
    action: Literal['point', 'undo', 'correct', 'pause', 'resume']
    winner: Literal[1, 2] | None = None
    point_index: int | None = Field(default=None, ge=0)


def replay_score(row, points=None):
    config = json.loads(row['live_config'])
    state = MatchState(match_id=row['id'], format=MatchFormat.BEST_OF_3 if config['best_of'] == 3 else MatchFormat.BEST_OF_5,
                       server_id=config['initial_server'], receiver_id=3-config['initial_server'],
                       initial_server_id=config['initial_server'], score_source='user_manual')
    initial_state = state
    points = points if points is not None else json.loads(row['points'])
    stats = [{'points_won': 0, 'service_points_won': 0, 'service_points': 0, 'return_points_won': 0, 'return_points': 0} for _ in range(2)]
    history = []
    for index, point in enumerate(points):
        if state.match_complete:
            raise HTTPException(422, 'This correction would leave points after match completion. Undo those points first.')
        winner, server = point['winner'], state.server_id
        stats[winner-1]['points_won'] += 1
        stats[server-1]['service_points'] += 1
        stats[2-server]['return_points'] += 1
        stats[winner-1]['service_points_won' if winner == server else 'return_points_won'] += 1
        state = TennisScoringRules.award_point(state, winner, 'Manually scored point')
        history.append({**point, 'point_id': point.get('point_id', f'legacy-point-{index}'),
                        'provenance': point.get('provenance', 'manual'),
                        'server': server, 'score': [state.get_points_display_p1(), state.get_points_display_p2()],
                        'games': [state.games_p1, state.games_p2], 'sets': [state.sets_won_p1, state.sets_won_p2]})
    predictions = replay_predictions(initial_state, points, revision=row['revision'])
    return {'state': {**state.to_dict(), 'games_p1': state.games_p1, 'games_p2': state.games_p2, 'sets_won_p1': state.sets_won_p1, 'sets_won_p2': state.sets_won_p2}, 'display_points': [state.get_points_display_p1(), state.get_points_display_p2()],
            'statistics': stats, 'history': history,
            'predictions': {**predictions['current'], 'history': predictions['history']}}


class ProductStore:
    def __init__(self, jobs):
        self.jobs = jobs

    def raw(self, conn, match_id, owner_id=LOCAL_OWNER):
        row = conn.execute('SELECT * FROM matches WHERE id=? AND owner_id=?', (match_id, owner_id)).fetchone()
        if row is None:
            raise HTTPException(404, 'Match not found.')
        return dict(row)

    def serialize(self, row, detail=True):
        metadata = MatchMetadata.model_validate(json.loads(row['metadata'])).model_dump(mode='json')
        job = self.jobs.get(row['analysis_id']) if row['analysis_id'] else None
        score = replay_score(row) if row['source'] == 'manual' else None
        status = ('COMPLETED' if score['state']['match_complete'] else 'PAUSED' if row['paused'] else 'LIVE') if score else job['status'] if job else 'DRAFT'
        result = {key: row[key] for key in ['id', 'source', 'analysis_id', 'revision', 'created_at', 'updated_at']}
        result.update(metadata=metadata, metadata_source='user' if any(p['name'] or p['profile_id'] for p in metadata['participants']) or metadata['tournament'] or metadata['match_date'] else 'unknown',
                      status=status, filename=job['filename'] if job else None, stage=job['stage'] if job else None,
                      error=job['error'] if job else None, score=score, notes=row['notes'] if detail else None,
                      identity_suggestions=[], predictions=(score['predictions'] if score else {
                          'status': 'unavailable', 'match_id': row['id'], 'revision': row['revision'],
                          'source': None, 'generated_at': None, 'evidence': [], 'probabilities': None,
                          'calibration_status': 'not_calibrated', 'history': [],
                          'reason': 'A confirmed initial score and qualified point outcomes are required.'}))
        if score and not detail:
            result['score'] = {**score, 'history': [], 'predictions': {**score['predictions'], 'history': []}}
            result['predictions'] = {**result['predictions'], 'history': []}
        if row['analysis_id'] and job and job['status'] == 'COMPLETED':
            try:
                from src.api.result_document import result_summary
                summary = result_summary(self.jobs.output_dir(row['analysis_id']), row['analysis_id'])
                result['analysis_summary'] = summary
            except (OSError, ValueError, sqlite3.Error, TypeError, KeyError):
                result['analysis_summary'] = None
        else:
            result['analysis_summary'] = None
        return result

    def get(self, match_id, owner_id=LOCAL_OWNER):
        with self.jobs.connect() as conn:
            return self.serialize(self.raw(conn, match_id, owner_id))

    def validate_metadata(self, conn, metadata, owner_id=LOCAL_OWNER):
        data = metadata.model_dump(mode='json')
        ids = []
        for participant in data['participants']:
            if participant['profile_id']:
                player = conn.execute('SELECT name FROM players WHERE id=? AND owner_id=?', (participant['profile_id'], owner_id)).fetchone()
                if not player:
                    raise HTTPException(422, 'Choose an existing player profile.')
                participant['name'] = player['name']
                ids.append(participant['profile_id'])
            participant['name'] = participant['name'] or None
        if len(ids) != len(set(ids)):
            raise HTTPException(422, 'The two participants must be different player profiles.')
        return json.dumps(data)

    def list(self, owner_id=LOCAL_OWNER):
        with self.jobs.connect() as conn:
            rows = conn.execute("SELECT * FROM matches WHERE owner_id=? AND (source='manual' OR analysis_id IS NOT NULL) ORDER BY created_at DESC", (owner_id,)).fetchall()
        return [self.serialize(dict(row), detail=False) for row in rows]

    def edit(self, match_id, payload, owner_id=LOCAL_OWNER):
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self.raw(conn, match_id, owner_id)
            if row['revision'] != payload.expected_revision:
                raise HTTPException(409, 'This match changed in another window. Refresh and try again.')
            if isinstance(payload, MetadataEdit):
                metadata = payload.metadata
                before = json.loads(row['metadata']).get('participants')
                if before != metadata.model_dump(mode='json')['participants']:
                    metadata = metadata.model_copy(update={'role_assignment': 'unconfirmed'})
                conn.execute('UPDATE matches SET metadata=?,revision=revision+1,updated_at=? WHERE id=?',
                             (self.validate_metadata(conn, metadata, owner_id), utc_now(), match_id))
            else:
                conn.execute('UPDATE matches SET notes=?,revision=revision+1,updated_at=? WHERE id=?', (payload.notes, utc_now(), match_id))
        return self.get(match_id, owner_id)

    def command(self, match_id, command, owner_id=LOCAL_OWNER):
        payload = command.model_dump_json()
        with self.jobs.connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = self.raw(conn, match_id, owner_id)
            previous = conn.execute('SELECT payload FROM score_commands WHERE match_id=? AND command_id=?', (match_id, command.command_id)).fetchone()
            if previous:
                if previous['payload'] != payload:
                    raise HTTPException(409, 'A command ID cannot be reused for a different update.')
                return self.serialize(row)
            if row['source'] != 'manual':
                raise HTTPException(422, 'Manual scoring is only available for manual matches.')
            if row['revision'] != command.expected_revision:
                raise HTTPException(409, 'The score changed in another window. Refresh before recording another point.')
            points = json.loads(row['points'])
            paused = row['paused']
            if command.action == 'point':
                if paused:
                    raise HTTPException(409, 'Resume scoring before recording a point.')
                if command.winner is None or replay_score(row)['state']['match_complete']:
                    raise HTTPException(422, 'Select a winner for an unfinished match.')
                points.append({'winner': command.winner, 'recorded_at': utc_now(),
                               'point_id': command.command_id, 'provenance': 'manual'})
            elif command.action == 'undo':
                if not points:
                    raise HTTPException(422, 'There are no points to undo.')
                points.pop()
            elif command.action == 'correct':
                if command.winner is None or command.point_index is None or command.point_index >= len(points):
                    raise HTTPException(422, 'Select a recorded point and its corrected winner.')
                point = points[command.point_index]
                point.setdefault('point_id', f'legacy-point-{command.point_index}')
                point.setdefault('corrections', []).append({'previous_winner': point['winner'],
                    'winner': command.winner, 'command_id': command.command_id,
                    'revision': row['revision']+1, 'recorded_at': utc_now()})
                point['winner'] = command.winner
                point['provenance'] = 'human_review'
            else:
                paused = command.action == 'pause'
            replay_score(row, points)
            conn.execute('UPDATE matches SET points=?,paused=?,revision=revision+1,updated_at=? WHERE id=?', (json.dumps(points), paused, utc_now(), match_id))
            conn.execute('INSERT INTO score_commands VALUES (?,?,?,?)', (match_id, command.command_id, payload, utc_now()))
        return self.get(match_id, owner_id)


def product_router(jobs):
    router = APIRouter(prefix='/api')
    store = ProductStore(jobs)

    @router.get('/matches')
    def matches(owner_id: str = Depends(current_owner)):
        return store.list(owner_id)

    @router.post('/matches', status_code=201)
    def create_match(payload: CreateMatch, owner_id: str = Depends(current_owner)):
        match_id, now = 'session_' + uuid.uuid4().hex, utc_now()
        with jobs.connect() as conn:
            metadata = store.validate_metadata(conn, payload.metadata, owner_id)
            config = json.dumps({'best_of': payload.best_of, 'initial_server': payload.initial_server}) if payload.source == 'manual' else None
            conn.execute('INSERT INTO matches(id,source,metadata,live_config,created_at,updated_at,owner_id) VALUES (?,?,?,?,?,?,?)',
                         (match_id, payload.source, metadata, config, now, now, owner_id))
        return store.get(match_id, owner_id)

    @router.get('/matches/by-analysis/{analysis_id}')
    def by_analysis(analysis_id: str, owner_id: str = Depends(current_owner)):
        with jobs.connect() as conn:
            row = conn.execute('SELECT id FROM matches WHERE analysis_id=? AND owner_id=?', (analysis_id, owner_id)).fetchone()
        if row is None:
            raise HTTPException(404, 'No match is linked to this analysis.')
        return store.get(row['id'], owner_id)

    @router.get('/matches/{match_id}')
    def match(match_id: str, owner_id: str = Depends(current_owner)):
        return store.get(match_id, owner_id)

    @router.patch('/matches/{match_id}/metadata')
    def metadata(match_id: str, payload: MetadataEdit, owner_id: str = Depends(current_owner)):
        return store.edit(match_id, payload, owner_id)

    @router.put('/matches/{match_id}/notes')
    def notes(match_id: str, payload: NoteEdit, owner_id: str = Depends(current_owner)):
        return store.edit(match_id, payload, owner_id)

    @router.post('/matches/{match_id}/score-commands')
    def scoring(match_id: str, payload: ScoreCommand, owner_id: str = Depends(current_owner)):
        return store.command(match_id, payload, owner_id)

    @router.get('/players')
    def players(owner_id: str = Depends(current_owner)):
        with jobs.connect() as conn:
            return [dict(row) for row in conn.execute('SELECT id,name,created_at FROM players WHERE owner_id=? ORDER BY name,created_at', (owner_id,))]

    @router.post('/players', status_code=201)
    def create_player(payload: CreatePlayer, owner_id: str = Depends(current_owner)):
        player = {'id': 'player_' + uuid.uuid4().hex, 'name': payload.name, 'created_at': utc_now()}
        with jobs.connect() as conn:
            conn.execute('INSERT INTO players(id,name,created_at,owner_id) VALUES (?,?,?,?)', (*player.values(), owner_id))
        return player

    @router.get('/players/{player_id}')
    def player_detail(player_id: str, owner_id: str = Depends(current_owner)):
        with jobs.connect() as conn:
            player = conn.execute('SELECT id,name,created_at FROM players WHERE id=? AND owner_id=?', (player_id, owner_id)).fetchone()
        if player is None:
            raise HTTPException(404, 'Player not found.')
        linked = [match for match in store.list(owner_id) if any(p['profile_id'] == player_id for p in match['metadata']['participants'])]
        motion = []
        for match in linked:
            assignment = match['metadata']['role_assignment']
            if match['source'] != 'upload' or match['status'] != 'COMPLETED' or assignment == 'unconfirmed':
                continue
            participant = next(i for i, p in enumerate(match['metadata']['participants']) if p['profile_id'] == player_id)
            role = participant if assignment == 'same' else 1-participant
            try:
                from src.api.result_document import open_result
                with open_result(jobs.output_dir(match['analysis_id']), match['analysis_id']) as data:
                    metric = data.get('playerMetrics', {}).get(f'player_{role+1}')
                    if metric:
                        motion.append({'match_id': match['id'], 'date': match['metadata']['match_date'], 'duration_seconds': data['summary']['durationSeconds'],
                                       'metrics': metric, 'source': 'video_estimate', 'confirmed_role': role+1})
            except (OSError, ValueError, sqlite3.Error, AttributeError, KeyError, TypeError):
                continue
        return {**dict(player), 'matches': linked, 'movement': motion}

    return router
