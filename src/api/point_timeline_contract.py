"""Optional, unqualified native point evidence for a saved web result.

Files are admitted together by their manifest, source identity and revision.
Large JSON arrays and reference checks stay on disk in the worker path.
Artifact integrity does not qualify the underlying video measurements.
"""
from contextlib import closing
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from src.api.result_storage import DiskSeries
from src.utils.point_timeline_artifact import (MANIFEST_NAME, POINT_TIMELINE_FILES,
                                             SHA256_PATTERN, artifact_sha256)


# These are the published physical-event wire values, not rule-only signals.
# Keeping this boundary literal avoids loading the detector and model runtime.
PHYSICAL_EVENT_TYPES = frozenset(('SERVE_CONTACT', 'PLAYER_1_HIT', 'PLAYER_2_HIT',
                                  'BOUNCE', 'UNKNOWN_EVENT'))
REVIEWED_RULE_TYPES = frozenset(('SERVICE_LET', 'CONSEQUENTIAL_GAP', 'POINT_OUTCOME_UNKNOWN'))


def _require(condition, message='Invalid native point-timeline evidence'):
    if not condition:
        raise ValueError(message)


def _object(value):
    _require(isinstance(value, dict))
    return value


def _array(value):
    _require(isinstance(value, (list, DiskSeries)))
    return value


def _int(value, *, nullable=False, minimum=0, choices=None):
    if nullable and value is None:
        return None
    _require(type(value) is int and value >= minimum and (choices is None or value in choices))
    return value


def _number(value, *, nullable=False):
    if nullable and value is None:
        return None
    _require(type(value) in (float, int) and math.isfinite(value) and value >= 0)
    return value


def _timestamp_key(value):
    """Match normalized event timestamps without rounding absolute source PTS.

    Fraction retains the exact numerical value of the decoded JSON integer or
    float. Source PTS and rational time bases remain separate context tokens.
    """
    return str(Fraction(_number(value)))


def _text(value, *, nullable=False):
    if nullable and value is None:
        return None
    _require(isinstance(value, str))
    return value


def _participant(value, *, nullable=False):
    return _int(value, nullable=nullable, minimum=1, choices=(1, 2))


def _uid(kind, anchor, manifest):
    body = [manifest['analysis_id'], manifest['source_sha256'], manifest['analysis_revision'], kind, anchor]
    return kind+'_'+hashlib.sha256(json.dumps(body, separators=(',', ':')).encode()).hexdigest()


def _context(value, manifest, frame_count):
    context = _object(value)
    frame = _int(context.get('frame_index'))
    _require(frame < frame_count)
    _require(type(context.get('analysis_allowed')) is bool)
    binding = context.get('binding')
    if binding is not None:
        _object(binding)
        _require(binding.get('analysis_id') == manifest['analysis_id']
                 and binding.get('source_fingerprint') == manifest['source_sha256']
                 and binding.get('frame_index') == frame)
        for key in ('source_frame_index', 'source_pts', 'source_time_base', 'segment_id'):
            _require(binding.get(key) == context.get(key))
    _require(type(context.get('binding_audit_matches')) is bool)
    _require(context['binding_audit_matches'] is False or binding is not None)
    if context.get('source_pts') is not None:
        token = context['source_pts']
        _require(type(token) is int or isinstance(token, str) and str(int(token)) == token)
        _require(isinstance(context.get('source_time_base'), str)
                 and Fraction(context['source_time_base']) > 0)
    return context


def _manifest(directory, analysis_id, authoritative):
    path = directory / MANIFEST_NAME
    _require(not path.is_symlink() and path.stat().st_size <= 16384)
    manifest_bytes = path.read_bytes()
    _require(len(manifest_bytes) <= 16384)
    manifest = _object(json.loads(manifest_bytes))
    _require(manifest.get('schema_version') == '1.0'
             and manifest.get('analysis_id') == analysis_id
             and manifest.get('independent_accuracy_validated') is False)
    _int(manifest.get('analysis_revision'))
    _require(type(manifest.get('scoring_revision')) is int and manifest['scoring_revision'] == 0,
             'Corrected scoring and point-timeline revisions are not yet jointly published')
    _require(type(manifest.get('authoritative_events_enabled')) is bool
             and manifest['authoritative_events_enabled'] == authoritative)
    _require(isinstance(manifest.get('source_sha256'), str)
             and SHA256_PATTERN.fullmatch(manifest['source_sha256']) is not None)
    artifacts = _object(manifest.get('artifacts'))
    _require(set(artifacts) == set(POINT_TIMELINE_FILES))
    _verify_hashes(directory, manifest)
    source = directory / 'source.mp4'
    _require(not source.is_symlink() and source.is_file()
             and artifact_sha256(source) == manifest['source_sha256'], 'Point timeline belongs to a different source video')
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def _verify_hashes(directory, manifest):
    for name in POINT_TIMELINE_FILES:
        record = _object(manifest['artifacts'][name])
        size = _int(record.get('bytes'))
        digest = record.get('sha256')
        _require(isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None)
        path = directory / name
        _require(not path.is_symlink() and path.is_file() and path.stat().st_size == size
                 and artifact_sha256(path) == digest, 'Point-timeline artifact changed: '+name)


def _point(row):
    row = _object(row)
    _require(row.get('uid') == row.get('point_id'))
    _require(type(row.get('is_review_pending')) is bool)
    _require(row.get('status') in ('OPEN', 'PARTIAL_UNANCHORED', 'REVIEW_REQUIRED', 'COMPLETED'))
    _require(row.get('status') != 'COMPLETED' or row.get('winner_id') in (1, 2))
    return {'id': _text(row.get('uid')), 'status': _text(row.get('status')),
        'anchorEventId': _int(row.get('anchor_event_id'), nullable=True),
        'terminalEventId': _int(row.get('terminal_event_id'), nullable=True),
        'serverId': _participant(row.get('server_id'), nullable=True),
        'receiverId': _participant(row.get('receiver_id'), nullable=True),
        'setNumber': _int(row.get('set_number'), nullable=True, minimum=1),
        'gameNumber': _int(row.get('game_number'), nullable=True, minimum=1),
        'serviceSide': _text(row.get('service_side'), nullable=True),
        'winnerId': _participant(row.get('winner_id'), nullable=True),
        'endingReason': _text(row.get('ending_reason'), nullable=True),
        'isReviewPending': row['is_review_pending'],
        'totalStrokesIncludingServe': _int(row.get('total_strokes_including_serve')),
        'rallyHitsExcludingServe': _int(row.get('rally_hits_excluding_serve')),
        'durationSeconds': _number(row.get('duration_s'), nullable=True)}


def _rally(row, manifest, frame_count):
    row = _object(row)
    _require(row.get('uid') == row.get('rally_id') and row.get('point_uid') == row.get('point_id'))
    context = _context(row.get('start_context'), manifest, frame_count)
    _require(row.get('status') in ('OPEN', 'REVIEW_REQUIRED', 'COMPLETED'))
    _require(row.get('status') == 'COMPLETED' or row.get('duration_s') is None)
    return {'id': _text(row.get('uid')), 'pointId': _text(row.get('point_uid')),
        'status': _text(row.get('status')), 'serverId': _participant(row.get('server_id'), nullable=True),
        'receiverId': _participant(row.get('receiver_id'), nullable=True),
        'serveAttempt': _participant(row.get('serve_attempt'), nullable=True),
        'winnerId': _participant(row.get('winner_id'), nullable=True),
        'endingReason': _text(row.get('ending_reason'), nullable=True),
        'startFrame': context['frame_index'],
        'terminalEventId': _int(row.get('terminal_event_id'), nullable=True),
        'durationSeconds': _number(row.get('duration_s'), nullable=True),
        'durationReason': _text(row.get('duration_reason'), nullable=True),
        'totalStrokesIncludingServe': _int(row.get('total_strokes_including_serve')),
        'rallyHitsExcludingServe': _int(row.get('rally_hits_excluding_serve'))}


def _attempt(row, manifest, frame_count):
    row = _object(row)
    _context(row.get('start_context'), manifest, frame_count)
    _require(row.get('status') in ('PENDING', 'FIRST_SERVE_FAULT', 'DOUBLE_FAULT',
                                  'SERVICE_LET', 'SERVE_IN_PLAY', 'REVIEW_REQUIRED'))
    return {'id': _text(row.get('uid')), 'pointId': _text(row.get('point_uid')),
        'status': _text(row.get('status')), 'officialAttempt': _participant(row.get('official_attempt')),
        'replayIndex': _int(row.get('replay_index')),
        'contactEventId': _int(row.get('contact_event_id')),
        'resultEventId': _int(row.get('result_event_id'), nullable=True),
        'serverId': _participant(row.get('server_id')), 'serverEnd': _text(row.get('server_end')),
        'serviceSide': _text(row.get('service_side')),
        'rallyId': _text(row.get('rally_uid'), nullable=True)}


def _serve_summary(match):
    by_player = _object(_object(match.get('serve_analytics')).get('by_player'))
    _require(set(by_player) == {'1', '2'})
    output = {}
    for player in (1, 2):
        raw = _object(by_player[str(player)])
        _require(raw.get('server_id') == player)
        values = {'serverId': player}
        for native, field in (('contact_executions', 'contactExecutions'), ('service_lets', 'serviceLets'),
                ('first_serves_attempted', 'firstServesAttempted'), ('second_serves_attempted', 'secondServesAttempted'),
                ('first_serves_in', 'firstServesIn'), ('first_serves_faulted', 'firstServesFaulted'),
                ('second_serves_in', 'secondServesIn'), ('double_faults', 'doubleFaults'),
                ('unresolved_executions', 'unresolvedExecutions'),
                ('first_serve_percentage_denominator', 'firstServePercentageDenominator')):
            values[field] = _int(raw.get(native))
        denominator = values['firstServesIn']+values['firstServesFaulted']
        percent = _number(raw.get('first_serve_percentage'), nullable=True)
        _require(values['firstServePercentageDenominator'] == denominator
                 and (percent is None if denominator == 0 else percent == values['firstServesIn']/denominator*100))
        values['firstServePercentage'] = percent
        output['player_'+str(player)] = values
    return output


def _load_point_analysis(directory, analysis_id, *, frame_count, authoritative, storage=None):
    """Return None for historical output; reject a present inconsistent bundle."""
    directory = Path(directory)
    if not (directory / MANIFEST_NAME).exists():
        _require(not any((directory / name).exists() for name in
                        ('serve_executions.json', 'shot_play_scope.json')),
                 'Native point-timeline manifest is missing')
        return None
    manifest, manifest_digest = _manifest(directory, analysis_id, authoritative)
    def read(name):
        return _object(storage.read_json(directory / name) if storage is not None else
                       json.loads((directory / name).read_text(encoding='utf-8')))
    documents = {name: read(name) for name in POINT_TIMELINE_FILES}
    event_path = directory / 'match_events.json'
    _require(not event_path.is_symlink() and event_path.is_file())
    event_digest = artifact_sha256(event_path)
    events = read('match_events.json')
    _require(events.get('schema_version') == '1.0'
             and events.get('analysis_id') == analysis_id
             and type(events.get('analysis_revision')) is int
             and events['analysis_revision'] == manifest['analysis_revision'])
    for name in POINT_TIMELINE_FILES:
        expected = '1.0' if name == 'match_state.json' else '2.0'
        _require(documents[name].get('schema_version') == expected)
    for name in ('serve_executions.json', 'shot_play_scope.json'):
        _require(type(documents[name].get('analysis_revision')) is int
                 and documents[name]['analysis_revision'] == manifest['analysis_revision'])
    history = documents['score_history.json']
    _require(type(history.get('revision')) is int and history['revision'] == manifest['scoring_revision'])
    _array(history.get('score_history'))
    _array(history.get('corrections'))
    state = documents['match_state.json']
    match = documents['match_analytics.json']
    _require(state.get('match_id') == analysis_id and match.get('match_id') == analysis_id
             and state.get('authoritative_events_enabled') is authoritative
             and match.get('authoritative_events_enabled') is authoritative
             and match.get('independent_accuracy_validated') is False)
    overview = _object(match.get('overview'))
    summary = {}
    for native, field, nullable, numeric in (
            ('total_points_played', 'totalPointsPlayed', True, False),
            ('partial_or_review_points', 'partialOrReviewPoints', False, False),
            ('total_rallies', 'totalRallies', False, False),
            ('average_rally_length', 'averageRallyLength', True, True),
            ('longest_rally_length', 'longestRallyLength', True, False),
            ('effective_play_seconds', 'effectivePlaySeconds', True, True),
            ('timed_completed_rallies', 'timedCompletedRallies', False, False),
            ('completed_rallies_duration_denominator', 'completedRalliesDurationDenominator', False, False)):
        summary[field] = (_number if numeric else _int)(overview.get(native), nullable=nullable)
    summary.update(scoreSummary=_text(match.get('score_summary'), nullable=True),
                   matchComplete=match.get('match_complete'), winnerId=_participant(match.get('winner_id'), nullable=True))
    _require(summary['matchComplete'] is None or type(summary['matchComplete']) is bool)
    _require(authoritative or all(summary[key] is None for key in
             ('totalPointsPlayed', 'averageRallyLength', 'longestRallyLength', 'scoreSummary', 'matchComplete', 'winnerId')))
    series = storage.series if storage is not None else list
    points, rallies, attempts = series(), series(), series()
    serve_summary = _serve_summary(match)
    serve_observed = {player: dict.fromkeys(('contactExecutions', 'serviceLets', 'firstServesAttempted',
        'secondServesAttempted', 'firstServesIn', 'firstServesFaulted', 'secondServesIn',
        'doubleFaults', 'unresolvedExecutions'), 0) for player in (1, 2)}
    # sqlite's empty filename is a private temporary disk database, deleted on
    # close. Indexes grow on disk; the page cache is capped independently of time.
    with closing(sqlite3.connect('')) as index:
        index.execute('PRAGMA cache_size=-2048')
        index.execute('PRAGMA temp_store=FILE')
        index.execute('PRAGMA mmap_size=0')
        index.execute('CREATE TABLE proof(kind TEXT, uid TEXT PRIMARY KEY, point TEXT, attempt TEXT, rally TEXT, event INTEGER, anchor INTEGER)')
        index.execute("CREATE UNIQUE INDEX event_ids ON proof(event) WHERE kind='event'")
        index.execute("CREATE INDEX attempt_rallies ON proof(rally) WHERE kind='attempt'")
        index.execute('CREATE TABLE physical_events(event INTEGER PRIMARY KEY, frame INTEGER NOT NULL, '
                      'kind TEXT NOT NULL, timestamp TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0)')
        physical_count = 0
        for raw in _array(events.get('events')):
            raw = _object(raw)
            ident, frame = _int(raw.get('event_id')), _int(raw.get('frame'))
            kind = raw.get('event_type')
            _require(frame < frame_count and kind in PHYSICAL_EVENT_TYPES
                     and raw.get('reviewed_rule_evidence_ref') is None,
                     'Physical event export contains an invalid frame or event type')
            index.execute('INSERT INTO physical_events(event,frame,kind,timestamp) VALUES(?,?,?,?)',
                          (ident, frame, kind, _timestamp_key(raw.get('timestamp_s'))))
            physical_count += 1
        _require(_int(events.get('events_count')) == physical_count,
                 'Physical event export count does not match its records')
        _require(authoritative or physical_count == 0,
                 'Physical event authority is withheld but physical events were exported')
        def put(kind, uid, point=None, attempt=None, rally=None, event=None, anchor=None):
            index.execute('INSERT INTO proof VALUES(?,?,?,?,?,?,?)', (kind, uid, point, attempt, rally, event, anchor))
        completed_points = completed_rallies = timed = stroke_total = longest = 0
        duration_total = 0.
        for raw in _array(documents['point_analytics.json'].get('points')):
            row = _point(raw)
            _require(_object(raw.get('pre_point_state')).get('match_id') == analysis_id)
            put('point', row['id'], anchor=_int(raw.get('first_observed_event_id')))
            completed_points += row['status'] == 'COMPLETED'
            points.append(row)
        for raw in _array(documents['rallies.json'].get('rallies')):
            row = _rally(raw, manifest, frame_count)
            put('rally', row['id'], point=row['pointId'])
            if row['status'] == 'COMPLETED':
                completed_rallies += 1
                stroke_total += row['totalStrokesIncludingServe']
                longest = max(longest, row['totalStrokesIncludingServe'])
                if row['durationSeconds'] is not None:
                    timed += 1
                    duration_total += row['durationSeconds']
            rallies.append(row)
        for raw in _array(documents['serve_executions.json'].get('serve_executions')):
            row = _attempt(raw, manifest, frame_count)
            put('attempt', row['id'], point=row['pointId'], rally=row['rallyId'], anchor=row['contactEventId'])
            observed = serve_observed[row['serverId']]
            observed['contactExecutions'] += 1
            if row['replayIndex'] == 0:
                observed['firstServesAttempted' if row['officialAttempt'] == 1 else 'secondServesAttempted'] += 1
            field = {'SERVICE_LET': 'serviceLets', 'FIRST_SERVE_FAULT': 'firstServesFaulted',
                     'DOUBLE_FAULT': 'doubleFaults'}.get(row['status'])
            if row['status'] == 'SERVE_IN_PLAY':
                field = 'firstServesIn' if row['officialAttempt'] == 1 else 'secondServesIn'
            observed[field or 'unresolvedExecutions'] += 1
            attempts.append(row)
        scoring_records = _array(documents['scoring_events.json'].get('scoring_events'))
        _require(authoritative or len(scoring_records) == 0,
                 'Scoring event authority is withheld but rule transitions were exported')
        for raw in scoring_records:
            row = _object(raw)
            context = _context(row.get('context'), manifest, frame_count)
            event = _int(row.get('event_id'))
            expected = _uid('event', [event, context.get('source_pts'), context.get('source_time_base'),
                                     context.get('source_frame_index'), context['frame_index']], manifest)
            _require(row.get('uid') == expected and _int(row.get('frame')) == context['frame_index'])
            _require(_object(row.get('pre_state')).get('match_id') == analysis_id
                     and _object(row.get('post_state')).get('match_id') == analysis_id)
            physical = index.execute('SELECT frame,kind,timestamp,consumed FROM physical_events WHERE event=?',
                                     (event,)).fetchone()
            reference = row.get('reviewed_rule_evidence_ref')
            timestamp = _timestamp_key(row.get('timestamp_s'))
            if reference is None:
                _require(physical is not None and physical ==
                         (context['frame_index'], row.get('physical_event_type'), timestamp, 0),
                         'Physical source event is missing, changed or consumed more than once')
                index.execute('UPDATE physical_events SET consumed=1 WHERE event=?', (event,))
            else:
                _require(isinstance(reference, str) and bool(reference.strip())
                         and row.get('physical_event_type') in REVIEWED_RULE_TYPES
                         and row.get('rule_event_type') in REVIEWED_RULE_TYPES
                         and physical is None,
                         'Reviewed rule transition requires a supported explicit reference and a distinct event ID')
            put('event', expected, row.get('point_uid'), row.get('attempt_uid'), row.get('rally_uid'), event)
        _require(index.execute('SELECT 1 FROM physical_events WHERE consumed=0 LIMIT 1').fetchone() is None,
                 'Physical event export contains an event absent from the scoring transitions')
        for kind, field in (('point', 'point'), ('attempt', 'attempt'), ('rally', 'rally')):
            missing = index.execute(f'SELECT 1 FROM proof AS child LEFT JOIN proof AS parent '
                f'ON child.{field}=parent.uid AND parent.kind=? WHERE child.{field} IS NOT NULL AND parent.uid IS NULL LIMIT 1', (kind,)).fetchone()
            _require(missing is None, 'Native point-timeline reference is missing')
        for kind in ('point', 'attempt'):
            cursor = index.execute("SELECT p.uid,e.uid FROM proof AS p LEFT JOIN proof AS e ON e.kind='event' AND e.event=p.anchor WHERE p.kind=?", (kind,))
            for uid, anchor in cursor:
                _require(anchor is not None and uid == _uid(kind, anchor, manifest))
        for uid, in index.execute("SELECT uid FROM proof WHERE kind='rally'"):
            parent = index.execute("SELECT uid FROM proof WHERE kind='attempt' AND rally=?", (uid,)).fetchone()
            _require(parent is not None and uid == _uid('rally', parent[0], manifest))
        for rows, fields in ((points, ('anchorEventId', 'terminalEventId')),
                             (rallies, ('terminalEventId',)),
                             (attempts, ('contactEventId', 'resultEventId'))):
            for row in rows:
                for field in fields:
                    if row[field] is not None:
                        _require(index.execute("SELECT 1 FROM proof WHERE kind='event' AND event=?",
                                               (row[field],)).fetchone() is not None)
        for raw in _array(documents['shot_play_scope.json'].get('shots')):
            row = _object(raw)
            for field, kind in (('point_uid', 'point'), ('attempt_uid', 'attempt'), ('rally_uid', 'rally')):
                if row.get(field) is not None:
                    _require(index.execute('SELECT 1 FROM proof WHERE kind=? AND uid=?', (kind, row[field])).fetchone() is not None)
        _require(summary['totalPointsPlayed'] == (completed_points if authoritative else None)
                 and summary['partialOrReviewPoints'] == len(points)-completed_points
                 and summary['totalRallies'] == completed_rallies
                 and summary['timedCompletedRallies'] == timed
                 and summary['completedRalliesDurationDenominator'] == completed_rallies)
        _require(summary['effectivePlaySeconds'] == (duration_total if timed else None))
        _require(summary['averageRallyLength'] == (stroke_total/completed_rallies if authoritative and completed_rallies else None)
                 and summary['longestRallyLength'] == (longest if authoritative and completed_rallies else None))
        for player, values in serve_observed.items():
            _require(all(serve_summary['player_'+str(player)][key] == value for key, value in values.items()),
                     'Serve statistics do not match their execution records')
    _verify_hashes(directory, manifest)
    _require(artifact_sha256(directory / MANIFEST_NAME) == manifest_digest
             and artifact_sha256(directory / 'source.mp4') == manifest['source_sha256']
             and artifact_sha256(event_path) == event_digest,
             'Point-timeline publication changed while it was read')
    return {'schemaVersion': '1.0', 'status': 'UNQUALIFIED', 'analysisId': analysis_id,
        'analysisRevision': manifest['analysis_revision'], 'scoringRevision': manifest['scoring_revision'],
        'sourceSha256': manifest['source_sha256'], 'independentAccuracyValidated': False,
        'authoritativeEventsEnabled': authoritative, 'summary': summary,
        'serveByPlayer': serve_summary, 'points': points, 'rallies': rallies, 'serveExecutions': attempts}


def load_point_analysis(directory, analysis_id, *, frame_count, authoritative, storage=None):
    try:
        return _load_point_analysis(directory, analysis_id, frame_count=frame_count,
                                    authoritative=authoritative, storage=storage)
    except (TypeError, KeyError, OverflowError, ZeroDivisionError, sqlite3.Error) as error:
        raise ValueError('Invalid native point-timeline evidence') from error
