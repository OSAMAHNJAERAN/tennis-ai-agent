"""Native analytics over sealed explicit point and serve-attempt memberships."""
from collections.abc import Sequence
from dataclasses import replace
import math

from src.analytics.shot_statistics import ShotStatisticsAnalyzer, _landing_record
from src.scoring.point_timeline import EligibleShots


class TimelineLandings(Sequence):
    def __init__(self, owner, shots):
        self.owner, self.shots = owner, shots
        self.positions = owner.eligible_positions(landings=True)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            from src.utils.mapped_sequence_view import MappedSequenceView
            return MappedSequenceView(self, lambda value, _: value)[index]
        position = self.positions[index]
        return _landing_record(replace(self.shots[position], player_id=self.owner.participant(position)))

    def __iter__(self):
        iterator = iter(self.positions)
        try:
            for position in iterator:
                yield _landing_record(replace(self.shots[position], player_id=self.owner.participant(position)))
        finally:
            iterator.close()


def timeline_serve_statistics(owner):
    result = {}
    for player in (1, 2):
        counts = {'server_id': player, 'first_serves_attempted': 0, 'first_serves_in': 0,
            'first_serves_faulted': 0, 'second_serves_attempted': 0, 'second_serves_in': 0,
            'second_serves_faulted': 0, 'double_faults': 0, 'service_lets': 0,
            'serve_contact_executions': 0, 'unresolved_attempts': 0}
        for attempt in owner.rows('attempts'):
            if attempt['server_id'] != player:
                continue
            counts['serve_contact_executions'] += 1
            if attempt['status'] == 'SERVICE_LET':
                counts['service_lets'] += 1
                continue
            prefix = 'first' if attempt['official_attempt'] == 1 else 'second'
            counts[prefix+'_serves_attempted'] += 1
            if attempt['status'] == 'SERVE_IN_PLAY':
                counts[prefix+'_serves_in'] += 1
            elif attempt['status'] in ('FIRST_SERVE_FAULT', 'DOUBLE_FAULT'):
                counts[prefix+'_serves_faulted'] += 1
                counts['double_faults'] += attempt['status'] == 'DOUBLE_FAULT'
            else:
                counts['unresolved_attempts'] += 1
        denominator = counts['first_serves_in']+counts['first_serves_faulted']
        counts.update(first_serve_percentage=round(counts['first_serves_in']/denominator*100., 1) if denominator else None,
            first_serve_percentage_denominator=denominator,
            percentage_scope='RESOLVED_OFFICIAL_FIRST_SLOTS_EXCLUDING_LET_EXECUTIONS',
            placement_distribution={'WIDE': 0, 'BODY': 0, 'T': 0, 'UNKNOWN': counts['serve_contact_executions']},
            placement_measurement_available=False)
        result[str(player)] = counts
    return {'by_player': result, 'scope': 'EXPLICIT_POINT_ATTEMPT_RULE_EVIDENCE_NOT_FINAL_SERVER_FILTER'}


def build_timeline_analytics(state, shots, timeline, *, authoritative_enabled, p1_positions=(), p2_positions=(),
                             p1_distance_m=None, p2_distance_m=None, p1_speed_kmh=None, p2_speed_kmh=None,
                             ball_speed_summary=None):
    owner = timeline.owner
    if not owner.sealed:
        raise ValueError('Native analytics require a sealed point revision')
    eligible = EligibleShots(owner, shots)
    players = {}
    for player, positions, distance, speed in ((1, p1_positions, p1_distance_m, p1_speed_kmh),
                                              (2, p2_positions, p2_distance_m, p2_speed_kmh)):
        statistics = ShotStatisticsAnalyzer.compute_player_shot_stats(eligible, player)
        players['player_'+str(player)] = {'id': player,
            'movement': {'distance_meters': round(distance, 2) if distance is not None and math.isfinite(distance) else None,
                         'average_speed_kmh': round(speed, 1) if speed is not None and math.isfinite(speed) else None},
            'shots': statistics['shot_type_counts'], 'direction': statistics['direction_counts'],
            'landing_zones': statistics['landing_zone_counts'],
            'court_occupancy_heatmap': ShotStatisticsAnalyzer.compute_court_occupancy_heatmap(positions)}
    completed = owner.rows('points', status='COMPLETED')
    legal_rallies = owner.rows('rallies', status='COMPLETED')
    total = longest = 0
    for rally in legal_rallies:
        total += rally['total_strokes_including_serve']
        longest = max(longest, rally['total_strokes_including_serve'])
    return {'schema_version': '2.0', 'match_id': state.match_id, 'format': state.format.value,
        'match_complete': state.match_complete if authoritative_enabled else None,
        'winner_id': state.winner_id if authoritative_enabled and state.score_confirmed else None,
        'score_summary': state.get_score_summary_str() if authoritative_enabled and state.score_confirmed else None,
        'overview': {'total_points_played': len(completed) if authoritative_enabled else None,
            'partial_or_open_points': len(owner.rows('points'))-len(completed) if authoritative_enabled else None,
            'total_shots_recorded': len(shots), 'live_shots_recorded': len(eligible),
            'shots_without_legal_rally_membership': len(shots)-len(eligible),
            'total_rallies': len(legal_rallies),
            'average_rally_length': round(total/len(legal_rallies), 1) if len(legal_rallies) else None,
            'longest_rally_length': longest if len(legal_rallies) else None},
        'players': players, 'serve_analytics': timeline_serve_statistics(owner),
        'ball_landing_heatmap': TimelineLandings(owner, shots), 'ball_speed_analytics': ball_speed_summary or {},
        'event_statistics_scope': 'EXPLICIT_QUALIFIED_RULE_MEMBERSHIPS_WITH_PARTIAL_COVERAGE',
        'point_timeline': {'analysis_id': owner.analysis_id, 'revision': owner.revision,
            'attempt_artifact': 'serve_attempts.json', 'shot_scope_artifact': 'shot_play_scope.json',
            'native_scoring_history_memory_bounded': False, 'full_match_capacity_proven': False}}
