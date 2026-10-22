"""Post-fix counterpart to the retained RED probe; no inference runtime."""
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc
import weakref
from datetime import datetime, timezone

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


test = module('analytics_capacity_tests', ROOT / 'tests/test_analytics_capacity_seam.py')
red = module('analytics_capacity_red', ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/analytics_capacity_probe_v1/reproduce_current_analytics_materialization.py')
OUT = ROOT / 'artifacts/validation/yolo26_execution_2026-10-16/analytics_capacity_measurement_v1'
OUT.mkdir(exist_ok=False)
paths = [ROOT / value for value in (
    'src/utils/shot_evidence_sequence.py', 'src/utils/analytics_json_artifact.py',
    'src/analytics/rally_analyzer.py', 'src/analytics/shot_statistics.py',
    'src/analytics/serve_analyzer.py', 'src/analytics/match_analytics.py',
    'src/pipeline/phase6_pipeline.py', 'tests/test_analytics_capacity_seam.py')]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
before = {str(path.relative_to(ROOT)): sha(path) for path in paths}
original_type, original_decode = test.storage.ShotEventEvidence, test.storage._decode_shot
live, peaks = weakref.WeakSet(), {'objects': 0, 'reads': 0}


class ObservedShot(original_type):
    __hash__ = object.__hash__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        live.add(self)
        peaks['objects'] = max(peaks['objects'], len(live))


def decode(payload):
    peaks['reads'] += 1
    return original_decode(payload)


test.storage.ShotEventEvidence = ObservedShot
test.storage._decode_shot = decode
classes = test.consumers()
Rally, Statistics, Serves, Aggregator = [classes[value] for value in (
    'RallyAnalyzer', 'ShotStatisticsAnalyzer', 'ServeAnalyzer', 'MatchAnalyticsAggregator')]
consumers = {
    'rally': lambda rows, directory: Rally.analyze_rallies(rows, fps=25., result_owner=rows),
    'player_statistics': lambda rows, directory: Statistics.compute_player_shot_stats(rows, 1),
    'serve_statistics': lambda rows, directory: Serves.compute_serve_statistics(rows, [], 1),
    'bounce_heatmap': lambda rows, directory: Statistics.compute_bounce_landing_heatmap(rows, result_owner=rows),
    'point_without_rally': lambda rows, directory: Aggregator.build_point_analytics(
        test.MatchState(match_id='synthetic-capacity-only'), rows, [], [], result_owner=rows),
    'actual_phase6_step10_and_three_exports_and_return': lambda rows, directory: test.phase6_run(rows, directory, disk=True),
}
cases = []
clock, started = time.perf_counter(), datetime.now(timezone.utc).isoformat()
try:
    for count in (257, 1025, 4097):
        directory = OUT / str(count)
        directory.mkdir()
        source = red.LazyShots(count)
        with test.ShotEvidenceSequence(directory) as rows:
            for index in range(count):
                rows.append(ObservedShot(**source[index].__dict__))
            rows.seal()
            for name, consume in consumers.items():
                gc.collect()
                assert len(live) == 0
                peaks.update(objects=0, reads=0)
                stage = time.perf_counter()
                tracemalloc.start()
                result = consume(rows, directory)
                retained, maximum = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if name.startswith('actual_phase6'):
                    summary_result = {'rallies': result['rallies'], 'points': result['point_analytics'],
                                      'match': result['match_analytics']}
                    summary = red.output_summary('actual_phase6_step10', summary_result)
                    arrays_materialized = any(isinstance(value, list) for value in (
                        result['rallies'][0].shot_ids, result['point_analytics'][0]['shot_ids'],
                        result['match_analytics']['ball_landing_heatmap']))
                    # Scalar outer records remain, but contain no duration-sized arrays.
                    summary_result = None
                else:
                    summary = red.output_summary(name, result)
                    arrays_materialized = (
                        isinstance(result[0].shot_ids, list) if name == 'rally' else
                        isinstance(result[0]['shot_ids'], list) if name == 'point_without_rally' else
                        isinstance(result, list) if name == 'bounce_heatmap' else False)
                cases.append({'consumer': name, 'input_shots': count,
                    'source_reads': peaks['reads'], 'peak_live_shot_objects': peaks['objects'],
                    'retained_live_shot_objects': len(live), 'traced_peak_bytes': maximum,
                    'traced_retained_bytes': retained, 'elapsed_seconds': time.perf_counter()-stage,
                    'duration_sized_python_array_retained': arrays_materialized, 'output': summary})
                assert peaks['objects'] <= 8 and len(live) == 0 and not arrays_materialized
                result = None
                gc.collect()
                assert rows._readers == set()
        assert not list(directory.glob('shot_evidence_*'))
finally:
    test.storage.ShotEventEvidence = original_type
    test.storage._decode_shot = original_decode
after = {str(path.relative_to(ROOT)): sha(path) for path in paths}
assert before == after
assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules
receipt = {'status': 'SCOPED_ANALYTICS_CONSUMERS_AND_EXPORTS_BOUNDED_GREEN',
    'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
    'elapsed_seconds': time.perf_counter()-clock, 'cases': cases,
    'source_bindings_before': before, 'source_bindings_after': after,
    'fixture_exactly_matches_red_probe': True, 'torch_imported': False, 'ultralytics_imported': False,
    'model_used': False, 'media_used': False, 'training_performed': False,
    'qualification_evidence': False, 'full_match_capacity_proven': False,
    'python_executable': sys.executable, 'script_sha256': sha(__file__)}
(OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'receipt': str(OUT / 'receipt.json'), 'cases': cases}))
