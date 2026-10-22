"""Actual freshly exported rule bundles through the saved web evidence reader.

All inputs are explicitly synthetic native Phase6 rule fixtures, not videos.
Both the compatibility parser and caller-owned bounded result storage execute.
"""
from datetime import datetime, timezone
import builtins
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
FORBIDDEN = {'torch', 'ultralytics', 'numpy', 'cv2', 'pandas', 'PIL'}
original_import = builtins.__import__

def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in FORBIDDEN:
        raise RuntimeError('Model-free saved-reader check refuses ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded
from src.api.point_timeline_contract import load_point_analysis
from src.api.result_storage import ResultStorage, DiskSeries
from src.utils.point_timeline_artifact import POINT_TIMELINE_FILES, MANIFEST_NAME

paths = [ROOT / name for name in (
    'src/api/point_timeline_contract.py', 'src/api/result_storage.py',
    'src/utils/point_timeline_artifact.py', 'src/pipeline/phase6_pipeline.py',
    'src/scoring/point_timeline.py', 'src/scoring/scoring_engine.py',
    'src/utils/scoring_journal.py', 'src/utils/point_analysis_sequence.py',
    'tests/test_point_timeline.py')]
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = {str(path.relative_to(ROOT)): digest(path) for path in paths}
started = datetime.now(timezone.utc).isoformat()
clock = time.perf_counter()
rows, code, error = [], 0, None
fixtures = ROOT / 'artifacts/validation/yolo26_execution_2026-10-17/native_landing_checks_v1/native_exports_v3/temp/pytest'
cases = (
    ('ordinary_on', 'test_exact_native_phase6_scori0', True, 2, .970),
    ('ordinary_off', 'test_exact_native_phase6_scori1', False, None, None),
    ('reviewed_let', 'test_exact_native_phase6_revie0', True, 1, .810),
)
try:
    for label, folder, authority, points, duration in cases:
        source = fixtures / folder
        target = HERE / label
        target.mkdir(exist_ok=False)
        names = (*POINT_TIMELINE_FILES, MANIFEST_NAME, 'match_events.json')
        bindings = {name: digest(source / name) for name in names}
        for name in names:
            shutil.copyfile(source / name, target / name)
        shutil.copyfile(source / 'synthetic_source_identity.bin', target / 'source.mp4')
        assert {name: digest(target / name) for name in names} == bindings
        raw = json.loads((target / 'match_analytics.json').read_text())
        assert 'LINE_CALL_COLLECTIONS' not in raw['remaining_capacity_limits']
        assert 'CUMULATIVE_MINI_COURT_REDRAW_COST' in raw['remaining_capacity_limits']
        assert 'LANDING_OVERLAY_COLLECTIONS' not in raw['remaining_capacity_limits']
        compatible = load_point_analysis(target, 'fixture', frame_count=32, authoritative=authority)
        with ResultStorage(target) as owner:
            bounded = load_point_analysis(target, 'fixture', frame_count=32,
                                          authoritative=authority, storage=owner)
            assert all(isinstance(bounded[key], DiskSeries) for key in ('points', 'rallies', 'serveExecutions'))
            stream = io.StringIO()
            owner.dump(bounded, stream)
            normalized = json.loads(stream.getvalue())
            assert normalized == compatible
            assert normalized['summary']['totalPointsPlayed'] == points
            assert normalized['summary']['effectivePlaySeconds'] == duration
            assert normalized['status'] == 'UNQUALIFIED'
            assert normalized['independentAccuracyValidated'] is False
            if label == 'ordinary_on':
                assert normalized['points'][1]['durationSeconds'] is None
                assert normalized['serveByPlayer']['player_2']['doubleFaults'] == 1
            if label == 'reviewed_let':
                assert [(row['officialAttempt'], row['replayIndex']) for row in normalized['serveExecutions']] == [(1, 0), (1, 1)]
            (target / 'pointAnalysis.json').write_text(json.dumps(normalized, indent=2), encoding='utf-8')
        assert {name: digest(source / name) for name in names} == bindings
        rows.append(dict(case=label, compatibility_and_bounded_equal=True,
            summary=normalized['summary'], source_bindings=bindings,
            normalized_sha256=digest(target / 'pointAnalysis.json')))
except BaseException as failure:
    code, error = 1, traceback.format_exc()
finally:
    after = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    if before != after:
        code = 1
    receipt = dict(status='PASS' if code == 0 else 'FAIL', exit_code=code,
        started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - clock, cases=rows, error=error,
        source_before=before, source_after=after, sources_unchanged=before == after,
        forbidden_runtime_imports_present=sorted(FORBIDDEN.intersection(sys.modules)),
        source_is_synthetic_rule_fixture_not_video=True, models_used=False,
        training_performed=False, qualification_evidence=False, runner_sha256=digest(Path(__file__)))
    with (HERE / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in ('status', 'exit_code', 'elapsed_seconds', 'sources_unchanged', 'forbidden_runtime_imports_present', 'error')}))
raise SystemExit(code)
