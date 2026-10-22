"""Normal production imports and exact native-export synthetic replay smoke."""
import builtins
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
SOURCES = ['src/scoring/rule_replay.py', 'src/scoring/rule_reproduction.py',
    'src/scoring/match_state_codec.py', 'src/scoring/match_state.py',
    'src/scoring/scoring_engine.py', 'src/scoring/scoring_rules.py',
    'src/scoring/point_timeline.py', 'src/scoring/point_outcome_resolver.py',
    'src/scoring/event_history.py', 'src/utils/scoring_journal.py',
    'src/utils/point_analysis_sequence.py', 'src/api/result_storage.py',
    'src/line_calling/line_call_engine.py', 'src/line_calling/line_geometry.py',
    'src/line_calling/contact_refinement.py', 'src/tracking/temporal_ball_tracker.py',
    'src/court/homography.py', 'src/shot_analysis/shot_types.py']
FIXTURES = ['test_exact_native_phase6_scori0', 'test_exact_native_phase6_scori1',
            'test_exact_native_phase6_revie0']
BASE = ROOT/'artifacts/validation/yolo26_execution_2026-10-17/rule_replay_final_v1/temp/pytest'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = {name: sha(ROOT/name) for name in SOURCES}
inputs = {name: {'artifact': sha(BASE/name/'native_rule_replay.json'),
                 'source': sha(BASE/name/'synthetic_source_identity.bin')} for name in FIXTURES}
attempts = []
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('Normal rule imports cannot load models: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
started = datetime.now(timezone.utc).isoformat()
begin = time.perf_counter()
from src.scoring.rule_reproduction import reproduce_rule_timeline
from src.scoring.match_state_codec import encode_match_state
from scripts.data.acquire_yolo26_court_source_wave import current_resources
rows = []
for name in FIXTURES:
    destination = RUN/name
    destination.mkdir(exist_ok=False)
    with reproduce_rule_timeline(BASE/name/'native_rule_replay.json', destination,
            artifact_sha256=inputs[name]['artifact'], source_sha256=inputs[name]['source'],
            analysis_id='fixture', analysis_revision=0) as result:
        rows.append(dict(fixture=name, authoritative=result.authoritative_events_enabled,
            transitions=len(result.owner.rows('transitions')), points=len(result.owner.rows('points')),
            attempts=len(result.owner.rows('attempts')), rallies=len(result.owner.rows('rallies')),
            shots=len(result.shots), source_frames=len(result.source_frames),
            final_raw_state=encode_match_state(result.engine.get_current_state())))
    assert not list(destination.iterdir()), 'Successful replay must remove owned scratch storage'
after = {name: sha(ROOT/name) for name in SOURCES}
inputs_after = {name: {'artifact': sha(BASE/name/'native_rule_replay.json'),
                      'source': sha(BASE/name/'synthetic_source_identity.bin')} for name in FIXTURES}
forbidden = sorted(name for name in sys.modules if name.split('.')[0] in {'torch', 'ultralytics'})
assert sources == after and inputs == inputs_after and not attempts and not forbidden
receipt = dict(status='PASS', scope='Actual ordinary production module imports; replay of three exact native Phase6 synthetic export fixtures; no vision/media/accuracy qualification',
    started_utc=started, ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-begin,
    source_before=sources, source_after=after, input_before=inputs, input_after=inputs_after,
    sources_unchanged=True, inputs_unchanged=True, forbidden_import_attempts=attempts,
    forbidden_modules_present=forbidden, rows=rows, resources=current_resources(RUN),
    numerical_image_libraries_observed=sorted({name.split('.')[0] for name in sys.modules
        if name.split('.')[0] in {'numpy', 'pandas', 'cv2', 'PIL'}}),
    model_imports_used=False, model_inference_used=False, correction_publication_implemented=False,
    independent_accuracy_validated=False, runner_sha256=sha(Path(__file__)))
(RUN/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key:receipt[key] for key in ('status', 'seconds', 'resources',
    'forbidden_import_attempts', 'numerical_image_libraries_observed')}))
