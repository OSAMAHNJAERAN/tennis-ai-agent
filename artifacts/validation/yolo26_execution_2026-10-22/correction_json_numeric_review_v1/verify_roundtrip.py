"""Independent Node JSON -> native correction -> typed replay admission proof.

Reads the existing explicit synthetic sealed source without mutating its store.
Additional arbitrary evidence fields are declared local synthetic probe inputs.
"""
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
FIXTURE = ROOT/'artifacts/validation/yolo26_execution_2026-10-22/correction_browser_fixture_v1'
TEMP = ART/'temp'
TEMP.mkdir(exist_ok=False)
os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), TENNIS_STORAGE_ROOT=str(ART/'bootstrap'),
                 PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
paths = [ROOT/name for name in ('src/scoring/corrected_rule_rebuild.py', 'src/api/corrections.py',
    'src/scoring/rule_replay_source.py', 'src/scoring/point_timeline.py', 'src/scoring/match_state_codec.py',
    'frontend/src/data/correctionDraft.ts', 'frontend/src/data/correctionApi.ts')]+[Path(__file__).resolve()]
def pins():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
before = pins()
attempts = []
native_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'torch', 'ultralytics'}:
        attempts.append(name)
        raise AssertionError('No model runtime in numeric JSON review')
    return native_import(name, *args, **kwargs)
builtins.__import__ = guarded
observations = []
started = time.perf_counter()
try:
    from src.api.analysis_generation import resolve_generation
    from src.scoring.corrected_rule_rebuild import _replacement, _canonical, _digest
    from src.scoring.rule_replay_source import open_rule_replay_source
    from src.scoring.point_timeline import event_input_record
    manifest = json.loads((FIXTURE/'fixture_manifest.json').read_text(encoding='utf-8'))
    analysis = Path(manifest['storage_root'])/'analyses'/manifest['analysis_id']
    marker = (analysis/'.result/current.store').read_bytes()
    snapshot = resolve_generation(analysis, manifest['base_generation'])
    source_path = snapshot.resolve_file('native_rule_replay.json')
    original_source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    node_program = "const fs=require('node:fs');const row=structuredClone(JSON.parse(fs.readFileSync(0,'utf8')));process.stdout.write(JSON.stringify(row));"
    def browser(value):
        result = subprocess.run([r'C:\Program Files\nodejs\node.exe', '-e', node_program],
            input=json.dumps(value, allow_nan=False), capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    with open_rule_replay_source(source_path, TEMP,
            artifact_sha256=snapshot.file_sha256('native_rule_replay.json'),
            source_sha256=snapshot.file_sha256('source.mp4'), analysis_id=snapshot.analysis_id,
            analysis_revision=snapshot.analysis_revision) as source:
        row = source.document['inputs'][0]
        original = json.loads(_canonical(row['source_input']))
        original_fingerprint = row['input_fingerprint']
        assert _digest(original) == original_fingerprint

        def check(label, original, replacement, expected):
            original_wire, replacement_wire = _canonical(original), _canonical(replacement)
            failure, normalized = None, None
            try:
                normalized = _replacement(original, {'replacement_source_input': replacement})
                event, context = source.decode_source_event(normalized)
                assert _canonical(event_input_record(event, context)) == _canonical(normalized)
            except (ValueError, TypeError, OverflowError) as error:
                failure = type(error).__name__+': '+str(error)
            passed = failure is None
            assert passed is expected, (label, failure, expected)
            assert _canonical(original) == original_wire
            assert _canonical(replacement) == replacement_wire
            if passed:
                for name in ('timestamp_s', 'confidence', 'ball_position_px', 'court_position_m', 'evidence'):
                    assert _canonical(normalized[name]) == _canonical(original[name])
            observations.append({'case': label, 'accepted': passed, 'expected': expected,
                'rejection': failure, 'raw_original_preserved': True, 'raw_client_preserved': True,
                'normalized_measurements_restore_original_json': passed,
                'typed_decoder_roundtrip_succeeded': passed})

        replacement = browser(original)
        replacement['event_type'] = 'UNKNOWN_EVENT'
        check('actual_browser_float_zero_and_integral_coordinates', original, replacement, True)
        check('formatting_only_without_semantic_change', original, browser(original), False)

        altered = copy.deepcopy(replacement)
        altered['timestamp_s'] = False
        check('boolean_timestamp_is_not_zero', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['ball_position_px'][0] = True
        check('boolean_coordinate_is_not_number', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['timestamp_s'] += 1e-12
        check('tiny_real_timestamp_change_is_rejected', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['ball_position_px'][0] += 1e-9
        check('tiny_real_coordinate_change_is_rejected', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['frame_index'] = float(original['frame_index'])
        check('counter_float_remains_invalid', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['context']['analysis_allowed'] = 1
        check('context_boolean_cannot_be_numeric', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['context']['source_pts'] = int(original['context']['source_pts'])
        check('exact_source_pts_must_stay_string', original, altered, False)
        altered = copy.deepcopy(replacement)
        altered['context']['binding']['identity_epoch'] = float(original['context']['binding']['identity_epoch'])
        check('binding_counter_float_remains_invalid', original, altered, False)

        nested = copy.deepcopy(original)
        nested['evidence'] = {'probe': 'EXPLICIT_SYNTHETIC_NESTED_EVIDENCE', 'negative_zero': -0.0,
            'numbers': [1.0, 2e20, .125], 'flag': False, 'null': None}
        altered = browser(nested)
        altered['event_type'] = 'UNKNOWN_EVENT'
        check('nested_evidence_numeric_format_and_negative_zero_restore', nested, altered, True)
        bad = copy.deepcopy(altered)
        bad['evidence']['flag'] = 0
        check('nested_evidence_boolean_type_is_strict', nested, bad, False)
        bad = copy.deepcopy(altered)
        bad['evidence']['numbers'] = list(reversed(bad['evidence']['numbers']))
        check('nested_evidence_list_order_is_strict', nested, bad, False)
        bad = copy.deepcopy(altered)
        bad['evidence']['added'] = None
        check('nested_evidence_dict_keys_are_strict', nested, bad, False)
        bad = copy.deepcopy(altered)
        bad['evidence']['null'] = 0
        check('nested_evidence_null_is_not_zero', nested, bad, False)

        large = copy.deepcopy(nested)
        large['evidence']['large_integer'] = 2**53+1
        altered = browser(large)
        altered['event_type'] = 'UNKNOWN_EVENT'
        assert altered['evidence']['large_integer'] != large['evidence']['large_integer']
        check('browser_loss_of_large_integer_precision_is_rejected', large, altered, False)
        assert _digest(original) == original_fingerprint
    assert (analysis/'.result/current.store').read_bytes() == marker
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == original_source_sha
finally:
    builtins.__import__ = native_import
after = pins()
receipt = {'status': 'PASS', 'scope': 'ACTUAL_NODE_JSON_PARSE_STRUCTUREDCLONE_STRINGIFY_TO_NATIVE_REPLACEMENT_AND_TYPED_DECODER; '
    'READ_ONLY_EXISTING_SYNTHETIC_FIXTURE_PLUS_DECLARED_LOCAL_NESTED_EVIDENCE_CASES',
    'elapsed_seconds': time.perf_counter()-started, 'observations': observations,
    'source_before': before, 'source_after': after, 'source_stable': before == after,
    'original_input_fingerprint': original_fingerprint, 'original_fingerprint_verified_unchanged': True,
    'browser_fixture_store_modified': False, 'inference_import_attempts': attempts,
    'inference_loaded': sorted({'torch','ultralytics'} & set(sys.modules))}
(ART/'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({key: receipt[key] for key in ('status','elapsed_seconds','source_stable','observations')}, indent=2))
assert before == after and not attempts and not receipt['inference_loaded']
