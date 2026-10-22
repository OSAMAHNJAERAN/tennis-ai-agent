"""Isolated functional crop-transition checks; no model or resource admission."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import builtins
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]
ART = Path(__file__).resolve().parent
TESTS = [
    "tests/test_yolo26_crop_provider.py",
    "tests/test_yolo26_crop_pipeline.py",
    "tests/test_yolo26_models.py",
    "tests/test_analytics_capacity_seam.py",
    "tests/test_video_frame_sequence.py",
    "tests/test_ball_pass_decoder.py",
    "tests/test_native_prefix_audits.py",
    "tests/test_ball_inference_journal.py",
    "tests/test_native_inference_prefix.py",
    "tests/test_native_pipeline_model_free.py"
]
DESELECT = [
    'tests/test_yolo26_models.py::test_pipeline_constructs_only_the_bundle_and_reuses_player_pose',
    'tests/test_analytics_capacity_seam.py::test_legacy_analytics_owner_views_and_exports_remain_bounded',
]
EXPECTED = {
    "tests/test_ball_pass_decoder.py": "62f6e93a98ac1db0cbea452d6f2c205a33154a570df7b288771e452566e9fc4b",
    "tests/test_video_frame_sequence.py": "3f910c6636cf534cb37e1087d2ade140568fdd5481468f575e59b1f97b74dc99",
    "src/utils/video_frame_sequence.py": "934a157eb4a11f0fd209b1120f31da74e0ab2c3b4f165e5ebd06a3434b5fda40",
    "src/detection/yolo26_adapters.py": "1543506f26dfd4b57cd28bdd754b2d1b6610bf6229385e0d8d9da92068cbbb39",
    "tests/test_yolo26_crop_provider.py": "7a7bbb65c460b14ff2d121e2e19b67f2c9d3f98efe6faef6328fda2c12d1137e",
    "src/pipeline/phase6_pipeline.py": "d2cf08ea6a000dfeba06227616ed9ebb029c8c315662c4672a5c5ccbd8b0d22f",
    "tests/test_yolo26_crop_pipeline.py": "8f21ac224e28dabaf692c42cc557e7b99121f1ec50d6f8215c2a8ea94cec7f54",
    "tests/test_analytics_capacity_seam.py": "d7716acc8e8a40ea5c153303e6b8602482d9b0bf9daf137b5e280739090752f5",
    "src/utils/ball_inference_journal.py": "f8d75a15e53cdabdbdec0353d2feee220e14b4954a9ce332336e47d443907113",
    "src/utils/native_inference_prefix.py": "ae238c38109512366bea86ef5fe88cc79fa6c53eba488b3ea9dee3a8c2967a97",
    "tests/test_yolo26_models.py": "5118ef9158f89270150a235a789d8ed0d1f5851e74c7b1a8ee2c31b087e65bad",
    "tests/test_native_prefix_audits.py": "ea2346dccac9474e5eb74f2c75838c7ba1f6bd9d733f488f183eef0fbec3f284",
    "tests/test_ball_inference_journal.py": "89797fbcf60f87b30e7e405505ba5890ad99e99d6e4ef1d8b334d63d462ef81f",
    "tests/test_native_inference_prefix.py": "4634437ba7e95af19dade9f7f460ccb681ded94630adf611915dc8e15d1ac745",
    "tests/test_native_pipeline_model_free.py": "be4ae3c624f1d9794e021703d3661d7d1214d072588dcead4939d795a230f8a5",
    "tests/helpers/native_pipeline_source.py": "845adad6b3dc4aa8393871ad337122a1623b9fd9624332895e3b17c8146fdc8a",
    "configs/production/yolo26.yaml": "8106e134eea5a520c10747a12921d2352643e7df7d3cdb1d30ef29ae94cee9fb",
    "configs/production/yolo26.artifacts.json": "8a902c6fe9eecbe97799b48c51e2ea55ca92355e44fd7b39527d7665af30f0cb",
    "configs/production/botsort_no_reid.yaml": "c83cbe84e582ac9c7973cdc99809efd9e01565293720a9a427d06d1d6b12abf8"
}
FORBIDDEN = {'ultralytics', 'torchvision', 'torch'}


def pins():
    paths = set((ROOT / 'src').rglob('*.py'))
    paths.update(ROOT / name for name in TESTS)
    paths.update(ROOT / name for name in ['configs/production/yolo26.yaml', 'configs/production/yolo26.artifacts.json', 'configs/production/botsort_no_reid.yaml'])
    paths.update((ROOT / 'pytest.ini', ROOT / 'tests/helpers/native_pipeline_source.py', Path(__file__).resolve()))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    if set(path.name for path in ART.iterdir()) != {'run_checks.py'}:
        raise RuntimeError('Refusing preexisting check outputs; preserve the attempt and use a new folder')
    already_loaded = sorted(FORBIDDEN & set(sys.modules))
    if already_loaded:
        raise RuntimeError('Inference runtime already loaded: ' + ','.join(already_loaded))
    before = pins()
    for path, expected in EXPECTED.items():
        if before.get(path) != expected:
            raise RuntimeError('Frozen source pin changed before checks: ' + path)
    temporary = ART / 'temp'
    temporary.mkdir(exist_ok=False)
    os.chdir(ROOT)
    os.environ.update(TEMP=str(temporary), TMP=str(temporary),
        TENNIS_STORAGE_ROOT=str(ART / 'bootstrap'), PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTHONDONTWRITEBYTECODE='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT))
    arguments = ['-q', '--noconftest', '-p', 'no:cacheprovider', '--tb=short',
                 '--basetemp', str(temporary / 'pytest'), '--junitxml', str(ART / 'junit.xml'),
                 *TESTS, *[value for node in DESELECT for value in ('--deselect', node)]]
    started = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'interpreter': sys.executable,
        'interpreter_sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        'isolated_interpreter': bool(sys.flags.isolated), 'pytest_arguments': arguments,
        'frozen_pins': EXPECTED, 'source_before': before,
        'scope': 'Durable raw-ball journal, completed player prefix and audit owners, full native software/codec workflow with injected selected models, crop/decoder and adjacent contracts; no worker partial resume'}
    with (ART / 'started.json').open('x', encoding='utf-8') as stream:
        json.dump(started, stream, indent=2)
    for name in before:
        target = ART / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)

    attempts = []
    native_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.', 1)[0] in FORBIDDEN:
            attempts.append({'mechanism': 'builtins.__import__', 'name': name})
            raise AssertionError('Light crop checks attempted inference import: ' + name)
        return native_import(name, *args, **kwargs)

    class ImportTripwire(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.', 1)[0] in FORBIDDEN:
                attempts.append({'mechanism': 'meta_path', 'name': fullname})
                raise AssertionError('Light crop checks attempted inference import: ' + fullname)
            return None

    tripwire = ImportTripwire()
    builtins.__import__ = guarded_import
    sys.meta_path.insert(0, tripwire)
    clock = time.perf_counter()
    code, fatal = 2, None
    try:
        with (ART / 'pytest.txt').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            import pytest
            code = int(pytest.main(arguments))
    except BaseException:
        fatal = traceback.format_exc()
    finally:
        builtins.__import__ = native_import
        sys.meta_path.remove(tripwire)
    elapsed = time.perf_counter() - clock
    after = pins()
    junit = ART / 'junit.xml'
    counts = None
    if junit.is_file():
        suites = list(ET.parse(junit).getroot().iter('testsuite'))
        counts = {key: sum(int(suite.get(key, 0)) for suite in suites)
                  for key in ('tests', 'errors', 'failures', 'skipped')}
    loaded = sorted(FORBIDDEN & set(sys.modules))
    fixture_module = sys.modules.get('tests.test_native_pipeline_model_free')
    controls = getattr(fixture_module, 'GUARD_CONTROL_OBSERVATIONS', [])
    workflows = getattr(fixture_module, 'WORKFLOW_IMPORT_OBSERVATIONS', [])
    guard_controls_valid = (len(controls) == 3 and all(
        len(value['deliberate_blocked_attempts']) == 1 and value['actual_import_delegations'] == 0
        for value in controls))
    workflow_imports_valid = (len(workflows) == 2 and all(
        value['workflow_attempts'] == value['source_loader_attempts'] == [] for value in workflows))
    child_records = []
    for name in ('child_before.json', 'child_after.json'):
        for path in sorted(temporary.rglob(name)):
            child_records.append({'path': path.relative_to(ART).as_posix(),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'observation': json.loads(path.read_text())})
    child_imports_valid = (len(child_records) == 2 and all(
        value['observation']['import_attempts'] == value['observation']['forbidden_loaded'] == []
        and value['observation']['pid'] != started['pid'] for value in child_records))
    passed = (code == 0 and fatal is None and before == after and not attempts and not loaded
              and guard_controls_valid and workflow_imports_valid and child_imports_valid
              and counts is not None and counts['tests'] > 0
              and not any(counts[key] for key in ('errors', 'failures', 'skipped')))
    receipt = {'status': 'PASS' if passed else 'FAIL', 'exit_code': code, 'fatal_error': fatal,
        'elapsed_seconds': elapsed, 'counts': counts, 'selected_test_modules': TESTS,
        'deselected': DESELECT, 'source_before': before, 'source_after': after,
        'source_stable': before == after, 'inference_import_attempts': attempts,
        'forbidden_inference_loaded': loaded,
        'deliberate_guard_controls': controls, 'actual_workflow_imports': workflows,
        'raw_journal_child_processes': child_records, 'child_imports_valid': child_imports_valid,
        'workflow_imports_valid': workflow_imports_valid, 'guard_controls_valid': guard_controls_valid,
        'source_python_files': sum(name.startswith('src/') for name in before),
        'scope': started['scope'],
        'bounds': {'actual_model_imports': False, 'model_loads': 0, 'inference': False,
                  'gpu_or_host_ram_probes': False, 'training': False,
                  'within_inference_resume_implemented_or_tested': False,
                  'independent_accuracy_or_long_match_qualification': False}}
    with (ART / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({key: receipt[key] for key in ('status', 'exit_code', 'elapsed_seconds',
        'counts', 'source_stable', 'source_python_files', 'inference_import_attempts',
        'forbidden_inference_loaded')}, indent=2))
    if not passed:
        print((ART / 'pytest.txt').read_text(encoding='utf-8')[-16000:])
        if fatal:
            print(fatal)
    return 0 if passed else code or 2


if __name__ == '__main__':
    raise SystemExit(main())
