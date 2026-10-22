"""Frozen model-free native inference integration checks; no resource admission."""
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
    "tests/test_native_pipeline_model_free.py",
    "tests/test_native_inference_session.py",
    "tests/test_worker_partial_inference.py",
    "tests/test_worker_result_publication.py",
    "tests/test_analysis_resume_identity.py",
    "tests/test_pipeline_checkpoint.py",
    "tests/test_checkpoint_process_restart.py",
    "tests/test_native_pipeline_process_resume.py",
]
DESELECT = [
    'tests/test_yolo26_models.py::test_pipeline_constructs_only_the_bundle_and_reuses_player_pose',
    'tests/test_analytics_capacity_seam.py::test_legacy_analytics_owner_views_and_exports_remain_bounded',
]
EXPECTED = {
    "tests/test_ball_pass_decoder.py": "17a811c438857fdc94d6d313a791c9a903236c632aa129c9a9a854370b1589a6",
    "tests/test_video_frame_sequence.py": "3f910c6636cf534cb37e1087d2ade140568fdd5481468f575e59b1f97b74dc99",
    "src/utils/video_frame_sequence.py": "934a157eb4a11f0fd209b1120f31da74e0ab2c3b4f165e5ebd06a3434b5fda40",
    "src/detection/yolo26_adapters.py": "1543506f26dfd4b57cd28bdd754b2d1b6610bf6229385e0d8d9da92068cbbb39",
    "tests/test_yolo26_crop_provider.py": "7a7bbb65c460b14ff2d121e2e19b67f2c9d3f98efe6faef6328fda2c12d1137e",
    "src/pipeline/phase6_pipeline.py": "1086c25de98f9e5c8a11986ef94e00095ca5499c4b6d2c707422318351ac31ff",
    "tests/test_yolo26_crop_pipeline.py": "c7cb23baea99852c45b3e846b9e2f0ba6b9bc7ff72c8b7a287f8c6fb690d70c0",
    "tests/test_analytics_capacity_seam.py": "a2f1ef723f1fc60b1c1e269b3b3318d2c39deab3b749af7b6a3985817062713c",
    "src/utils/ball_inference_journal.py": "f8d75a15e53cdabdbdec0353d2feee220e14b4954a9ce332336e47d443907113",
    "src/utils/native_inference_prefix.py": "ae238c38109512366bea86ef5fe88cc79fa6c53eba488b3ea9dee3a8c2967a97",
    "tests/test_yolo26_models.py": "5118ef9158f89270150a235a789d8ed0d1f5851e74c7b1a8ee2c31b087e65bad",
    "tests/test_native_prefix_audits.py": "c6452d2c8a01a4d9cdfc7e015e7d0a6a384ddbeee81aff3affc9bf171db63fff",
    "tests/test_ball_inference_journal.py": "89797fbcf60f87b30e7e405505ba5890ad99e99d6e4ef1d8b334d63d462ef81f",
    "tests/test_native_inference_prefix.py": "4634437ba7e95af19dade9f7f460ccb681ded94630adf611915dc8e15d1ac745",
    "tests/test_native_pipeline_model_free.py": "ca8f69d775398fe77c5b69bf31a97e9ffe5c42caf5f7c59bbdf1cede43432e3d",
    "tests/helpers/native_pipeline_source.py": "b4b99dcff081a22d2beda8315d3a14ff7d9405b17f20ed1cd7d42e808efc9c47",
    "configs/production/yolo26.yaml": "8106e134eea5a520c10747a12921d2352643e7df7d3cdb1d30ef29ae94cee9fb",
    "configs/production/yolo26.artifacts.json": "8a902c6fe9eecbe97799b48c51e2ea55ca92355e44fd7b39527d7665af30f0cb",
    "configs/production/botsort_no_reid.yaml": "c83cbe84e582ac9c7973cdc99809efd9e01565293720a9a427d06d1d6b12abf8",
    "tests/test_native_inference_session.py": "f785a8dbf576b40d5d47b00989dc2e3f7ff44f88b496c5977bbe12bfb5b9ce59",
    "tests/test_worker_partial_inference.py": "8fe4a444f6106e37f2259d76e918c5327a4c7be8559b4e1ee50b289b7af57da5",
    "tests/test_worker_result_publication.py": "58dc675338c6906d21bddec2b0fb3cabd084198a085b8dfcaa0806de3bee3560",
    "tests/test_analysis_resume_identity.py": "1e6bdf5df58111735e6a44efc998a226ce271a3ab2e70bd8345b86284bbd48dd",
    "tests/test_pipeline_checkpoint.py": "78aa40d04328d9aa12dca6556ec33e55e83c1d3bf656083b55d637bc61fd0a44",
    "tests/test_checkpoint_process_restart.py": "8b214230a2d138827bee295040d9fffce1d282200cb21559be1a01e23d907668",
    "src/utils/native_inference_session.py": "d694abc98138b9409197fcbb908d98b757d836ac285fd02d1dc511a960c3c19b",
    "src/api/worker.py": "5338aa08a599bfd54677f6abb5cff7153ed53fe9b4a4cff422834dcd85a91ce9",
    "tests/helpers/checkpoint_worker_child.py": "ea468341210d76a96d039728d657006e95779f7b0a0442ebca5fa64bbd9677f2",
    "tests/test_native_pipeline_process_resume.py": "63e8e5ac3e8077690b03e035c1c1e18fa1ea6e208060c6841f33cdb107d5db46",
    "tests/helpers/native_pipeline_resume_child.py": "30c8cdfa31ddf0d977c5406a897b454f0b81a4a1074301059ffff8134a558761"
}
FORBIDDEN = {'ultralytics', 'torchvision', 'torch'}


def pins():
    paths = set((ROOT / 'src').rglob('*.py'))
    paths.update(ROOT / name for name in TESTS)
    paths.update(ROOT / name for name in ['configs/production/yolo26.yaml', 'configs/production/yolo26.artifacts.json', 'configs/production/botsort_no_reid.yaml'])
    paths.update(ROOT / name for name in EXPECTED)
    paths.update((ROOT / 'pytest.ini', Path(__file__).resolve()))
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
        'scope': 'Native completed-player prefix and raw-ball continuation, normalized source seals, queue and completed-stage precedence, actual separate-process worker crash/resume plus software/codecs with injected model boundaries; no actual selected-model inference'}
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
    recovery = getattr(fixture_module, 'RECOVERY_IMPORT_OBSERVATIONS', [])
    recovery_imports_valid = (len(recovery) == 5 and all(
        value['workflow_attempts'] == value['source_loader_attempts'] == [] for value in recovery))
    process_module = sys.modules.get('tests.test_native_pipeline_process_resume')
    process_recovery = getattr(process_module, 'PROCESS_OBSERVATIONS', [])
    process_recovery_valid = (len(process_recovery) == 1 and
        process_recovery[0]['recovery_child_exit_codes'] == [73, 0] and
        process_recovery[0]['actual_model_loads_or_forwards'] == 0 and
        process_recovery[0]['forbidden_import_attempts_or_loaded_runtimes'] == [])
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
              and guard_controls_valid and workflow_imports_valid and child_imports_valid and recovery_imports_valid and process_recovery_valid
              and counts is not None and counts['tests'] > 0
              and not any(counts[key] for key in ('errors', 'failures', 'skipped')))
    receipt = {'status': 'PASS' if passed else 'FAIL', 'exit_code': code, 'fatal_error': fatal,
        'elapsed_seconds': elapsed, 'counts': counts, 'selected_test_modules': TESTS,
        'deselected': DESELECT, 'source_before': before, 'source_after': after,
        'source_stable': before == after, 'inference_import_attempts': attempts,
        'forbidden_inference_loaded': loaded,
        'deliberate_guard_controls': controls, 'actual_workflow_imports': workflows, 'recovery_workflow_imports': recovery,
        'recovery_imports_valid': recovery_imports_valid,
        'actual_worker_process_recovery': process_recovery, 'process_recovery_valid': process_recovery_valid,
        'raw_journal_child_processes': child_records, 'child_imports_valid': child_imports_valid,
        'workflow_imports_valid': workflow_imports_valid, 'guard_controls_valid': guard_controls_valid,
        'source_python_files': sum(name.startswith('src/') for name in before),
        'scope': started['scope'],
        'bounds': {'actual_model_imports': False, 'model_loads': 0, 'inference': False,
                  'gpu_or_host_ram_probes': False, 'training': False,
                  'within_inference_software_resume_with_injected_models': True,
                  'actual_selected_model_or_long_match_resume': False,
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
