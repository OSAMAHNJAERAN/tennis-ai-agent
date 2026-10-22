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
    'tests/test_yolo26_crop_provider.py',
    'tests/test_yolo26_crop_pipeline.py',
    'tests/test_yolo26_models.py',
    'tests/test_analytics_capacity_seam.py',
]
DESELECT = [
    'tests/test_yolo26_models.py::test_pipeline_constructs_only_the_bundle_and_reuses_player_pose',
    'tests/test_analytics_capacity_seam.py::test_legacy_analytics_owner_views_and_exports_remain_bounded',
]
EXPECTED = {
    'src/detection/yolo26_adapters.py': '1543506f26dfd4b57cd28bdd754b2d1b6610bf6229385e0d8d9da92068cbbb39',
    'tests/test_yolo26_crop_provider.py': '7a7bbb65c460b14ff2d121e2e19b67f2c9d3f98efe6faef6328fda2c12d1137e',
    'src/pipeline/phase6_pipeline.py': '216b23fa22d1460c04b3685de417dd514dcec53baeee23abd13b4056a645ba2d',
    'tests/test_yolo26_crop_pipeline.py': '8f21ac224e28dabaf692c42cc557e7b99121f1ec50d6f8215c2a8ea94cec7f54',
    'tests/test_analytics_capacity_seam.py': 'd7716acc8e8a40ea5c153303e6b8602482d9b0bf9daf137b5e280739090752f5',
}
FORBIDDEN = {'torch', 'ultralytics'}


def pins():
    paths = set((ROOT / 'src').rglob('*.py'))
    paths.update(ROOT / name for name in TESTS)
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
        'scope': 'Injected selected-model crop adapter, exact pipeline AST blocks/extractor body and adjacent functional contracts only'}
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
    passed = (code == 0 and fatal is None and before == after and not attempts and not loaded
              and counts is not None and counts['tests'] > 0
              and not any(counts[key] for key in ('errors', 'failures', 'skipped')))
    receipt = {'status': 'PASS' if passed else 'FAIL', 'exit_code': code, 'fatal_error': fatal,
        'elapsed_seconds': elapsed, 'counts': counts, 'selected_test_modules': TESTS,
        'deselected': DESELECT, 'source_before': before, 'source_after': after,
        'source_stable': before == after, 'inference_import_attempts': attempts,
        'forbidden_inference_loaded': loaded,
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
