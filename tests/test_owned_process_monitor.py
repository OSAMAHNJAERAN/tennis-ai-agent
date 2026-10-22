"""Exercise failure teardown without launching models or unrelated processes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest


def monitor_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit/measure_candidate_storage.py'
    spec = importlib.util.spec_from_file_location('owned_monitor_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Redirector:
    def __init__(self, fallback_error=None):
        self.returncode = None
        self.calls = []
        self.fallback_error = fallback_error

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append('terminate')
        if self.fallback_error:
            raise self.fallback_error
        self.returncode = 15

    def wait(self, timeout):
        self.calls.append(('wait', timeout))
        return self.returncode


@pytest.mark.parametrize('fallback_fails', [False, True])
def test_redirector_fallback_runs_after_descendant_cleanup_failure(fallback_fails):
    module = monitor_module()
    original = RuntimeError('Descendant cleanup failed')
    retirement_calls = []

    def fail():
        raise original

    def retired():
        retirement_calls.append(True)
        return []

    owned = SimpleNamespace(terminate=fail, wait_for_retirement=retired)
    redirector = Redirector(OSError('Fallback failed') if fallback_fails else None)
    with pytest.raises(RuntimeError, match='Descendant cleanup failed') as caught:
        module.stop_owned(redirector, owned)
    assert caught.value is original
    assert redirector.calls[0] == 'terminate'
    assert retirement_calls == [True]
    if fallback_fails:
        assert any('Fallback failed' in note for note in original.__notes__)
    else:
        assert redirector.returncode == 15
        assert redirector.calls == ['terminate', ('wait', 10)]


def test_live_owned_descendant_cannot_be_reported_as_retired():
    module = monitor_module()
    owned = SimpleNamespace(terminate=lambda: [], wait_for_retirement=lambda: [{'pid': 42}])
    redirector = Redirector()
    with pytest.raises(RuntimeError, match='descendants remain'):
        module.stop_owned(redirector, owned)
    assert redirector.returncode == 15


def test_termination_handles_process_exiting_before_second_signal():
    module = monitor_module()
    tree = module.OwnedProcessTree.__new__(module.OwnedProcessTree)

    class ExitingProcess:
        pid = 42
        def terminate(self):
            return None
        def create_time(self):
            raise psutil.NoSuchProcess(self.pid)

    process = ExitingProcess()
    record = {'pid': 42, 'process': process, 'createTimeUnixSeconds': 1, 'depth': 1}
    tree._owned = {(42, 1): record}
    tree.discover = lambda: None
    checks = iter([True, False])
    tree._matches = lambda record: next(checks)
    tree._psutil = SimpleNamespace(NoSuchProcess=psutil.NoSuchProcess,
                                   wait_procs=lambda values, timeout: ([], [process]))
    assert tree.terminate()[0]['signal'] == 'terminate'
