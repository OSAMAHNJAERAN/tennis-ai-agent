"""One-shot retained 600-frame readonly export transaction probe."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
TRACE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_trace_2026-10-06'
SOURCE = TRACE / 'run/600_current'
ANALYSIS = SOURCE / 'analysis'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
MODES = ('ordinary', 'begin_after_enter')
RESULTS_REBIND = {
    'previous': {'bytes': 35337, 'sha256': '30a64418e141d2ab5d847c33aa6fcc0386807ff7a04a5891b5172a904f9c1a55'},
    'current': {'bytes': 35538, 'sha256': '01d8fb7d04edb0c0d5d980d0c79f71b54c8cc3acea8200c9a486d2b2952f02a2'},
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': checksum}


def save(path, value):
    with path.open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def handshake(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    save(temporary, value)
    if path.exists():
        raise FileExistsError(path)
    temporary.rename(path)


def load_file(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def source_bindings():
    previous = json.loads((TRACE / 'run/source_bindings.json').read_text(encoding='utf-8'))
    names = list(previous)
    manifest = json.loads((ANALYSIS / '.result/current.store').read_text(encoding='utf-8'))
    database = ANALYSIS / '.result' / manifest['database']
    for path in (TRACE / 'run/source_bindings.json', TRACE / 'run/report.json',
                 SOURCE / 'child_receipt.json', ANALYSIS / '.result/current.store', database,
                 SOURCE / 'telemetry.csv', SOURCE / 'report.html', Path(__file__),
                 UNIT / 'protocol.json', UNIT / 'PROTOCOL.md', UNIT / 'binding_exception.json'):
        names.append(path.resolve().relative_to(ROOT).as_posix())
    actual = {name: binding(ROOT / name) for name in names}
    for name, expected in previous.items():
        if name == 'src/api/results.py':
            assert expected == RESULTS_REBIND['previous'], 'Unexpected old results source binding'
            assert actual[name] == RESULTS_REBIND['current'], 'Root-reviewed results rebind changed'
        else:
            assert actual[name] == expected, 'Prior sealed source/helper changed: ' + name
    assert binding(database) == {'bytes': manifest['databaseBytes'], 'sha256': manifest['databaseSha256']}
    assert manifest['frames'] == 600
    assert actual[MONITOR.relative_to(ROOT).as_posix()]['sha256'] == '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'
    assert actual[(TRACE / 'helper.py').relative_to(ROOT).as_posix()]['sha256'] == '0d14e16801958c623e8ccc59c673d20c1ed9af72388f08a5e4efec251158d954'
    return actual


def equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            first, second = a.read(65536), b.read(65536)
            if first != second:
                return False
            if not first:
                return True


def child(directory):
    import psutil
    process = psutil.Process()
    identity = {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()}
    report = {'startedAt': stamp(), 'passed': False, 'frames': 600, 'actualWorkloadIdentity': identity,
              'runs': [], 'qualificationEvidence': False, 'inferenceRun': False,
              'scope': 'Same clean child exports unchanged retained synthetic 600-frame result; ordinary reader versus only BEGIN after normal reader entry.',
              'interpretation': 'Instrumented query counts and spans test a read-transaction hypothesis; one ordered two-mode probe is not throughput/variance/full-match qualification. Method spans include Python execute/fetch/cursor close. Source is an immutable owned generation; externally mutable v1 DB semantics are not exercised.',
              'singleVariable': 'After the same ResultDocument constructor, BEGIN is issued on its readonly connection only for begin_after_enter before CSV/HTML. No schema/query/cache/decoder/source changes.'}
    started = time.perf_counter()
    active_document = None
    try:
        handshake(directory / 'actual_identity.json', identity)
        deadline = time.monotonic() + 10
        ack_path = directory / 'actual_identity_sampled.json'
        while not ack_path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Actual interpreter sampling acknowledgment deadline')
            time.sleep(.01)
        acknowledgment = json.loads(ack_path.read_text(encoding='utf-8'))
        assert acknowledgment['identity'] == identity and acknowledgment['sampledByExactIdentity']
        expected = json.loads((RUN / 'source_bindings.json').read_text(encoding='utf-8'))
        assert source_bindings() == expected
        report['sourceBindingsBefore'] = expected
        report['explicitRootReviewedBindingException'] = json.loads((UNIT / 'binding_exception.json').read_text(encoding='utf-8'))
        trace = load_file('retained_result_counters_20261006_transaction', TRACE / 'helper.py')
        from src.api.result_storage import ResultStorage
        from src.api.result_document import ResultDocument
        from src.api.reporting import telemetry_csv_chunks, readable_report_chunks
        manifest = json.loads((ANALYSIS / '.result/current.store').read_text(encoding='utf-8'))
        original_files = sorted(path.relative_to(ANALYSIS).as_posix() for path in ANALYSIS.rglob('*') if path.is_file())
        report['sourceAnalysisFilesBefore'] = original_files
        for mode in MODES:
            output = directory / mode
            output.mkdir()
            counters = trace.KernelCounters(ResultStorage)
            mode_report = {'mode': mode, 'passed': False}
            try:
                with counters:
                    with counters.stage('reader_open'):
                        active_document = ResultDocument(ANALYSIS, manifest['analysisId'])
                        assert active_document.storage._readonly
                        mode_report['normalReaderEnterInTransaction'] = active_document.storage._connection.in_transaction
                        assert mode_report['normalReaderEnterInTransaction'] is False
                        if mode == 'begin_after_enter':
                            active_document.storage._execute('BEGIN')
                        mode_report['beforeExportsInTransaction'] = active_document.storage._connection.in_transaction
                        assert mode_report['beforeExportsInTransaction'] is (mode == 'begin_after_enter')
                    with counters.stage('presentation'):
                        value = active_document.presented(manifest['analysisId'])
                    with counters.stage('csv'):
                        mode_report['csvChunks'] = trace.write_chunks(output / 'telemetry.csv', telemetry_csv_chunks(value))
                    with counters.stage('html'):
                        mode_report['htmlChunks'] = trace.write_chunks(output / 'report.html', readable_report_chunks(value, None))
                    mode_report['afterExportsInTransaction'] = active_document.storage._connection.in_transaction
                    assert mode_report['afterExportsInTransaction'] is (mode == 'begin_after_enter')
                    with counters.stage('reader_close'):
                        active_document.close()
                        assert active_document.storage._connection is None
                    try:
                        active_document.storage.root()
                    except RuntimeError:
                        mode_report['closedHandleRejectsReads'] = True
                    else:
                        raise AssertionError('Closed result reader unexpectedly returned a root')
                    active_document = None
                    del value
                mode_report['stageCounters'] = counters.report()
                csv_sql = mode_report['stageCounters']['csv']['sql']
                assert csv_sql['length_read']['calls'] == 5405 and csv_sql['item_payload_read']['calls'] == 3000
                mode_report['csvQueryPredictionPassed'] = True
                mode_report['exactExports'] = {name: {'exactBytesEqual': equal(output / name, SOURCE / name),
                                                     'actual': binding(output / name), 'retained': binding(SOURCE / name)}
                                              for name in ('telemetry.csv', 'report.html')}
                assert all(row['exactBytesEqual'] for row in mode_report['exactExports'].values())
                assert source_bindings() == expected
                mode_report['passed'] = True
            except BaseException as error:
                mode_report['error'] = type(error).__name__ + ': ' + str(error)
                raise
            finally:
                mode_report['stageCounters'] = counters.report()
                report['runs'].append(mode_report)
        for stage in ('csv', 'html'):
            ordinary, pinned = (row['stageCounters'][stage] for row in report['runs'])
            assert ordinary['sqlMethodCalls'] == pinned['sqlMethodCalls']
            assert all(ordinary['sql'][label]['calls'] == pinned['sql'][label]['calls'] for label in trace.LABELS)
        ordinary_begin = report['runs'][0]['stageCounters']['reader_open']['sql']['begin']['calls']
        pinned_begin = report['runs'][1]['stageCounters']['reader_open']['sql']['begin']['calls']
        assert ordinary_begin == 0 and pinned_begin == 1
        report['equalExportStatementCountsWithOnlyOneBeginAdded'] = True
        report['sourceAnalysisFilesAfter'] = sorted(path.relative_to(ANALYSIS).as_posix() for path in ANALYSIS.rglob('*') if path.is_file())
        assert report['sourceAnalysisFilesAfter'] == original_files
        report['sourceBindingsAfter'] = source_bindings()
        assert report['sourceBindingsAfter'] == expected
        report['sourceReadonlyDatabaseAndExportsUnchanged'] = True
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        if active_document is not None:
            active_document._close_preserving(error)
        raise
    finally:
        primary = sys.exception()
        try:
            memory = process.memory_info()
            report.update({'completedAt': stamp(), 'childWallSeconds': time.perf_counter() - started,
                           'windowsPeakWorkingSetBytes': getattr(memory, 'peak_wset', None),
                           'windowsPeakPrivateCommitBytes': getattr(memory, 'peak_pagefile', None),
                           'modelsImported': [name for name in ('torch', 'torchvision', 'ultralytics', 'onnxruntime') if name in sys.modules]})
            if report['modelsImported']:
                report['passed'] = False
            save(directory / 'child_receipt.json', report)
            print(json.dumps({'passed': report['passed'], 'completedModes': len(report['runs']), 'error': report.get('error')}), flush=True)
        except BaseException as receipt_error:
            if primary is None:
                raise
            primary.add_note('Transaction probe receipt save also failed: ' + str(receipt_error))
    return 0 if report['passed'] else 1


def parent():
    import psutil
    RUN.mkdir(exist_ok=False)
    report = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False,
              'qualificationEvidence': False, 'inferenceRun': False, 'frames': 600}
    process = owned = monitor = None
    started = time.perf_counter()
    try:
        protocol = json.loads((UNIT / 'protocol.json').read_text(encoding='utf-8'))
        assert protocol['maximumChildWallSeconds'] == 30 and protocol['maximumOwnedTreeRssBytes'] == 256 * 1024 * 1024
        before = source_bindings()
        prepared_path = UNIT / 'prepared_source_bindings.json'
        assert before == json.loads(prepared_path.read_text(encoding='utf-8')), 'Reviewed prepared bindings changed'
        report['sourceBindingsBefore'] = before
        report['preparedBindingsSha256'] = binding(prepared_path)['sha256']
        save(RUN / 'source_bindings.json', before)
        with (RUN / 'executed_helper.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        assert binding(RUN / 'executed_helper.py') == binding(Path(__file__))
        assert psutil.virtual_memory().available >= protocol['preflightAvailableRamBytes'], 'Insufficient preflight RAM; no retry'
        assert shutil.disk_usage(UNIT).free >= protocol['minimumFreeDiskBytes'], 'Insufficient preflight disk; no retry'
        monitor = load_file('transaction_probe_owned_monitor_20261006', MONITOR)
        directory = RUN / '600_current'
        directory.mkdir()
        report.update({'samples': 0, 'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
                       'minimumSampledAvailableRamBytes': psutil.virtual_memory().available, 'allPrivateSamplesAvailable': True})
        started = time.perf_counter()
        with (directory / 'stdout.txt').open('x', encoding='utf-8') as log, \
                (directory / 'samples.jsonl').open('x', encoding='utf-8', newline='') as samples:
            process = subprocess.Popen([str(CLEAN), str(Path(__file__)), '--child', str(directory)], cwd=ROOT,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log,
                                       stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            report['status'] = 'RUNNING'
            owned = monitor.OwnedProcessTree(process)
            low_ram = 0
            while process.poll() is None:
                rows = owned.sample()
                identity_path, ack_path = directory / 'actual_identity.json', directory / 'actual_identity_sampled.json'
                if identity_path.exists() and not ack_path.exists():
                    actual = json.loads(identity_path.read_text(encoding='utf-8'))
                    if owned.observed_workload(actual):
                        handshake(ack_path, {'at': stamp(), 'identity': actual, 'sampledByExactIdentity': True})
                private = [row['memoryBytes'].get('private') for row in rows]
                available = psutil.virtual_memory().available
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - started, 'processes': rows,
                          'ownedTreeRssBytes': sum(row['memoryBytes']['rss'] for row in rows),
                          'ownedTreePrivateBytes': sum(value for value in private if value is not None),
                          'allPrivateValuesAvailable': all(value is not None for value in private), 'availableRamBytes': available}
                samples.write(json.dumps(sample, allow_nan=False) + '\n')
                samples.flush()
                report['samples'] += 1
                report['peakSampledOwnedTreeRssBytes'] = max(report['peakSampledOwnedTreeRssBytes'], sample['ownedTreeRssBytes'])
                report['peakSampledOwnedTreePrivateBytes'] = max(report['peakSampledOwnedTreePrivateBytes'], sample['ownedTreePrivateBytes'])
                report['minimumSampledAvailableRamBytes'] = min(report['minimumSampledAvailableRamBytes'], available)
                report['allPrivateSamplesAvailable'] &= sample['allPrivateValuesAvailable']
                low_ram = low_ram + 1 if available < protocol['minimumAvailableRamBytes'] else 0
                reason = ('Owned RSS safeguard' if sample['ownedTreeRssBytes'] > protocol['maximumOwnedTreeRssBytes'] else
                          'Available RAM safeguard' if low_ram >= 5 else
                          'Wall safeguard' if sample['elapsedSeconds'] >= 30 else None)
                if reason:
                    report['safeguardStop'] = reason
                    raise RuntimeError(reason + '; preserve first failure, no retry')
                time.sleep(.25)
        report['exitCode'] = process.returncode
        assert process.returncode == 0, 'Transaction probe child failed'
        receipt = json.loads((directory / 'child_receipt.json').read_text(encoding='utf-8'))
        assert receipt['passed'] and receipt['modelsImported'] == []
        assert owned.observed_workload(receipt['actualWorkloadIdentity'])
        report['actualWorkloadWasSampled'] = True
        report['remainingOwnedIdentities'] = owned.wait_for_retirement()
        assert not report['remainingOwnedIdentities']
        report['allRememberedOwnedIdentitiesRetired'] = True
        report['childReceipt'] = binding(directory / 'child_receipt.json')
        report['passed'] = True
        report['status'] = 'COMPLETED'
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        report['status'] = 'FAILED' if process is not None else 'NOT_STARTED'
        if process is not None and monitor is not None:
            try:
                report['terminationActions'] = monitor.stop_owned(process, owned)
                if owned is not None:
                    report['remainingOwnedIdentitiesAfterCleanup'] = owned.wait_for_retirement()
                    report['allRememberedOwnedIdentitiesRetired'] = not report['remainingOwnedIdentitiesAfterCleanup']
            except BaseException as cleanup_error:
                error.add_note('Owned probe cleanup also failed: ' + str(cleanup_error))
                report['cleanupError'] = str(cleanup_error)
    finally:
        report['completedAt'] = stamp()
        report['monitoredWallSeconds'] = time.perf_counter() - started
        if owned is not None:
            report['ownedIdentities'] = owned.identities()
        try:
            report['sourceBindingsAfter'] = source_bindings()
            report['sourcesUnchanged'] = report.get('sourceBindingsBefore') == report['sourceBindingsAfter']
            report['preparedBindingsUnchanged'] = (report.get('preparedBindingsSha256') ==
                                                   binding(UNIT / 'prepared_source_bindings.json')['sha256'])
            if not report['sourcesUnchanged'] or not report['preparedBindingsUnchanged']:
                report['passed'] = False
                report['status'] = 'FAILED' if process is not None else 'NOT_STARTED'
        except BaseException as binding_error:
            report['sourceBindingError'] = str(binding_error)
            report['passed'] = False
            report['status'] = 'FAILED' if process is not None else 'NOT_STARTED'
        save(RUN / 'report.json', report)
        print(json.dumps({'passed': report['passed'], 'status': report['status'], 'error': report.get('error')}), flush=True)
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', metavar='FRESH_CASE_DIRECTORY')
    arguments = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if arguments.child:
        directory = Path(arguments.child).resolve()
        assert directory == RUN / '600_current' and directory.is_dir()
        assert not (directory / 'child_receipt.json').exists()
        return child(directory)
    assert sys.prefix == sys.base_prefix, 'Use base Python313 for monitoring parent'
    return parent()


if __name__ == '__main__':
    raise SystemExit(main())
