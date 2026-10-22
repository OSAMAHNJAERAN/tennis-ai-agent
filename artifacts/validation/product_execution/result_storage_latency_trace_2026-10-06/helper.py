"""Prespecified 600-frame result kernel counter diagnostic; do not auto-run."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
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
PRIOR = ROOT / 'artifacts/validation/product_execution/result_storage_measurement_2026-10-06'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_baseline_2026-10-06'
SEAL = ROOT / 'artifacts/validation/product_execution/finalize_result_storage_evidence_2026-10-06.json'
CLEAN = ROOT / 'artifacts/validation/product_installation/clean_env/Scripts/python.exe'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
FRAMES = 600
EXPECTED = {
    PRIOR / 'helper.py': '8a1585543a904826fca354f5fa727debfd7b5071f330cae9cfe8c41eb0f446dc',
    MONITOR: '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39',
    SEAL: '05f2cdce18ece5066e4507a2c846e894eeaf7fd800f83f2450c42431bde9e19f',
    BASELINE / 'src/api/results.py': 'd043172fa3674fc311176f97dc979e1660e1af8a17f8dec4f2dca34f6e33b5d7',
    BASELINE / 'src/api/reporting.py': 'c39639f8033fd68ee91b39f2e9a27332e570fd776fb00201cdd46283df7e1631',
}
STAGES = ('storage_open', 'build', 'publication', 'writer_close', 'reader_open',
          'presentation', 'json', 'csv', 'html', 'reader_close')
LABELS = ('length_read', 'item_payload_read', 'root_read', 'format_read', 'series_insert',
          'item_write', 'refs_delete', 'refs_insert', 'length_update', 'root_write',
          'root_refs_delete', 'root_refs_insert', 'begin', 'savepoint', 'release',
          'rollback', 'pragma', 'schema', 'compaction', 'vacuum', 'other')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    with Path(path).open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': Path(path).stat().st_size, 'sha256': checksum}


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
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
    # The previous reviewed snapshot binds the exact production dependencies,
    # frozen references, fixture generator, clean interpreter and monitor.
    previous = json.loads((PRIOR / 'prepared_source_bindings.json').read_text(encoding='utf-8'))
    names = list(previous)
    for name in (PRIOR / 'prepared_source_bindings.json', SEAL, Path(__file__),
                 UNIT / 'protocol.json', UNIT / 'PROTOCOL.md'):
        names.append(name.resolve().relative_to(ROOT).as_posix())
    actual = {name: binding(ROOT / name) for name in names}
    for name, expected in previous.items():
        if actual[name] != expected:
            raise AssertionError('Preserved source/fixture binding changed: ' + name)
    for path, expected in EXPECTED.items():
        if actual[path.relative_to(ROOT).as_posix()]['sha256'] != expected:
            raise AssertionError('Frozen reference, sealed source or verified helper changed: ' + str(path))
    return actual


def files_equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            first, second = a.read(65536), b.read(65536)
            if first != second:
                return False
            if not first:
                return True


def write_chunks(path, chunks):
    count, maximum = 0, 0
    with path.open('x', encoding='utf-8', newline='') as stream:
        for chunk in chunks:
            stream.write(chunk)
            count += 1
            maximum = max(maximum, len(chunk))
    return {'chunks': count, 'maximumChunkCharacters': maximum}


def sql_label(sql):
    # Only source-owned unexpanded templates are inspected; no parameter,
    # evidence payload, value, SQL log or unbounded statement set is retained.
    if sql.startswith('SELECT length FROM series'):
        return 'length_read'
    if sql.startswith('SELECT payload FROM items'):
        return 'item_payload_read'
    if sql.startswith('SELECT payload FROM root'):
        return 'root_read'
    if sql.startswith('SELECT version FROM storage_format'):
        return 'format_read'
    if sql.startswith('INSERT INTO series'):
        return 'series_insert'
    if sql.startswith('INSERT OR REPLACE INTO items'):
        return 'item_write'
    if sql.startswith('DELETE FROM refs WHERE parent_series=?'):
        return 'refs_delete'
    if sql.startswith('INSERT OR IGNORE INTO refs'):
        return 'refs_insert'
    if sql.startswith('UPDATE series SET length=length+1'):
        return 'length_update'
    if sql.startswith('INSERT OR REPLACE INTO root'):
        return 'root_write'
    if sql == 'DELETE FROM root_refs':
        return 'root_refs_delete'
    if sql.startswith('INSERT OR IGNORE INTO root_refs'):
        return 'root_refs_insert'
    if sql == 'BEGIN':
        return 'begin'
    if sql.startswith('SAVEPOINT '):
        return 'savepoint'
    if sql.startswith('RELEASE '):
        return 'release'
    if sql.startswith('ROLLBACK TO '):
        return 'rollback'
    if sql.startswith('PRAGMA '):
        return 'pragma'
    if sql.startswith(('CREATE TABLE ', 'INSERT INTO storage_format')):
        return 'schema'
    if sql == 'VACUUM':
        return 'vacuum'
    if sql.startswith(('CREATE TEMP TABLE ', 'INSERT INTO reachable ',
                       'DELETE FROM items WHERE series_id NOT IN ',
                       'DELETE FROM refs WHERE parent_series NOT IN ',
                       'DELETE FROM series WHERE id NOT IN ', 'DROP TABLE reachable')):
        return 'compaction'
    return 'other'


def statistic():
    return {'calls': 0, 'totalNanoseconds': 0, 'maximumNanoseconds': 0, 'failures': 0,
            'nestedCallsExcludedFromTiming': 0}


class KernelCounters:
    """Fixed metadata; forwarding wrappers do not change values/cleanup."""
    def __init__(self, storage_class):
        self.storage_class = storage_class
        self.current = None
        self.depth = {'_encode': 0, '_decode': 0}
        self.original = {}
        self.data = {stage: {'elapsedNanoseconds': 0, 'sql': {label: statistic() for label in LABELS},
                             'encode': statistic(), 'decode': statistic()} for stage in STAGES}

    @contextmanager
    def stage(self, name):
        if name not in STAGES or self.current is not None:
            raise RuntimeError('Invalid or nested diagnostic stage')
        self.current = name
        begin = time.perf_counter_ns()
        try:
            yield
        finally:
            self.data[name]['elapsedNanoseconds'] += time.perf_counter_ns() - begin
            self.current = None

    def sql_wrapper(self, method, original):
        def forwarded(storage, sql, parameters=()):
            if self.current is None:
                return original(storage, sql, parameters)
            target = self.data[self.current]['sql'][sql_label(sql)]
            target['calls'] += 1
            begin = time.perf_counter_ns()
            try:
                return original(storage, sql, parameters)
            except BaseException:
                target['failures'] += 1
                raise
            finally:
                elapsed = time.perf_counter_ns() - begin
                target['totalNanoseconds'] += elapsed
                target['maximumNanoseconds'] = max(target['maximumNanoseconds'], elapsed)
        return forwarded

    def codec_wrapper(self, method, original):
        def forwarded(storage, *arguments, **keywords):
            if self.current is None:
                return original(storage, *arguments, **keywords)
            target = self.data[self.current]['encode' if method == '_encode' else 'decode']
            outer = self.depth[method] == 0
            self.depth[method] += 1
            if outer:
                target['calls'] += 1
                begin = time.perf_counter_ns()
            else:
                target['nestedCallsExcludedFromTiming'] += 1
            try:
                return original(storage, *arguments, **keywords)
            except BaseException:
                if outer:
                    target['failures'] += 1
                raise
            finally:
                self.depth[method] -= 1
                if outer:
                    elapsed = time.perf_counter_ns() - begin
                    target['totalNanoseconds'] += elapsed
                    target['maximumNanoseconds'] = max(target['maximumNanoseconds'], elapsed)
        return forwarded

    def __enter__(self):
        for method in ('_one', '_execute', '_encode', '_decode'):
            original = getattr(self.storage_class, method)
            self.original[method] = original
            wrapper = (self.sql_wrapper(method, original) if method in ('_one', '_execute') else
                       self.codec_wrapper(method, original))
            setattr(self.storage_class, method, wrapper)
        return self

    def __exit__(self, *_error):
        for method, original in self.original.items():
            setattr(self.storage_class, method, original)

    def report(self):
        value = {}
        for stage, data in self.data.items():
            assigned = sum(row['totalNanoseconds'] for row in data['sql'].values())
            assigned += data['encode']['totalNanoseconds'] + data['decode']['totalNanoseconds']
            elapsed = data['elapsedNanoseconds']
            value[stage] = {**data, 'elapsedSeconds': elapsed / 1e9, 'kernelAssignedNanoseconds': assigned,
                            'kernelAssignedSeconds': assigned / 1e9,
                            'unassignedNanoseconds': elapsed - assigned,
                            'unassignedSeconds': (elapsed - assigned) / 1e9,
                            'sqlMethodCalls': sum(row['calls'] for row in data['sql'].values()),
                            'kernelAssignedFractionOfInstrumentedStage': assigned / elapsed if elapsed else None}
        return value


def child(directory):
    import psutil
    process = psutil.Process()
    actual = {'pid': process.pid, 'createTimeUnixSeconds': process.create_time()}
    report = {'startedAt': stamp(), 'passed': False, 'frames': FRAMES, 'actualWorkloadIdentity': actual,
              'qualificationEvidence': False, 'inferenceRun': False, 'pythonVersion': sys.version,
              'scope': 'One synthetic 600-frame production-kernel counter trace; no footage or inference.',
              'instrumentation': 'Only this isolated interpreter patches four ResultStorage methods; fixed labels, depth guards and forwarding originals. No production file edit or global JSON patch.',
              'unassignedCostMeaning': 'Includes parser grammar/JSONDecoder, stdlib private json.loads/json.dumps outside codec methods, public strict JSON encoding and key encoding, repeated record validation, charts/CSV formatting, file I/O, commit work not exposed by these methods, and instrumentation overhead outside measured wrappers. It is not proven wasted time.',
              'overheadLimits': 'Instrumented times are not benchmark throughput and cannot be directly subtracted from prior uninstrumented observations. Per-call clocks/wrappers and per-node depth guards add overhead. Kernel spans are disjoint top-level encode/decode and _one/_execute; nested codec spans are counted but not double-timed.',
              'countsAreMechanismEvidence': True}
    counters = None
    start = time.perf_counter()
    try:
        handshake(directory / 'actual_identity.json', actual)
        deadline = time.monotonic() + 10
        ack_path = directory / 'actual_identity_sampled.json'
        while not ack_path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Actual interpreter was not sampled/acknowledged within 10 seconds')
            time.sleep(.01)
        ack = json.loads(ack_path.read_text(encoding='utf-8'))
        assert ack['identity'] == actual and ack['sampledByExactIdentity'] is True
        expected = json.loads((RUN / 'source_bindings.json').read_text(encoding='utf-8'))
        assert source_bindings() == expected
        report['sourceBindingsBefore'] = expected
        fixture = load_file('retained_result_fixture_20261006_trace', PRIOR / 'helper.py')
        frozen_results = load_file('frozen_result_20261006_trace', BASELINE / 'src/api/results.py')
        frozen_reporting = load_file('frozen_report_20261006_trace', BASELINE / 'src/api/reporting.py')
        from src.api.results import build_result
        from src.api.result_storage import ResultStorage
        from src.api.result_document import ResultDocument, publish_result
        from src.api.reporting import telemetry_csv_chunks, readable_report_chunks
        raw = directory / 'raw_fixture'
        names, job = fixture.generate_inputs(raw, FRAMES)
        report['rawFixtureBefore'] = {name: binding(raw / name) for name in names}
        analysis = directory / 'analysis'
        analysis.mkdir()
        for name in names:
            shutil.copyfile(raw / name, analysis / name)
            assert files_equal(raw / name, analysis / name)
        # Frozen code sees the same pre-publication artifact filenames/bytes.
        # The reference lives outside analysis, so artifacts lists stay exact.
        frozen = frozen_results.build_result(raw, job)
        write_chunks(directory / 'frozen_result.json', json.JSONEncoder(allow_nan=False).iterencode(frozen))
        write_chunks(directory / 'frozen_summary.json', json.JSONEncoder(allow_nan=False).iterencode(frozen['summary']))
        write_chunks(directory / 'frozen_telemetry.csv', frozen_reporting.telemetry_csv_chunks(frozen))
        write_chunks(directory / 'frozen_report.html', iter((frozen_reporting.readable_report(frozen, None),)))
        del frozen
        gc.collect()
        report['referencePreparedBeforeInstrumentation'] = True
        counters = KernelCounters(ResultStorage)
        storage = ResultStorage(analysis)
        document = None
        with counters:
            try:
                with counters.stage('storage_open'):
                    storage.__enter__()
                with counters.stage('build'):
                    value = build_result(analysis, job, _storage=storage)
                with counters.stage('publication'):
                    publish_result(storage, value, analysis)
                with counters.stage('writer_close'):
                    storage.__exit__(None, None, None)
                del value
                gc.collect()
                with counters.stage('reader_open'):
                    document = ResultDocument(analysis, job['id'])
                with counters.stage('presentation'):
                    value = document.presented(job['id'])
                with counters.stage('json'):
                    report['jsonChunks'] = write_chunks(directory / 'read_result.json', document.json_chunks(value))
                with counters.stage('csv'):
                    report['csvChunks'] = write_chunks(directory / 'telemetry.csv', telemetry_csv_chunks(value))
                with counters.stage('html'):
                    report['htmlChunks'] = write_chunks(directory / 'report.html', readable_report_chunks(value, None))
                with counters.stage('reader_close'):
                    document.close()
                document = None
                del value
            except BaseException as error:
                # Diagnostic cleanup has exactly the production primary-error
                # rules: no retry or success cleanup after a failed builder.
                if document is not None:
                    document._close_preserving(error)
                storage.__exit__(type(error), error, error.__traceback__)
                raise
        report['stageCounters'] = counters.report()
        comparisons = (
            ('savedJson', analysis / 'result.json', directory / 'frozen_result.json'),
            ('readJson', directory / 'read_result.json', directory / 'frozen_result.json'),
            ('summary', analysis / 'summary.json', directory / 'frozen_summary.json'),
            ('csv', directory / 'telemetry.csv', directory / 'frozen_telemetry.csv'),
            ('html', directory / 'report.html', directory / 'frozen_report.html'),
        )
        report['exactComparisons'] = {name: {'exactBytesEqual': files_equal(actual_path, reference_path),
                                          'actual': binding(actual_path), 'frozen': binding(reference_path)}
                                      for name, actual_path, reference_path in comparisons}
        assert all(row['exactBytesEqual'] for row in report['exactComparisons'].values())
        assert report['exactComparisons']['savedJson']['actual'] == report['exactComparisons']['readJson']['actual']
        report['rawFixtureAfter'] = {name: binding(raw / name) for name in names}
        report['analysisInputAfter'] = {name: binding(analysis / name) for name in names}
        assert report['rawFixtureBefore'] == report['rawFixtureAfter'] == report['analysisInputAfter']
        report['sourceBindingsAfter'] = source_bindings()
        assert report['sourceBindingsAfter'] == expected
        report['mechanismPredictions'] = {
            'csvDenseLengthSelectExpected': 9 * FRAMES + 5,
            'csvDensePayloadSelectExpected': 5 * FRAMES,
            'csvDenseSelectExpectedTotal': 14 * FRAMES + 5,
            'jsonDenseArrayCountExpected': 12,
            'jsonDenseRowSelectLowerBound': 24 * FRAMES,
            'htmlDenseRowSelectLowerBound': 16 * FRAMES,
            'meaning': 'Static source predictions for this exact fixture; observed counters establish whether they hold, not throughput.'}
        observed = report['stageCounters']['csv']['sql']
        report['csvPredictedPointQueryCostObserved'] = (observed['length_read']['calls'] == 9 * FRAMES + 5 and
                                                       observed['item_payload_read']['calls'] == 5 * FRAMES)
        # A failed mechanism prediction is useful evidence, not silently green.
        assert report['csvPredictedPointQueryCostObserved'], 'CSV static query prediction did not match production counters'
        assert all(stage['unassignedNanoseconds'] >= 0 for stage in report['stageCounters'].values())
        report['passed'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        primary = sys.exception()
        try:
            if counters is not None:
                report['stageCounters'] = counters.report()
            memory = process.memory_info()
            report.update({'completedAt': stamp(), 'childWallSeconds': time.perf_counter() - start,
                           'windowsPeakWorkingSetBytes': getattr(memory, 'peak_wset', None),
                           'windowsPeakPrivateCommitBytes': getattr(memory, 'peak_pagefile', None),
                           'modelsImported': [name for name in ('torch', 'torchvision', 'ultralytics', 'onnxruntime') if name in sys.modules]})
            if report['modelsImported']:
                report['passed'] = False
            save(directory / 'child_receipt.json', report)
            print(json.dumps({'passed': report['passed'], 'frames': FRAMES, 'error': report.get('error')}), flush=True)
        except BaseException as receipt_error:
            if primary is None:
                raise
            primary.add_note('Diagnostic receipt save also failed: ' + str(receipt_error))
    return 0 if report['passed'] else 1


def parent():
    import psutil
    RUN.mkdir(exist_ok=False)
    report = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False,
              'frames': FRAMES, 'qualificationEvidence': False, 'inferenceRun': False,
              'scope': 'One instrumented synthetic diagnostic child; no benchmark repeat or live product.'}
    process = owned = monitor = None
    begin = time.perf_counter()
    try:
        protocol = json.loads((UNIT / 'protocol.json').read_text(encoding='utf-8'))
        assert protocol['frames'] == FRAMES and protocol['maximumChildWallSeconds'] == 60
        before = source_bindings()
        reviewed = UNIT / 'prepared_source_bindings.json'
        assert before == json.loads(reviewed.read_text(encoding='utf-8')), 'Prepared reviewed sources changed; refuse launch'
        report['preparedBindingsSha256'] = binding(reviewed)['sha256']
        report['sourceBindingsBefore'] = before
        report['baseInterpreter'] = {'path': sys.executable, **binding(Path(sys.executable)), 'pythonVersion': sys.version}
        save(RUN / 'source_bindings.json', before)
        with (RUN / 'executed_helper.py').open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
        assert binding(RUN / 'executed_helper.py') == binding(Path(__file__))
        assert psutil.virtual_memory().available >= protocol['preflightAvailableRamBytes'], 'Insufficient preflight RAM; no retry'
        assert shutil.disk_usage(UNIT).free >= protocol['minimumFreeDiskBytes'], 'Insufficient free disk; no retry'
        monitor = load_file('result_trace_owned_monitor_20261006', MONITOR)
        directory = RUN / '600_current'
        directory.mkdir()
        report.update({'samples': 0, 'peakSampledOwnedTreeRssBytes': 0, 'peakSampledOwnedTreePrivateBytes': 0,
                       'minimumSampledAvailableRamBytes': psutil.virtual_memory().available,
                       'allPrivateSamplesAvailable': True})
        begin = time.perf_counter()
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
                identity_path = directory / 'actual_identity.json'
                acknowledged = directory / 'actual_identity_sampled.json'
                if identity_path.exists() and not acknowledged.exists():
                    identity = json.loads(identity_path.read_text(encoding='utf-8'))
                    if owned.observed_workload(identity):
                        handshake(acknowledged, {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                private = [row['memoryBytes'].get('private') for row in rows]
                available = psutil.virtual_memory().available
                sample = {'at': stamp(), 'elapsedSeconds': time.perf_counter() - begin, 'processes': rows,
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
                          'Available RAM safeguard' if low_ram >= protocol['lowRamConsecutiveSamples'] else
                          'Wall-time safeguard' if sample['elapsedSeconds'] >= protocol['maximumChildWallSeconds'] else None)
                if reason:
                    report['safeguardStop'] = reason
                    raise RuntimeError(reason + '; no retry')
                time.sleep(protocol['sampleSeconds'])
        report['exitCode'] = process.returncode
        assert process.returncode == 0, 'Diagnostic child failed; retain first failure'
        receipt = json.loads((directory / 'child_receipt.json').read_text(encoding='utf-8'))
        assert receipt['passed'] and receipt['modelsImported'] == []
        assert owned.observed_workload(receipt['actualWorkloadIdentity']), 'Actual interpreter not sampled'
        report['actualWorkloadWasSampled'] = True
        report['remainingOwnedIdentities'] = owned.wait_for_retirement()
        assert not report['remainingOwnedIdentities'], 'Owned exact identity remained after normal exit'
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
                error.add_note('Owned diagnostic cleanup also failed: ' + str(cleanup_error))
                report['cleanupError'] = str(cleanup_error)
    finally:
        report['completedAt'] = stamp()
        report['monitoredWallSeconds'] = time.perf_counter() - begin
        if owned is not None:
            report['ownedIdentities'] = owned.identities()
        try:
            report['sourceBindingsAfter'] = source_bindings()
            report['sourcesUnchanged'] = report.get('sourceBindingsBefore') == report['sourceBindingsAfter']
            report['preparedBindingsUnchanged'] = (report.get('preparedBindingsSha256') ==
                                                    binding(UNIT / 'prepared_source_bindings.json')['sha256'])
            if not report['sourcesUnchanged'] or not report['preparedBindingsUnchanged']:
                report['passed'] = False
                if process is not None:
                    report['status'] = 'FAILED'
        except BaseException as binding_error:
            report['sourceBindingError'] = str(binding_error)
            report['passed'] = False
            if process is not None:
                report['status'] = 'FAILED'
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
    assert sys.prefix == sys.base_prefix, 'Parent must use base Python313'
    return parent()


if __name__ == '__main__':
    raise SystemExit(main())
