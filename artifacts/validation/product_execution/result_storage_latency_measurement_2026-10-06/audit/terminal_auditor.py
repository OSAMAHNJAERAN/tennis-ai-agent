"""Independent one-shot readonly audit, invoked only after benchmark terminal.

Never imports/executes benchmark or production code, launches a child, signals a
process, rewrites a receipt or remeasures a workload. Only this auditor's new
exclusive terminal_audit.json is written.
"""
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5]
AUDIT = Path(__file__).resolve().parent
UNIT = AUDIT.parent
RUN = UNIT / 'run'
BASELINE = ROOT / 'artifacts/validation/product_execution/result_storage_latency_baseline_2026-10-06'
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
SEQUENCE = ((600, 'old'), (600, 'new'), (18000, 'old'), (18000, 'new'))
OUTPUTS = ('analysis/result.json', 'analysis/summary.json', 'read_result.json', 'telemetry.csv', 'report.html')
PRODUCTION_STAGES = ('constructionSeconds', 'publicationSeconds', 'readerOpenSeconds',
                     'presentationSeconds', 'strictReadJsonSeconds', 'csvExportSeconds', 'htmlExportSeconds')
MODULES = ('result_storage', 'results', 'reporting', 'result_document')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    def reject(token):
        raise ValueError('Nonfinite receipt constant: ' + token)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def binding(path):
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': checksum}


def equal(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while True:
            x, y = a.read(65536), b.read(65536)
            if x != y:
                return False
            if not x:
                return True


def identity(value):
    return value['pid'], value['createTimeUnixSeconds']


class Audit:
    def __init__(self):
        self.value = {'auditStartedAt': stamp(), 'passed': False, 'errors': [], 'errorCount': 0,
                      'runs': [], 'comparisons': [], 'qualificationEvidence': False,
                      'inferenceRun': False, 'workloadsLaunchedOrRerun': False,
                      'productionOrBenchmarkCodeImported': False,
                      'scope': 'Read-only terminal audit of separate synthetic frozen-disk/current-disk 600/18000 comparison.',
                      'timingMeaning': 'Record raw stages, child elapsed, after-import work elapsed and monitored wall separately. Production stages include output-file verification hash reads; reader open follows bound generation hash validation/cache. Audit wall is its own file/process inspection. No independently remeasured throughput, variance, accuracy or full-match claim.',
                      'memoryMeaning': 'Actual Windows OS peak working set and peak process commit cover whole children. Exact owned-tree 0.25s sampled sums may miss peaks and double-count shared pages; peak process commit is not resident RAM or disk pagefile occupancy.',
                      'diskMeaning': 'Logical file sizes and recorded boundaries; no peak allocated blocks/write-amplification claim.',
                      'limitations': ['Generated index/60 timestamps/PTS, not decoded footage or independent measurement.',
                                      'Pose is a 65-frame prefix, followed by unavailable frames.',
                                      'Browser whole-result arrays remain outside this server measurement.',
                                      'One frozen/current observation at each count; no performance qualification gate.',
                                      'Synthetic attribution tests escaping; no footage/rights assertion.']}

    def check(self, condition, message):
        if not condition:
            self.value['errorCount'] += 1
            if len(self.value['errors']) < 64:
                self.value['errors'].append(message)
        return condition

    def retirement_now(self, record, psutil):
        value = {'pid': record['pid'], 'createTimeUnixSeconds': record['createTimeUnixSeconds'], 'checkedAt': stamp()}
        try:
            process = psutil.Process(record['pid'])
            created = process.create_time()
            value.update({'currentPidExists': True, 'currentCreateTimeUnixSeconds': created,
                          'originalIdentityStillLive': process.is_running() and created == record['createTimeUnixSeconds'],
                          'pidReused': created != record['createTimeUnixSeconds']})
        except psutil.NoSuchProcess:
            value.update({'currentPidExists': False, 'originalIdentityStillLive': False, 'pidReused': False})
        except psutil.Error as error:
            value['retirementVerificationError'] = str(error)
        self.check(value.get('originalIdentityStillLive') is False, 'Owned identity remains live or unverifiable: ' + str(identity(record)))
        return value

    def executed_order_proof(self, tree):
        parent = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'parent')
        loops = [node for node in ast.walk(parent) if isinstance(node, ast.While) and ast.unparse(node.test) == 'process.poll() is None']
        if len(loops) != 1:
            raise ValueError('Executed monitor loop does not match reviewed ownership pattern')
        body = loops[0].body
        rows = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign) and
                    ast.unparse(node.targets[0]) == 'rows' and ast.unparse(node.value) == 'owned.sample()')
        ack = next(index for index, node in enumerate(body) if isinstance(node, ast.If) and
                   'publish_handshake(ack_path' in ast.unparse(node))
        sample = next(index for index, node in enumerate(body) if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'sample')
        self.check(rows < ack < sample, 'Executed sample/ack/log stamp ordering changed')
        predicate = next(node for node in ast.walk(body[ack]) if isinstance(node, ast.If) and
                         ast.unparse(node.test) == 'owned.observed_workload(identity)')
        self.check(any('publish_handshake(ack_path' in ast.unparse(node) for node in predicate.body), 'Ack does not require exact sampled identity')
        entries = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(body[sample].value.keys, body[sample].value.values, strict=True)}
        self.check(entries.get('at') == 'stamp()' and entries.get('processes') == 'rows', 'Sample stamp is not for previously collected rows')
        child = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'child')
        child_text = ast.unparse(child)
        self.check('deadline = time.monotonic() + 10' in child_text and 'if time.monotonic() >= deadline:' in child_text,
                   'Executed actual-child handshake deadline changed')
        ack_decode = next(node.lineno for node in ast.walk(child) if isinstance(node, ast.Assign) and
                          any(isinstance(target, ast.Name) and target.id == 'ack' for target in node.targets))
        load = next(node.lineno for node in ast.walk(child) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and node.func.id == 'load_disk_implementation')
        self.check(ack_decode < load, 'Selected implementation imports run before actual identity acknowledgment')
        monitor_tree = ast.parse(MONITOR.read_text(encoding='utf-8'))
        owner = next(node for node in monitor_tree.body if isinstance(node, ast.ClassDef) and node.name == 'OwnedProcessTree')
        observed = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == 'observed_workload')
        self.check("return record is not None and record['sampleCount'] > 0" in ast.unparse(observed), 'Monitor observation predicate changed')
        self.value['executedHandshakeOrderProof'] = {
            'sampleRowsLine': body[rows].lineno, 'guardedAckLine': body[ack].lineno,
            'sampleLogAtStampLine': body[sample].lineno, 'childAckDecodeLine': ack_decode,
            'childImplementationLoadLine': load, 'childDeadlineSeconds': 10,
            'meaning': 'Rows are sampled first, then exact identity/sampleCount gates atomic ack, then sample.at is stamped. Do not require logged sample.at <= ack.at.',
            'verifiedMonitor': binding(MONITOR)}

    def inspect_case(self, count, mode, resource, expected, protocol, psutil):
        case = RUN / f'{count}_{mode}'
        child = read(case / 'child_receipt.json')
        actual, ack = read(case / 'actual_identity.json'), read(case / 'actual_identity_sampled.json')
        prefix = f'{count}_{mode}: '
        check = lambda condition, message: self.check(condition, prefix + message)
        check(child['passed'] and resource['passed'] and resource['exitCode'] == 0, 'Case did not pass with terminal exit0')
        check(child['frames'] == count and child['mode'] == mode, 'Recorded child count/mode incorrect')
        check(not any(key in resource for key in ('error', 'safeguardStop', 'cleanupError', 'retirementVerificationError')), 'Resource error/guard was recorded')
        check(child['modelsImported'] == [] and child['inferenceRun'] is False and child['qualificationEvidence'] is False, 'Model/inference/qualification flags incorrect')
        check(child['sourceBindingsBefore'] == child['sourceBindingsAfter'] == expected, 'Child bindings differ from prepared/executed sources')
        check(binding(case / 'child_receipt.json')['sha256'] == resource['childReceiptSha256'], 'Child receipt hash does not match resource binding')
        check(actual == child['actualWorkloadIdentity'] == ack['identity'] and ack['sampledByExactIdentity'] is True, 'Actual/ack identity mismatch')
        check(child['allSelectedStorageClassesConsistent'] is True, 'Selected imported classes not reported consistent')
        implementation = 'FROZEN_PREOPTIMIZATION_DISK' if mode == 'old' else 'CURRENT_DISK_LATENCY_CANDIDATE'
        check(child['implementation'] == implementation, 'Wrong implementation classification')
        loaded = child['loadedProductionModules']
        check(set(loaded) == set(MODULES), 'Wrong selected production module set')
        for name, value in loaded.items():
            path = ((BASELINE / 'src/api' / (name + '.py')) if mode == 'old' else (ROOT / 'src/api' / (name + '.py')))
            observed = binding(path)
            check(value['moduleAlias'] == 'src.api.' + name and value['relativeSource'] == path.relative_to(ROOT).as_posix(), 'Selected module alias/source mismatch: ' + name)
            check({'bytes': value['bytes'], 'sha256': value['sha256']} == observed, 'Selected module binding mismatch: ' + name)
        check(child['inputBindingsBefore'] == child['inputBindingsAfter'], 'Recorded raw inputs changed')
        for name, value in child['inputBindingsBefore'].items():
            check(binding(case / 'analysis' / name) == value, 'Retained original input changed: ' + name)
        remembered = {identity(row): row for row in resource['ownedIdentities']}
        check(identity(actual) in remembered, 'Actual interpreter not remembered')
        counts, peaks = {}, {}
        samples = peak_rss = peak_private = 0
        minimum_ram = None
        all_private = True
        rss_and_wall_guards_clear = True
        low_ram = longest_low = 0
        first_actual = first_after_ack = None
        previous_elapsed = -1.
        with (case / 'samples.jsonl').open('r', encoding='utf-8') as stream:
            for line in stream:
                sample = json.loads(line)
                samples += 1
                check(sample['elapsedSeconds'] >= previous_elapsed, 'Sample monotonic elapsed moved backwards')
                previous_elapsed = sample['elapsedSeconds']
                rows = sample['processes']
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                private_values = [row['memoryBytes'].get('private') for row in rows]
                private = sum(value for value in private_values if value is not None)
                available = all(value is not None for value in private_values)
                check(rss == sample['ownedTreeRssBytes'] and private == sample['ownedTreePrivateBytes'] and
                      available == sample['allPrivateValuesAvailable'], 'Per-sample aggregate memory differs')
                peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
                minimum_ram = sample['availableRamBytes'] if minimum_ram is None else min(minimum_ram, sample['availableRamBytes'])
                all_private &= available
                low_ram = low_ram + 1 if sample['availableRamBytes'] < protocol['minimumAvailableRamBytes'] else 0
                longest_low = max(longest_low, low_ram)
                check(rss <= protocol['maximumOwnedTreeRssBytes'], 'Owned RSS guard threshold exceeded')
                check(sample['elapsedSeconds'] < protocol['maximumChildWallSeconds'], 'Child wall guard threshold exceeded')
                rss_and_wall_guards_clear &= (rss <= protocol['maximumOwnedTreeRssBytes'] and
                                              sample['elapsedSeconds'] < protocol['maximumChildWallSeconds'])
                for row in rows:
                    key = identity(row)
                    check(key in remembered, 'Sample has unknown owned identity')
                    if key not in remembered:
                        continue
                    check(row['parentPid'] == remembered[key]['parentPid'] and row['depth'] == remembered[key]['depth'], 'Recorded ownership ancestry changed')
                    counts[key] = counts.get(key, 0) + 1
                    prior = peaks.setdefault(key, {'rss': 0, 'private': 0, 'osPeakWorkingSet': 0})
                    prior['rss'] = max(prior['rss'], row['memoryBytes']['rss'])
                    prior['private'] = max(prior['private'], row['memoryBytes'].get('private') or 0)
                    prior['osPeakWorkingSet'] = max(prior['osPeakWorkingSet'], row['memoryBytes'].get('peak_wset') or 0)
                    if key == identity(actual):
                        record = {'logRow': samples, 'sampleAt': sample['at'], 'elapsedSeconds': sample['elapsedSeconds'], 'actualProcess': row}
                        if first_actual is None:
                            first_actual = record
                        if first_after_ack is None and sample['at'] >= ack['at']:
                            first_after_ack = record
        check(samples == resource['samples'], 'Per-case sample count differs')
        check(peak_rss == resource['peakSampledOwnedTreeRssBytes'] and peak_private == resource['peakSampledOwnedTreePrivateBytes'], 'Recomputed aggregate peaks differ')
        check(minimum_ram == resource['minimumSampledAvailableRamBytes'] and all_private == resource['allPrivateSamplesAvailable'], 'Recomputed RAM/private flags differ')
        check(longest_low < protocol['lowRamConsecutiveSamples'], 'Low-RAM consecutive guard threshold exceeded')
        check(first_actual is not None and first_after_ack is not None, 'Actual identity not observed in sample/ack iteration logs')
        for key, row in remembered.items():
            check(counts.get(key, 0) == row['sampleCount'], 'Per-identity sample count differs')
            values = peaks.get(key, {'rss': 0, 'private': 0})
            check(values['rss'] == row['peakSampledRssBytes'] and values['private'] == row['peakSampledPrivateBytes'], 'Per-identity peaks differ')
        check(resource['actualWorkloadWasSampled'] and remembered[identity(actual)]['sampleCount'] > 0, 'Actual workload sampling gate failed')
        check(resource['remainingOwnedIdentities'] == [] and resource['allRememberedOwnedIdentitiesRetired'], 'Normal-exit retirement gate failed')
        retired = [self.retirement_now(row, psutil) for row in remembered.values()]
        actual_peak = peaks[identity(actual)]
        check(child['windowsPeakWorkingSetBytes'] >= actual_peak['rss'] and
              child['windowsPeakWorkingSetBytes'] >= actual_peak['osPeakWorkingSet'], 'Child OS peak below observed actual process peak')
        for name, path in (('publishedJson', case / 'analysis/result.json'), ('readJson', case / 'read_result.json'),
                           ('csv', case / 'telemetry.csv'), ('html', case / 'report.html')):
            observed = binding(path)
            check({'bytes': child[name]['bytes'], 'sha256': child[name]['sha256']} == observed, 'Retained output receipt mismatch: ' + name)
        check(equal(case / 'analysis/result.json', case / 'read_result.json'), 'Published and read JSON bytes differ')
        marker = read(case / 'analysis/.result/current.store')
        check(marker == child['publicationManifest'], 'Publication manifest differs')
        database = case / 'analysis/.result' / marker['database']
        check(binding(database) == {'bytes': marker['databaseBytes'], 'sha256': marker['databaseSha256']}, 'Published database binding differs')
        check(marker['frames'] == count and marker['databaseBytes'] == child['publishedDatabaseBytes'], 'Published database count/size differs')
        check(child['ownedScratchAndPendingExportsAbsent'] and not (case / 'analysis/result.tmp').exists() and
              not (case / 'analysis/summary.tmp').exists() and not list((case / 'analysis').glob('.result_storage_*.sqlite3*')),
              'Scratch or pending exports remain')
        summary = read(case / 'analysis/summary.json')
        check(summary['frames'] == count and summary['fps'] == 60 and summary['durationSeconds'] == count / 60,
              'Summary synthetic frame/time values differ')
        csv_rows = 0
        with (case / 'telemetry.csv').open('r', encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            check(len(reader.fieldnames) == 25, 'CSV column count changed')
            for index, row in enumerate(reader):
                check(int(row['frame']) == index and float(row['normalized_seconds']) == index / 60, 'CSV dense synthetic index/clock differs')
                check(row['analysis_id'] == summary['id'], 'CSV analysis identity differs')
                csv_rows += 1
        check(csv_rows == count, 'CSV complete data count differs')
        stages = child['stages']
        check(abs(sum(stages[name] for name in PRODUCTION_STAGES) - child['measuredProductionStagesSeconds']) < 1e-8,
              'Declared production stage sum differs')
        check(resource['monitoredChildWallSeconds'] >= child['totalChildElapsedSeconds'] >= child['workAfterImportsSeconds'] >=
              child['measuredProductionStagesSeconds'], 'Monitor/child/work/stage timing scopes are inconsistent')
        for name in ('readJson', 'csv', 'html'):
            export = child[name]
            check(export['firstChunkAvailableSeconds'] <= export['allChunksWrittenAndClosedSeconds'] and
                  export['verificationHashFullFileReadSeconds'] >= 0, 'Export first/all/hash time scopes inconsistent')
        self.value['runs'].append({
            'frames': count, 'mode': mode, 'implementation': implementation, 'loadedProductionModules': loaded,
            'actualWorkloadIdentity': actual, 'acknowledgment': ack,
            'firstActualLoggedSample': first_actual, 'firstActualLoggedSampleAtOrAfterAck': first_after_ack,
            'sampleBeforeAckProof': 'Bound executed code samples rows before checking exact observed identity and publishing ack; log sample.at follows ack. Recorded actual identity/sample counts match and child verifies ack before imports.',
            'allOwnedIdentitiesRetiredNow': all(row.get('originalIdentityStillLive') is False for row in retired),
            'ownershipNow': retired, 'samples': samples, 'guardThresholdsNeverTriggered': rss_and_wall_guards_clear and longest_low < protocol['lowRamConsecutiveSamples'],
            'recomputedOwnedTreeRssPeakBytes': peak_rss, 'recomputedOwnedTreePrivatePeakBytes': peak_private,
            'actualSampledRssPeakBytes': actual_peak['rss'], 'actualOsPeakWorkingSetBytes': child['windowsPeakWorkingSetBytes'],
            'actualOsPeakPrivateCommitBytes': child['windowsPeakPrivateCommitBytes'],
            'minimumSampledAvailableRamBytes': minimum_ram, 'allPrivateSamplesAvailable': all_private,
            'rawStagesSeconds': stages, 'productionStageTotalSeconds': child['measuredProductionStagesSeconds'],
            'childElapsedSeconds': child['totalChildElapsedSeconds'], 'afterImportWorkElapsedSeconds': child['workAfterImportsSeconds'],
            'monitorWallSeconds': resource['monitoredChildWallSeconds'], 'publishedDatabaseBytes': child['publishedDatabaseBytes'],
            'diskBoundarySnapshots': child['diskSnapshots'], 'retainedResourceBoundaryDisk': resource['retainedDisk'],
            'csvRowsAndSyntheticTimestampsIndependentlyChecked': csv_rows,
            'recordedIndependentContractChecks': child['semanticChecks'], 'modelsImported': child['modelsImported']})
        return child

    def run(self, terminal, psutil):
        protocol = read(UNIT / 'protocol.json')
        self.check(protocol['counts'] == [600, 18000] and protocol['modes'] == ['old', 'new'] and
                   protocol['caseOrder'] == [f'{count}_{mode}' for count, mode in SEQUENCE], 'Fixed counts/mode order changed')
        self.check(protocol['maximumChildWallSeconds'] == 180 and protocol['minimumAvailableRamBytes'] == 419430400 and
                   protocol['lowRamConsecutiveSamples'] == 5 and protocol['sampleSeconds'] == .25, 'Reviewed guards changed')
        self.check(terminal['status'] == 'COMPLETED' and terminal['passed'], 'Benchmark terminal report did not pass COMPLETED')
        self.check(len(terminal['runs']) == 4 and len(terminal['comparisons']) == 2, 'Incomplete case/comparison set')
        self.check(terminal['qualificationEvidence'] is False and terminal['inferenceRun'] is False, 'Terminal qualification flags changed')
        prepared, executed = read(UNIT / 'prepared_source_bindings.json'), read(RUN / 'source_bindings.json')
        self.check(prepared == executed == terminal['sourceBindingsBefore'] == terminal['sourceBindingsAfter'], 'Prepared/executed/before/after source bindings differ')
        self.check(terminal['sourcesUnchanged'] and terminal['preparedBindingsUnchanged'], 'Recorded source freeze gates failed')
        self.check(binding(UNIT / 'prepared_source_bindings.json')['sha256'] == terminal['preparedBindingsSha256'], 'Prepared binding record hash differs')
        self.check(equal(UNIT / 'helper.py', RUN / 'executed_helper.py'), 'Prepared/executed helper bytes differ')
        now = {name: binding(ROOT / name) for name in executed}
        self.check(now == executed, 'Current source/baseline/helper/protocol/historical evidence binding differs')
        self.value['sourceBindingsNow'] = now
        self.value['preparedBindings'] = binding(UNIT / 'prepared_source_bindings.json')
        monitor_interpreter = terminal['monitorInterpreter']
        self.check(binding(Path(monitor_interpreter['path']))['sha256'] == monitor_interpreter['sha256'], 'Monitoring interpreter changed')
        self.value['monitorInterpreter'] = monitor_interpreter
        tree = ast.parse((RUN / 'executed_helper.py').read_text(encoding='utf-8'))
        self.executed_order_proof(tree)
        constants = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and
                    node.targets[0].id in ('FROZEN', 'FROZEN_RECEIPT')):
                constants[node.targets[0].id] = ast.literal_eval(node.value)
        baseline_receipt = read(BASELINE / 'receipt.json')
        self.check(binding(BASELINE / 'receipt.json') == constants['FROZEN_RECEIPT'], 'Historical baseline preservation receipt changed')
        self.check(baseline_receipt['bindings'] == constants['FROZEN'] and len(constants['FROZEN']) == 17,
                   'Historical 17-file baseline record differs')
        self.value['historicalBaseline17Bindings'] = {name: binding(BASELINE / name) for name in constants['FROZEN']}
        self.check(self.value['historicalBaseline17Bindings'] == constants['FROZEN'], 'Historical preserved baseline bytes changed')
        if len(terminal['runs']) != 4:
            return
        children = {}
        for ordinal, (count, mode) in enumerate(SEQUENCE):
            resource = read(RUN / f'{count}_{mode}/resource_receipt.json')
            self.check(resource == terminal['runs'][ordinal], 'Per-case resource/terminal records differ: ' + str((count, mode)))
            children[count, mode] = self.inspect_case(count, mode, resource, executed, protocol, psutil)
        for ordinal, count in enumerate((600, 18000)):
            comparison = read(RUN / f'comparison_{count}.json')
            self.check(comparison == terminal['comparisons'][ordinal], 'Comparison/terminal record differs')
            self.check(comparison['passed'] and comparison['volatileFieldsExcluded'] == [], 'Comparison had failure or exclusions')
            self.check(comparison['baselineImplementation'] == 'FROZEN_PREOPTIMIZATION_DISK' and
                       comparison['currentImplementation'] == 'CURRENT_DISK_LATENCY_CANDIDATE', 'Comparison baseline is not disk/current-disk')
            old, new = children[count, 'old'], children[count, 'new']
            self.check(old['inputBindingsBefore'] == new['inputBindingsBefore'] and old['semanticChecks'] == new['semanticChecks'],
                       'Pair streaming inputs or recorded semantic checks differ')
            outputs = {}
            for name in OUTPUTS:
                a, b = RUN / f'{count}_old' / name, RUN / f'{count}_new' / name
                left, right, identical = binding(a), binding(b), equal(a, b)
                declared = comparison['outputs'][name]
                self.check(identical and declared['exactBytesEqual'] and
                           left == {'bytes': declared['oldBytes'], 'sha256': declared['oldSha256']} and
                           right == {'bytes': declared['newBytes'], 'sha256': declared['newSha256']},
                           'Exact output byte/hash comparison differs: ' + str(count) + '/' + name)
                outputs[name] = {'exactBytesEqualNow': identical, 'bytes': left['bytes'], 'sha256': left['sha256']}
            self.value['comparisons'].append({'frames': count, 'outputs': outputs, 'fiveExactByteComparisonsPassed': all(row['exactBytesEqualNow'] for row in outputs.values()),
                                             'sameStreamingInputsAndRecordedSemantics': old['inputBindingsBefore'] == new['inputBindingsBefore'] and old['semanticChecks'] == new['semanticChecks'],
                                             'volatileFieldsExcluded': [], 'baselineRawStagesSeconds': old['stages'], 'currentRawStagesSeconds': new['stages'],
                                             'baselineProductionStageTotalSeconds': old['measuredProductionStagesSeconds'],
                                             'currentProductionStageTotalSeconds': new['measuredProductionStagesSeconds'],
                                             'baselineChildElapsedSeconds': old['totalChildElapsedSeconds'], 'currentChildElapsedSeconds': new['totalChildElapsedSeconds'],
                                             'baselineActualOsPeakWorkingSetBytes': old['windowsPeakWorkingSetBytes'],
                                             'currentActualOsPeakWorkingSetBytes': new['windowsPeakWorkingSetBytes'],
                                             'baselineDatabaseBytes': old['publishedDatabaseBytes'], 'currentDatabaseBytes': new['publishedDatabaseBytes'],
                                             'noPerformanceQualificationGate': True})
        # Include every retained benchmark output/receipt/protocol/auditor input.
        # This remains separate from the source-frozen benchmark bindings.
        destination = AUDIT / 'terminal_audit.json'
        paths = sorted(path for path in UNIT.rglob('*') if path.is_file() and path != destination)
        files = {path.relative_to(UNIT).as_posix(): binding(path) for path in paths}
        self.value['retainedFileBindings'] = files
        self.value['retainedFileCount'] = len(files)
        self.value['retainedLogicalFileBytesExcludingAudit'] = sum(row['bytes'] for row in files.values())
        self.check(all(binding(UNIT / name) == expected for name, expected in files.items()), 'Retained receipt/output changed during audit')
        self.check({name: binding(ROOT / name) for name in executed} == now, 'Source/baseline changed during audit')
        self.value['allRetainedFilesStableDuringAudit'] = self.value['errorCount'] == 0


def main():
    destination = AUDIT / 'terminal_audit.json'
    if destination.exists():
        raise FileExistsError('Preserve one-shot terminal audit: ' + str(destination))
    # Refuse early calls without consuming the one-shot terminal receipt.
    terminal_path = RUN / 'report.json'
    if not terminal_path.exists():
        raise RuntimeError('Benchmark terminal report is not available; no audit receipt written')
    terminal = read(terminal_path)
    if terminal.get('status') not in ('COMPLETED', 'FAILED', 'NOT_STARTED'):
        raise RuntimeError('Benchmark is not terminal; no audit receipt written')
    import psutil
    audit = Audit()
    begin = time.perf_counter()
    try:
        audit.run(terminal, psutil)
        audit.value['passed'] = audit.value['errorCount'] == 0
    except BaseException as error:
        audit.check(False, type(error).__name__ + ': ' + str(error))
        audit.value['traceback'] = traceback.format_exc()
        audit.value['passed'] = False
    finally:
        audit.value['completedAt'] = stamp()
        audit.value['auditWallSeconds'] = time.perf_counter() - begin
        with destination.open('x', encoding='utf-8', newline='') as stream:
            json.dump(audit.value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'passed': audit.value['passed'], 'errors': audit.value['errors'],
                          'retainedFileCount': audit.value.get('retainedFileCount'),
                          'completedCasesAudited': len(audit.value['runs']),
                          'exactComparisonsAudited': len(audit.value['comparisons']) * 5,
                          'terminalAuditSha256': binding(destination)['sha256']}), flush=True)
    return 0 if audit.value['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
