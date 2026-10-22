"""Read-only bounded terminal audit; writes only its exclusive audit receipt."""
from __future__ import annotations
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time
import traceback
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
UNIT = HERE.parent
RUN = HERE / 'run'
RECEIPT = HERE / 'terminal_audit.json'
LIMIT = 100_000_000
MAX_VALUE_BUFFER = 262144
EXPECTED_HELPER = '056855431ba0c5abe433c15e556f96e6540d69f5626e90ce350626ce34692d4f'
FPS = 60000 / 1001
JOINTS = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder',
          'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')
RACKET_KEYS = ('state', 'position_px', 'bbox_xyxy', 'confidence', 'velocity_px_per_s',
               'motion_direction_deg', 'contact_event')
OUTPUTS = ('observations.jsonl', 'render_audit.json', 'player_motion.json', 'racket_tracking.json')
TRACKED = {}
PROCESS = None
MAX_OBSERVED_RSS = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def memory_guard():
    global MAX_OBSERVED_RSS
    rss = PROCESS.memory_info().rss
    MAX_OBSERVED_RSS = max(MAX_OBSERVED_RSS, rss)
    require(rss <= LIMIT, 'Audit own RSS exceeded 100 MB limit')


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': sha}


def track(path, expected=None):
    pathname = name(path)
    if pathname not in TRACKED:
        TRACKED[pathname] = binding(path)
    actual = TRACKED[pathname]
    if isinstance(expected, str):
        require(actual['sha256'] == expected, 'SHA mismatch: ' + pathname)
    elif expected is not None:
        require(actual == expected, 'Binding mismatch: ' + pathname)
    return actual


def load(path, expected=None):
    track(path, expected)
    return json.loads(Path(path).read_bytes())


def maps(values):
    for pathname, value in values.items():
        track(resolve(pathname), value)


def signature(value):
    # Independent native comparison: every scalar type and IEEE float bit,
    # every mapping's key type/insertion order and sequence type are retained.
    if value is None:
        return ('null',)
    if type(value) is bool:
        return ('bool', value)
    if type(value) is int:
        return ('int', value)
    if type(value) is float:
        return ('float', struct.pack('!d', value))
    if type(value) is str:
        return ('str', value)
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(signature(item) for item in value))
    if type(value) is dict:
        return ('dict', tuple((signature(key), signature(item)) for key, item in value.items()))
    raise TypeError('Unexpected native observation type: ' + type(value).__name__)


def decode_native(node):
    require(type(node) is list and node and type(node[0]) is str, 'Malformed independent native output node')
    kind = node[0]
    if kind == 'null':
        require(len(node) == 1, 'Malformed null')
        return None
    if kind in ('bool', 'int', 'str'):
        require(len(node) == 2 and type(node[1]) is {'bool': bool, 'int': int, 'str': str}[kind], 'Scalar type changed')
        return node[1]
    if kind == 'float64':
        require(len(node) == 2 and type(node[1]) is str and len(node[1]) == 16, 'Invalid IEEE float bytes')
        return struct.unpack('!d', bytes.fromhex(node[1]))[0]
    if kind in ('list', 'tuple'):
        require(len(node) == 2 and type(node[1]) is list, 'Sequence type changed')
        result = [decode_native(item) for item in node[1]]
        return result if kind == 'list' else tuple(result)
    if kind == 'dict':
        require(len(node) == 2 and type(node[1]) is list, 'Mapping type changed')
        result = {}
        for pair in node[1]:
            require(type(pair) is list and len(pair) == 2, 'Invalid native mapping entry')
            key = decode_native(pair[0])
            require(key not in result, 'Duplicate native mapping key')
            result[key] = decode_native(pair[1])
        return result
    raise ValueError('Unexpected native node kind: ' + kind)


def independently_expected_frame(index):
    order = [2, 1] if index % 2 else [1, 2]
    poses, rackets = [], []
    for identity in order:
        joints = {}
        for offset, joint_name in enumerate(JOINTS):
            x = -0.0 if offset == 0 else math.nextafter(float(80 + identity * 140 + offset), math.inf)
            y = math.nextafter(float(150 + offset), -math.inf) + index / FPS
            joints[joint_name] = {'position_px': [x, y], 'confidence': .9000000000000001}
        poses.append([identity, joints])
        if (index + identity) % 5 == 1:
            racket = {'state': 'MISSING', 'position_px': None, 'bbox_xyxy': None, 'confidence': None,
                      'velocity_px_per_s': None, 'motion_direction_deg': None, 'contact_event': None}
        else:
            left = math.nextafter(float(180 + identity * 50), math.inf) + index / FPS
            racket = {'state': 'DETECTED', 'position_px': [left + 12.5, 172.5],
                      'bbox_xyxy': [left, 160.0, left + 25.0, 185.0], 'confidence': .8750000000000001,
                      'velocity_px_per_s': [-0.0, index / FPS] if index else None,
                      'motion_direction_deg': 90.0 if index else None, 'contact_event': None}
        rackets.append([identity, racket])
    return {'frame_index': index, 'timestamp_seconds': index / FPS, 'identity_order': order,
            'pose': poses, 'racket': rackets}


def jsonl_rows(path):
    with Path(path).open('rb') as stream:
        while True:
            line = stream.readline(MAX_VALUE_BUFFER + 1)
            if not line:
                return
            require(len(line) <= MAX_VALUE_BUFFER and line.endswith(b'\n'), 'Unbounded/truncated JSONL row')
            yield line


def audit_native(count, fixture_receipt):
    fixture = RUN / f'fixture_{count}.jsonl'
    paths = [RUN / f'{count}_{mode}/observations.jsonl' for mode in ('old', 'new')]
    rows, joints, racket_states, signed_zeros = 0, 0, {'DETECTED': 0, 'MISSING': 0}, 0
    digests = [hashlib.sha256(), hashlib.sha256()]
    iterators = [iter(jsonl_rows(path)) for path in paths]
    for index, line in enumerate(jsonl_rows(fixture)):
        require(index < count, 'Extra complete fixture row')
        row = json.loads(line)
        require(signature(row) == signature(independently_expected_frame(index)), 'Complete fixture differs from independent expected native record')
        pose, racket = dict(row['pose']), dict(row['racket'])
        expected = {'frame_index': index, 'timestamp_seconds': index / FPS, 'pose': pose, 'racket': racket}
        for side, iterator in enumerate(iterators):
            encoded = next(iterator, None)
            require(encoded is not None, 'Incomplete native observation output')
            digests[side].update(encoded)
            native = decode_native(json.loads(encoded))
            require(signature(native) == signature(expected), 'Native bits/types/identity or joint order changed')
        for identity in row['identity_order']:
            require(type(identity) is int and list(pose[identity]) == list(JOINTS), 'Missing integer identity or joint')
            require(struct.pack('!d', pose[identity]['nose']['position_px'][0]) == struct.pack('!d', -0.0), 'Signed zero lost')
            signed_zeros += 1
            joints += len(pose[identity])
            require(tuple(racket[identity]) == RACKET_KEYS, 'Racket fields omitted/reordered')
            racket_states[racket[identity]['state']] += 1
        rows += 1
        if rows % 1000 == 0:
            memory_guard()
    require(rows == count and all(next(iterator, None) is None for iterator in iterators), 'Extra/missing complete native rows')
    require(all(digest.hexdigest() == fixture_receipt['expectedNativeObservationSha256'] for digest in digests), 'Native full-row digest differs')
    return {'framesPerOldAndNewOutput': rows, 'poseIdentitiesPerOutput': rows * 2,
            'poseJointsPerOutput': joints, 'signedZeroPoseCoordinatesPerOutput': signed_zeros,
            'racketStatesPerOutput': racket_states, 'exactNativeAllRowsTypesOrderAndIeeeBits': True}


class StreamingJson:
    """Incremental stdlib JSON decoder with one bounded metadata/frame value."""
    def __init__(self, path):
        self.stream = Path(path).open('r', encoding='utf-8', newline=None)
        self.buffer, self.position, self.eof = '', 0, False
        self.decoder = json.JSONDecoder()

    def close(self):
        self.stream.close()

    def refill(self):
        self.buffer = self.buffer[self.position:]
        self.position = 0
        require(len(self.buffer) <= MAX_VALUE_BUFFER, 'Export JSON value exceeded bounded parser buffer')
        addition = self.stream.read(32768)
        self.buffer += addition
        self.eof = not addition
        return bool(addition)

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof or not self.refill():
                return ''

    def consume(self, token):
        require(self.peek() == token, 'Unexpected export JSON token: ' + repr(token))
        self.position += 1

    def value(self):
        require(bool(self.peek()), 'Truncated complete export JSON')
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                # A number can be split at a chunk boundary. A complete value
                # must be followed by a JSON delimiter/whitespace or EOF.
                if end == len(self.buffer) and not self.eof:
                    self.refill()
                    continue
                if end < len(self.buffer):
                    if not (self.buffer[end].isspace() or self.buffer[end] in ',]}:'):
                        require(not self.eof and self.refill(), 'Incomplete primitive export value')
                        continue
                self.position = end
                return value
            except json.JSONDecodeError:
                require(not self.eof and self.refill(), 'Invalid/truncated export JSON value')


def audit_export(path, count, fixture, metadata, key, kind):
    reader = StreamingJson(path)
    keys, rows = [], 0
    try:
        reader.consume('{')
        while reader.peek() != '}':
            field = reader.value()
            require(type(field) is str and field not in keys, 'Duplicate export field')
            keys.append(field)
            reader.consume(':')
            if field != key:
                require(field in metadata and signature(reader.value()) == signature(metadata[field]), 'Complete export metadata changed')
            else:
                reader.consume('[')
                for index, line in enumerate(jsonl_rows(fixture)):
                    require(index < count and reader.peek() != ']', 'Incomplete export frame array')
                    row = json.loads(line)
                    expected = {str(identity): value for identity, value in row[kind]}
                    require(signature(reader.value()) == signature(expected), 'Full exported native values/order/precision changed')
                    rows += 1
                    if rows < count:
                        reader.consume(',')
                    if rows % 1000 == 0:
                        memory_guard()
                require(rows == count, 'Export frame count differs')
                reader.consume(']')
            if reader.peek() != '}':
                reader.consume(',')
        reader.consume('}')
        require(reader.peek() == '' and keys == list(metadata) + [key], 'Trailing output bytes or metadata/frame order changed')
    finally:
        reader.close()
    with Path(path).open('rb') as stream:
        stream.seek(-1, 2)
        require(stream.read() == b'}', 'Native exporter added a final newline')
    return {'completeArrayRowsParsed': rows, 'metadataAndAllFrameFieldsExact': True,
            'integerIdentitiesExportedAsOrderedJsonKeys': True, 'nativeFloatBitsAndSignedZeroPreserved': True,
            'noFieldExclusions': True, 'noFinalNewline': True,
            'bothOldAndNewCoveredByCompleteByteEquality': True}


def exact_files(first, second):
    with Path(first).open('rb') as a, Path(second).open('rb') as b:
        while True:
            left, right = a.read(65536), b.read(65536)
            require(left == right, 'Full file comparison differs: ' + name(first))
            if not left:
                return True


def retired(identity, psutil):
    try:
        process = psutil.Process(identity['pid'])
        require(process.create_time() != identity['createTimeUnixSeconds'], 'Exact benchmark owned identity remains live')
    except psutil.NoSuchProcess:
        pass


def audit_case(mode, count, report_resource, psutil):
    folder = RUN / f'{count}_{mode}'
    child = load(folder / 'child_receipt.json', report_resource['childReceiptBinding'])
    resource = load(folder / 'resource_receipt.json')
    require(resource == report_resource and resource['passed'] is True and resource['status'] == 'COMPLETED'
            and child['passed'] is True and child['frames'] == resource['frames'] == count
            and child['mode'] == resource['mode'] == mode, 'Case terminal receipt classification differs')
    require(resource['childStarted'] is True and resource['actualWorkloadWasSampled'] is True
            and resource['allOwnedIdentitiesRetiredAtTerminal'] is True and resource['remainingOwnedIdentitiesAtTerminal'] == []
            and child['parentActualIdentitySampledBeforeHeavyImports'] is True
            and child['allActualImportedSourcesUnchanged'] is True and child['launchSourceBindingsUnchanged'] is True,
            'Actual workload sampling/source/retirement assertions missing')
    require(not any(field in resource for field in ('safeguardStop', 'cleanupError', 'retirementCheckError')), 'Guard/cleanup failure exists')
    identity = load(folder / 'actual_identity.json')
    ack = load(folder / 'actual_identity_sampled.json')
    require(set(identity) == {'pid', 'createTimeUnixSeconds'} and identity == child['actualWorkloadIdentity'] == resource['actualWorkloadIdentity']
            and ack['identity'] == identity and ack['sampledByExactIdentity'] is True, 'Exact canonical actual identity acknowledgement changed')
    imported = load(folder / 'imported_sources.json', resource['retainedImportedSourceRecordBinding'])
    require(imported['bindings'] == child['actualImportedSourceBindingsBefore'] and imported['modules'] == child['actualImportedModules']
            and imported['actualInterpreter'] == child['actualInterpreterBeforeImports'], 'Actual precomputation import record differs')
    maps(imported['bindings'])
    maps(child['actualImportedSourceBindingsAfter'])
    require(all(child['actualImportedSourceBindingsAfter'].get(key) == value for key, value in imported['bindings'].items()), 'Imported source continuity changed')
    prohibited = ('torch', 'ultralytics', 'tensorflow', 'transformers', 'onnxruntime', 'cv2', 'src.pipeline', 'src.models', 'src.detection')
    require(not any(module == prefix or module.startswith(prefix + '.') for module in child['actualImportedModulesAfter'] for prefix in prohibited), 'Model/media/pipeline module imported')
    require('numpy' in imported['modules'] and 'src.utils.identity_frame_sequence' in imported['modules'], 'Actual ordinary implementation import omitted')
    actual_exe = child['actualInterpreterBeforeImports']
    track(actual_exe['osProcessExe'], actual_exe['osProcessExeBinding'])
    require(resolve(actual_exe['osProcessExe']) == resolve(r'C:\Users\ac-98\AppData\Local\Programs\Python\Python313\python.exe'), 'Actual interpreter differed')
    count_samples, peak_rss, peak_private, minimum_ram = 0, 0, 0, None
    identities = {}
    matched_sample_times, low = [], 0
    for line in jsonl_rows(folder / 'resource_samples.jsonl'):
        sample = json.loads(line)
        count_samples += 1
        rows = sample['processes']
        rss = sum(row['memoryBytes']['rss'] for row in rows)
        private = sum(row['memoryBytes'].get('private', 0) for row in rows)
        require(rss == sample['ownedTreeRssBytes'] and private == sample['ownedTreePrivateBytes'], 'Raw aggregate memory sample differs')
        require(sample['allPrivateValuesAvailable'] is True, 'Private sample unavailable')
        peak_rss, peak_private = max(peak_rss, rss), max(peak_private, private)
        available = sample['availableRamBytes']
        minimum_ram = available if minimum_ram is None else min(minimum_ram, available)
        low = low + 1 if available < 400_000_000 else 0
        require(rss <= 1_000_000_000 and low < 5 and sample['elapsedSeconds'] <= 180, 'Passed case raw samples breach declared guard')
        for row in rows:
            key = (row['pid'], row['createTimeUnixSeconds'])
            summary = identities.setdefault(key, {'samples': 0, 'rss': 0, 'private': 0})
            summary['samples'] += 1
            summary['rss'] = max(summary['rss'], row['memoryBytes']['rss'])
            summary['private'] = max(summary['private'], row['memoryBytes'].get('private', 0))
            if key == (identity['pid'], identity['createTimeUnixSeconds']):
                matched_sample_times.append(datetime.fromisoformat(sample['at']))
    require(count_samples == resource['samples'] and peak_rss == resource['peakSampledOwnedTreeRssBytes']
            and peak_private == resource['peakSampledOwnedTreePrivateBytes'] and minimum_ram == resource['minimumSampledAvailableRamBytes'],
            'Raw sample aggregate/count/minimum does not match terminal receipt')
    require(matched_sample_times and datetime.fromisoformat(ack['at']) <= matched_sample_times[0], 'Acknowledgement does not coincide with first sampled actual identity')
    owned = resource['ownedProcessIdentitiesAndPeaks']
    require(set(identities) == {(item['pid'], item['createTimeUnixSeconds']) for item in owned}, 'Sampled owned identity inventory differs')
    for item in owned:
        key = (item['pid'], item['createTimeUnixSeconds'])
        require(identities[key] == {'samples': item['sampleCount'], 'rss': item['peakSampledRssBytes'], 'private': item['peakSampledPrivateBytes']}, 'Per-identity memory/sample summary differs')
        retired(item, psutil)
    root_identity = resource['redirectorIdentity']
    require((root_identity['pid'], root_identity['createTimeUnixSeconds']) in identities
            and (identity['pid'], identity['createTimeUnixSeconds']) in identities, 'Redirector or actual interpreter never sampled')
    actual_memory = child['actualProcessMemoryBytes']
    require(child['windowsPeakWorkingSetBytes'] == resource['actualOsPeakWorkingSetBytes'] == actual_memory['peak_wset']
            and child['windowsPeakProcessCommitBytes'] == resource['actualOsPeakProcessCommitBytes'] == actual_memory['peak_pagefile'],
            'OS actual process peaks mixed with sampled tree counters')
    require(child['constructedPoseFrames'] == child['constructedRacketFrames'] == child['completeNativeFidelityRows'] == count,
            'Complete construction/native verification counts differ')
    require(child['ownedScratchCleanupPassed'] is True and resource['partialScratchFilesAtParentTerminal'] == {}
            and child['partialScratchFilesAtChildTerminal'] == {} and not any((folder / 'scratch').iterdir()), 'Owned database scratch not cleaned')
    disk = child['storageDisk']
    require(sum(disk['filesBeforeCleanup'].values()) == disk['databaseAndSidecarLogicalBytesBeforeCleanup'], 'Logical database/sidecar sum differs')
    require(sum(value for key, value in disk['filesBeforeCleanup'].items() if key.endswith('.sqlite3')) == disk['databaseMainLogicalBytesBeforeCleanup'], 'Main database sum differs')
    for pathname, length in disk['filesBeforeCleanup'].items():
        require(resolve(pathname).parent == (folder / 'scratch').resolve() and type(length) is int and length >= 0, 'Database receipt refers outside exact owned scratch')
    if mode == 'old':
        require(disk['filesBeforeCleanup'] == {} and child['observationContainerRetainsAllFrames'] is True, 'Old resident list classification differs')
    else:
        require(len([key for key in disk['filesBeforeCleanup'] if key.endswith('.sqlite3')]) == 2
                and child['observationContainerRetainsAllFrames'] is False and child['newWorkloadMaterializedFullFrameArray'] is False, 'New owned disk sequence classification differs')
    for output, value in child['outputBindings'].items():
        track(folder / output, value)
    memory_guard()
    return {'mode': mode, 'frames': count, 'passed': True, 'rawSamplesAudited': count_samples,
            'ownedIdentitiesRetiredNow': len(owned), 'actualInterpreterSampledAndAcknowledged': True,
            'constructionSeconds': child['completeObservationConstructionAndAppendSeconds'],
            'indexedRenderLikeReadSeconds': child['indexedRenderLikeReadsSeconds'],
            'playerMotionSerializationSeconds': child['playerMotionSerializationSeconds'],
            'racketTrackingSerializationSeconds': child['racketTrackingSerializationSeconds'],
            'nativeFidelitySerializationSeconds': child['completeNativeFidelitySerializationSeconds'],
            'childHelperElapsedSeconds': child['totalChildElapsedSeconds'],
            'processLaunchToObservedExitSeconds': resource['processLaunchToObservedExitSeconds'],
            'monitoredWallSeconds': resource['monitoredChildWallSeconds'],
            'actualOsPeakWorkingSetBytes': child['windowsPeakWorkingSetBytes'],
            'actualOsPeakProcessCommitBytes': child['windowsPeakProcessCommitBytes'],
            'sampledOwnedTreePeakRssBytes': peak_rss, 'sampledOwnedTreePeakPrivateBytes': peak_private,
            'reportedDatabaseAndSidecarLogicalBytesBeforeCleanup': disk['databaseAndSidecarLogicalBytesBeforeCleanup'],
            'reportedMainDatabaseLogicalBytesBeforeCleanup': disk['databaseMainLogicalBytesBeforeCleanup'],
            'scratchAbsentNow': True}


def audit_backend(report):
    folder = UNIT / 'backend_full'
    receipt = load(folder / 'receipt.json', report['freshBackendPrerequisite']['receiptBinding'])
    before = load(folder / 'source_bindings_before.json')
    after = load(folder / 'source_bindings_after.json')
    require(before == after and receipt['passed'] is True and receipt['sourcesUnchanged'] is True
            and receipt['allOwnedIdentitiesRetired'] is True and receipt['status'] == 'PASSED', 'Fresh backend binding or terminal gate changed')
    maps(before['files'])
    inventory = {name(path) for folder_name in ('src', 'tests', 'configs') for path in (ROOT / folder_name).rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyo')}
    inventory.update(name(ROOT / leaf) for leaf in ('pytest.ini', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.coveragerc',
                     'requirements.txt', 'requirements-product.txt') if (ROOT / leaf).is_file())
    inventory.update((name(ROOT / 'scripts/audit/measure_candidate_storage.py'), name(folder / 'verify_backend_full.py')))
    require(inventory == set(before['files']), 'Full current backend input inventory changed')
    for item in (before['baseInterpreter'], before['actualHelperInterpreter']):
        track(item['path'], {key: item[key] for key in ('bytes', 'sha256')})
    for pathname, value in receipt['artifacts'].items():
        track(folder / pathname, value)
    suites = list(ET.parse(folder / 'junit.xml').getroot().iter('testsuite'))
    counts = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    require(all(receipt['counts'][key] == value for key, value in counts.items())
            and counts['tests'] == receipt['counts']['passed'] and all(counts[key] == 0 for key in ('failures', 'errors', 'skipped')),
            'Actual all-passed backend JUnit counts changed')
    return {'actualCounts': receipt['counts'], 'boundInputInventoryFiles': len(inventory),
            'sourceAndInterpreterBindingsUnchanged': True}


def audit_code_order():
    track(HERE / 'helper.py', EXPECTED_HELPER)
    tree = ast.parse((HERE / 'helper.py').read_bytes())
    child = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'child')
    ack_line = next(node.lineno for node in ast.walk(child) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'acknowledgement' for target in node.targets))
    imports = [node.lineno for node in ast.walk(child) if isinstance(node, ast.Import) and any(alias.name == 'numpy' for alias in node.names)]
    project_import = [node.lineno for node in ast.walk(child) if isinstance(node, ast.ImportFrom) and node.module == 'src.utils.identity_frame_sequence']
    require(len(imports) == len(project_import) == 1 and ack_line < imports[0] < project_import[0], 'Frozen child source no longer gates numeric imports after acknowledgement')
    require(b'owned.sample -> observed_workload exact PID/createTime -> atomic ack' in (HERE / 'helper.py').read_bytes(), 'Reviewed parent sample-before-ack order changed')
    return {'frozenHelperHashUnchanged': True, 'acknowledgementValidatedBeforeNumericImportsInFrozenCode': True,
            'observedParentSamplesAndIdentityReceiptAgree': True,
            'limitation': 'The first raw sample containing the actual identity is timestamped after atomic acknowledgement in that iteration; frozen parent code calls owned.sample and observed_workload before publishing the acknowledgement. No separate import telemetry is fabricated.'}


def main():
    global PROCESS
    import psutil
    PROCESS = psutil.Process()
    require(not RECEIPT.exists(), 'Exclusive terminal audit already consumed')
    record = {'startedAt': now(), 'passed': False, 'status': 'AUDITING', 'scope': 'Independent read-only terminal evidence audit; no benchmark rerun/model/media/server/product workload.',
              'qualificationEvidence': False, 'inferenceRun': False, 'benchmarkRerun': False,
              'maximumAllowedAuditRssBytes': LIMIT, 'runs': [], 'comparisons': []}
    began = time.perf_counter()
    try:
        track(Path(__file__))
        memory_guard()
        report = load(RUN / 'report.json')
        require(report['passed'] is True and report['status'] == 'COMPLETED' and len(report['runs']) == 4 and len(report['comparisons']) == 2,
                'Benchmark not completely passed exactly four cases/two pairs')
        require(all(report[key] is True for key in ('sourcesUnchanged', 'allCompleteFixtureInputsUnchanged',
                    'actualImportedSourcesUnchangedAcrossMeasuredCases', 'freshBackendPrerequisiteUnchangedAtTerminal')), 'Terminal unchanged evidence flag missing')
        require(all(report[key] is False for key in ('inferenceRun', 'modelsImported', 'qualificationEvidence', 'actualRenderOrMediaRun')), 'Benchmark scope changed')
        require(report['sourceBindingsBefore'] == report['sourceBindingsAfter']
                and report['inputBindingsBefore'] == report['inputBindingsAfter'], 'Source/input before-after maps differ')
        maps(report['sourceBindingsBefore'])
        maps(report['inputBindingsBefore'])
        maps(report['actualImportedSourceBindingsUnion'])
        maps(load(HERE / 'prepared_bindings.json'))
        require({name(path) for path in RUN.rglob('*') if path.is_file()} == set(report['retainedFileBindingsAtTerminal']) | {name(RUN / 'report.json')},
                'Consumed benchmark full file inventory changed')
        maps(report['retainedFileBindingsAtTerminal'])
        record['backend'] = audit_backend(report)
        record['samplingAndImportOrderEvidence'] = audit_code_order()
        expected_order = [(600, 'old'), (600, 'new'), (18000, 'old'), (18000, 'new')]
        require([(row['frames'], row['mode']) for row in report['runs']] == expected_order, 'One-shot case order differs')
        for resource in report['runs']:
            record['runs'].append(audit_case(resource['mode'], resource['frames'], resource, psutil))
        for count in (600, 18000):
            fixture = load(RUN / f'fixture_{count}_receipt.json')
            fixture_binding = {key: fixture['fixture'][key] for key in ('bytes', 'sha256')}
            track(RUN / f'fixture_{count}.jsonl', fixture_binding)
            require(fixture['frames'] == count and fixture['passed'] is True and fixture['inputFileCountPerPair'] == 1
                    and fixture['completePoseJoints'] == count * 34 and fixture['fullRacketIdentities'] == count * 2,
                    'Complete immutable fixture scope/count differs')
            pair = load(RUN / f'comparison_{count}.json')
            require(pair == next(item for item in report['comparisons'] if item['frames'] == count)
                    and pair['passed'] is True and pair['fieldExclusions'] == [] and set(pair['outputs']) == set(OUTPUTS), 'Pair full comparison scope differs')
            exact = {}
            for output in OUTPUTS:
                old, new = RUN / f'{count}_old' / output, RUN / f'{count}_new' / output
                require(pair['outputs'][output]['exactBytesEqual'] is True, 'Full comparison assertion missing')
                track(old, pair['outputs'][output]['old'])
                track(new, pair['outputs'][output]['new'])
                exact[output] = exact_files(old, new)
            native = audit_native(count, fixture)
            exports = {}
            for filename, key, kind, metadata_key in (('player_motion.json', 'pose_frames', 'pose', 'playerMotion'),
                                                       ('racket_tracking.json', 'frames', 'racket', 'racketTracking')):
                exports[filename] = audit_export(RUN / f'{count}_new' / filename, count, RUN / f'fixture_{count}.jsonl',
                                                 fixture['metadata'][metadata_key], key, kind)
            record['comparisons'].append({'frames': count, 'passed': True, 'completeExactFileComparisons': exact,
                                          'nativeAllRows': native, 'completeRawExports': exports})
        require(all(binding(resolve(pathname)) == value for pathname, value in TRACKED.items()), 'Bound evidence/source changed during streaming audit')
        memory_guard()
        record['inputAndEvidenceBindings'] = TRACKED
        record['allAuditBindingsUnchanged'] = True
        record['fullFileComparisonsIndependentlyPassed'] = sum(len(row['completeExactFileComparisons']) for row in record['comparisons'])
        record['nativeRowsAuditedAcrossBothModes'] = sum(row['nativeAllRows']['framesPerOldAndNewOutput'] * 2 for row in record['comparisons'])
        record['exportRowsFullyDecodedPerKindAcrossPairs'] = sum(row['nativeAllRows']['framesPerOldAndNewOutput'] for row in record['comparisons'])
        record['memoryScope'] = 'Actual interpreter Windows OS peak working set and process commit are distinct from sampled aggregate owned-tree RSS/private. Peak_pagefile is commit, not disk size. Imports/export/fidelity work count toward process peaks; shared RSS can be counted twice and sampled peaks can be missed.'
        record['diskScope'] = 'Database/sidecar lengths are actual child-reported pre-cleanup logical sizes, arithmetically checked against bound receipts; scratch is absent now. The audit does not retrospectively remeasure deleted databases or treat lengths as allocated blocks.'
        record['limitations'] = ['One paired case per count, no repetitions or throughput qualification.',
                                'Dense synthetic pose/racket observations only; complete unknown/empty ground-motion summaries intentionally exclude unrelated pipeline arrays.',
                                'No real render, inference, model, accuracy, physical-speed/3D or full-match qualification.',
                                'The disk-backed 18000 case uses less resident memory and is slower in construction/indexed-read/export spans; these are separate outcomes.']
        record['passed'] = True
        record['status'] = 'PASSED'
    except BaseException as error:
        record['status'] = 'FAILED'
        record['error'] = type(error).__name__ + ': ' + str(error)
        record['traceback'] = traceback.format_exc()
    finally:
        record['completedAt'] = now()
        record['auditElapsedSeconds'] = time.perf_counter() - began
        record['maximumObservedAuditRssBytes'] = MAX_OBSERVED_RSS
        actual_memory = PROCESS.memory_info()._asdict()
        record['actualAuditOsPeakWorkingSetBytes'] = actual_memory.get('peak_wset')
        if actual_memory.get('peak_wset', 0) > LIMIT:
            record['passed'] = False
            record['status'] = 'FAILED'
            record['auditMemoryLimitFailure'] = True
        with RECEIPT.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: record.get(key) for key in ('passed', 'status', 'fullFileComparisonsIndependentlyPassed',
                    'nativeRowsAuditedAcrossBothModes', 'actualAuditOsPeakWorkingSetBytes', 'auditElapsedSeconds', 'error')}, allow_nan=False))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
