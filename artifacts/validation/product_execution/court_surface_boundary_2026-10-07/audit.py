"""Independent terminal arithmetic/integrity audit; never rerun a court fit.

Run only after the owned benchmark is terminal. The only writes are into one
new audit directory. This module imports neither candidate nor benchmark,
OpenCV, NumPy, SciPy, Torch or a model. Saved PNG pixels are decoded with the
standard library; source videos are not decoded again. Historical original
refinement parity remains an in-trial assertion because actual replay arrays
were not persisted. Native errors, support, boundary locks and proposal
preservation are independently recomputed from the recorded observations.
"""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import sys
import traceback
import xml.etree.ElementTree as ET
import zlib

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
RUN = UNIT / 'run'
LINES = ((0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
STAGES = ('projected', 'original', 'boundary_baseline', 'candidate')
GUARDS = {'preflightAvailableBytes': 750000000, 'maximumOwnedRssBytes': 1000000000,
          'minimumAvailableBytes': 400000000, 'lowSamples': 5, 'sampleSeconds': .25, 'wallSeconds': 600}
EXPECTED_CANDIDATE = '0e0a6420432ca3866326d0fe6c5fa69494110c2719e6faa6b7a658931ebccfd0'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify(values):
    for path, expected in values.items():
        require(binding(resolve(path)) == expected, 'Binding changed: ' + path)


def near(first, second, message, tolerance=1e-8):
    if isinstance(first, (list, tuple)):
        require(isinstance(second, (list, tuple)) and len(first) == len(second), message + ' shape')
        for a, b in zip(first, second, strict=True):
            near(a, b, message, tolerance)
    elif first is None or second is None:
        require(first is second, message + ' null')
    else:
        require(math.isfinite(first) and math.isfinite(second) and math.isclose(first, second, abs_tol=tolerance, rel_tol=1e-12), message)


def distance(first, second):
    return math.sqrt(sum((a-b)**2 for a, b in zip(first, second, strict=True)))


def points(value):
    require(isinstance(value, list) and len(value) == 14, 'Expected fourteen ordered court points')
    for point in value:
        require(isinstance(point, list) and len(point) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in point), 'Invalid court point')
    return value


def multiply(values, scale):
    return [[coordinate * scale for coordinate in point] for point in values]


def score(record, labels, predictions):
    require(len(record['landmarks']) == len(labels) == len(predictions) == 14, 'Landmark scorer identity count')
    counts, correct = Counter(), []
    for identity, (row, label, prediction) in enumerate(zip(record['landmarks'], labels, predictions, strict=True)):
        require(row['identity'] == identity and row['label'] == label, 'Landmark identity or frozen label changed')
        near(row['prediction'], prediction, 'Scorer prediction disagrees with recorded court')
        eligible = 0 <= label[0] < 1280 and 0 <= label[1] < 720
        error = None if prediction is None else distance(prediction, label)
        hit = eligible and error is not None and error <= 7
        require(row['eligible'] is eligible and row['correct'] is hit, 'Inclusive seven-pixel score classification differs')
        near(row['distance_native_px'], error, 'Native pixel error differs', 1e-9)
        if eligible:
            counts['tp'] += int(hit)
            counts['fn'] += int(not hit)
            counts['fp'] += int(prediction is not None and not hit)
        correct.append(hit)
    require({key: record[key] for key in ('tp', 'fp', 'fn')} == {key: counts[key] for key in ('tp', 'fp', 'fn')}, 'Score counts differ')
    return {key: counts[key] for key in ('tp', 'fp', 'fn')}, correct


def proposals(entry):
    baseline = entry['baselineProposals']
    observed = entry['proposals']
    preserved = [row for row in observed if row['evidence_origin'] == 'BASELINE']
    encode = lambda row: json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(Counter(map(encode, baseline)) == Counter(encode(row['baseline_record']) for row in preserved), 'Baseline proposal multiset/duplicates changed')
    require([encode(row) for row in baseline] == [encode(row['baseline_record']) for row in preserved], 'Relative baseline proposal ordering changed')
    lsd = set()
    for row in observed:
        require(row['lsd_index'] not in lsd, 'Duplicate LSD observation')
        lsd.add(row['lsd_index'])
        require(len(row['segment']) == 4 and all(math.isfinite(v) for v in row['segment']), 'Invalid recorded segment')
        require(row['support_fraction'] >= .5 and row['length'] >= 30, 'Candidate passed a rejected segment')
        if row['evidence_origin'] == 'BASELINE':
            require(all(row[key] == value for key, value in row['baseline_record'].items()), 'Passing baseline photometry or coordinates changed')
        else:
            require(row['evidence_origin'] == 'SURFACE_BOUNDARY_RECOVERY' and row['baseline_record']['support_fraction'] < .5, 'Recovery did not originate in baseline rejection')
            require(row['boundary_gate_pass'] is True and row['boundary_pattern_fraction'] >= .8 and row['boundary_replacement_sample_count'] > 0, 'Conditional recovery lacks boundary evidence')
            require(row['boundary_inside_side'] in ('NEGATIVE_NORMAL', 'POSITIVE_NORMAL'), 'Unknown boundary side')
    require(entry['baselineProposalCount'] == len(baseline) and entry['candidateProposalCount'] == len(observed), 'Proposal counts differ')
    require(entry['addedProposalCount'] == len(observed)-len(baseline), 'Added proposal count differs')
    return [row['segment'] for row in observed], [row['segment'] for row in baseline]


def support(values, segments, width, height):
    rows = []
    for first, last in LINES:
        a, b = values[first], values[last]
        if a is None or b is None:
            rows.append({'supported_fraction': 0., 'visible_samples': 0, 'reason': 'MISSING_ENDPOINT'})
            continue
        positions = [[a[axis]+(.05+i*(.9/99))*(b[axis]-a[axis]) for axis in (0, 1)] for i in range(100)]
        visible = [point for point in positions if 0 <= point[0] < width and 0 <= point[1] < height]
        if len(visible) < 10 or not segments:
            rows.append({'supported_fraction': 0., 'visible_samples': len(visible), 'reason': 'NO_COMPARABLE_SEGMENTS'})
            continue
        supported = 0
        for point in visible:
            best = math.inf
            for x1, y1, x2, y2 in segments:
                vx, vy = x2-x1, y2-y1
                along = min(1., max(0., ((point[0]-x1)*vx+(point[1]-y1)*vy)/max(vx*vx+vy*vy, 1e-8)))
                best = min(best, distance(point, [x1+along*vx, y1+along*vy]))
            supported += best <= 3
        rows.append({'supported_fraction': supported/len(visible), 'visible_samples': len(visible)})
    considered = [row for row in rows if row['visible_samples'] >= 10]
    mean = sum(row['supported_fraction'] for row in considered)/len(considered) if considered else 0.
    return {'lines': rows, 'mean_visible_line_support': mean, 'considered_lines': len(considered), 'diagnostic_gate_pass': len(considered) >= 6 and mean >= .5}


def compare_support(record, expected):
    require(len(record['lines']) == 9 and record['considered_lines'] == expected['considered_lines'], 'Support line identities differ')
    require(record['diagnostic_gate_pass'] is expected['diagnostic_gate_pass'], 'Support admission differs')
    near(record['mean_visible_line_support'], expected['mean_visible_line_support'], 'Mean line support differs')
    for actual, replay in zip(record['lines'], expected['lines'], strict=True):
        require(actual['visible_samples'] == replay['visible_samples'] and actual.get('reason') == replay.get('reason'), 'Visible support sample count/reason differs')
        near(actual['supported_fraction'], replay['supported_fraction'], 'Line support differs')


def projected(values, matrix):
    require(len(matrix) == 3 and all(len(row) == 3 for row in matrix), 'Transform dimensions')
    result = []
    for x, y in values:
        x, y = x/960., y/960.
        denominator = matrix[2][0]*x+matrix[2][1]*y+matrix[2][2]
        require(denominator > .1, 'Recorded transform crosses projective horizon')
        result.append([(matrix[axis][0]*x+matrix[axis][1]*y+matrix[axis][2])/denominator*960 for axis in (0, 1)])
    return result


def refinement(seed, returned, evidence, segments, width, height):
    points(seed); points(returned)
    compare_support(evidence['before'], support(seed, segments, width, height))
    locked, last = set(), seed
    for iteration in evidence['iterations']:
        matches = iteration['matches']
        ids = {row['line'] for row in matches}
        require(len(matches) == len(ids) >= 6 and len(ids & {0, 1, 6, 7}) >= 2 and len(ids & {2, 3, 4, 5, 8}) >= 2, 'Semantic line/family requirement differs')
        require(len({row['segment'] for row in matches}) == len(matches) and all(0 <= row['segment'] < len(segments) for row in matches), 'Segment-to-semantic identity invalid')
        locked.update({0, 1, 2, 3}-ids)
        require(iteration['locked_boundary_lines'] == sorted(locked), 'Missing outer boundary was unlocked')
        last = projected(seed, iteration['normalized_image_transform'])
        near(iteration['maximum_displacement_reference_px'], max(distance(a, b) for a, b in zip(seed, last, strict=True)), 'Recorded maximum displacement differs')
        for line in locked:
            a, b = (seed[index] for index in LINES[line])
            length = distance(a, b)
            for index in LINES[line]:
                p = last[index]
                require(abs((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))/length < 1e-6, 'Unsupported boundary moved off original infinite line')
        if evidence['accepted']:
            require(iteration['solver_success'] is True and iteration['maximum_displacement_reference_px'] <= 20, 'Accepted fit bypassed stability gate')
    if 'proposed_after' in evidence:
        compare_support(evidence['proposed_after'], support(last, segments, width, height))
    if evidence['accepted']:
        require(len(evidence['iterations']) == 3 and evidence['reason'] == 'PROJECTIVE_IMAGE_SUPPORT_IMPROVED', 'Accepted refinement reason/iteration count differs')
        near(returned, last, 'Accepted court differs from final recorded transform')
        after = evidence['proposed_after']
        require(after['diagnostic_gate_pass'] and sum(row['supported_fraction'] >= .5 for row in after['lines']) >= 6 and after['mean_visible_line_support'] >= evidence['before']['mean_visible_line_support']+.02, 'Accepted fit bypassed support improvement')
    else:
        near(returned, seed, 'Rejected fit changed seed')


def png_pixels(path):
    """Decode bounded OpenCV RGB8 noninterlaced PNG bytes into native BGR."""
    data = Path(path).read_bytes()
    require(data[:8] == b'\x89PNG\r\n\x1a\n', 'Original is not PNG')
    offset, compressed, shape = 8, bytearray(), None
    while offset < len(data):
        length = struct.unpack('!I', data[offset:offset+4])[0]
        kind, payload = data[offset+4:offset+8], data[offset+8:offset+8+length]
        checksum = struct.unpack('!I', data[offset+8+length:offset+12+length])[0]
        require(zlib.crc32(kind+payload) & 0xffffffff == checksum, 'PNG CRC changed')
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('!IIBBBBB', payload)
            require(depth == 8 and color == 2 and compression == filtering == interlace == 0, 'Unsupported saved-original PNG type')
            require(0 < width*height*3 <= 64000000, 'PNG decode exceeds bounded still budget')
            shape = width, height
        elif kind == b'IDAT':
            compressed.extend(payload)
        offset += 12+length
        if kind == b'IEND':
            break
    require(shape is not None, 'PNG shape missing')
    width, height = shape
    stride = width*3
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, height*(stride+1)+1)
    require(decoder.eof and len(raw) == height*(stride+1), 'PNG pixel payload size differs')
    previous, rgb = bytearray(stride), bytearray()
    def paeth(a, b, c):
        prediction = a+b-c
        distances = abs(prediction-a), abs(prediction-b), abs(prediction-c)
        return (a, b, c)[distances.index(min(distances))]
    for y in range(height):
        start = y*(stride+1)
        filter_kind = raw[start]
        row = bytearray(raw[start+1:start+1+stride])
        require(filter_kind in range(5), 'Unknown PNG row filter')
        for x in range(stride):
            a, b, c = row[x-3] if x >= 3 else 0, previous[x], previous[x-3] if x >= 3 else 0
            correction = (0, a, b, (a+b)//2, paeth(a, b, c))[filter_kind]
            row[x] = (row[x]+correction) & 255
        rgb.extend(row)
        previous = row
    bgr = bytearray(len(rgb))
    bgr[0::3], bgr[1::3], bgr[2::3] = rgb[2::3], rgb[1::3], rgb[0::3]
    return [width, height], hashlib.sha256(bgr).hexdigest()


def product():
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    return {'jobs': rows, 'files': {name(path): binding(path) for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}}


def case_inputs():
    result = []
    for group, folder in zip(('selection', 'external_source_check', 'pilot'), ('court_refinement_selection01', 'court_refinement_raw_labels01', 'court_line_refinement01'), strict=True):
        historical = read(ROOT / 'outputs/vision_upgrade_audit' / folder / 'report.json')
        require(historical['complete'] and historical['qualification_evidence'] is False, 'Historical report scope changed')
        for old in historical['cases'] if group == 'pilot' else historical['images']:
            identity = old.get('id', old.get('identity'))
            path = Path(old['video']) if group == 'pilot' else ROOT / 'data/external/court_heatmap_pilot/images' / (identity+'.png')
            result.append({'group': group, 'id': identity, 'path': path, 'frame': old['frame'] if group == 'pilot' else None, 'old': old,
                           'sha256': old['video_sha256'] if group == 'pilot' else old['input_sha256']})
    return result


def auto_inputs():
    result = [{'id': row['id'], 'group': 'Commons_development', 'path': ROOT / row['inputPath'], 'sha256': row['inputSha256']}
              for row in read(ROOT / 'data/demos/sources.json')['sources']]
    for dataset in ('uvy_tennis_videos', 'caltennis_diagnostic'):
        manifest = read(ROOT / 'data/external' / dataset / 'manifest.json')
        require(manifest['complete'], 'Incomplete source manifest')
        files = {row['path']: row for row in manifest['files']}
        sequences = manifest['sequences'].items() if isinstance(manifest['sequences'], dict) else enumerate(manifest['sequences'], 1)
        result.extend({'id': dataset+'_'+str(identity), 'group': dataset, 'path': ROOT / 'data/external' / dataset / row['video'], 'sha256': files[row['video']]['sha256']} for identity, row in sequences)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=UNIT / 'audit')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == UNIT and output.name.startswith('audit'), 'Audit output must be one new directory in this unit')
    require((RUN / 'resource_receipt.json').is_file(), 'Wait for root terminal receipt before audit execution')
    output.mkdir(exist_ok=False)
    receipt = {'startedAt': datetime.now(timezone.utc).isoformat(), 'passed': False, 'qualificationEvidence': False, 'productionPromoted': False,
               'originalRunStatus': 'ORIGINAL_RUN_FAILED_FINALIZATION', 'originalRunPassed': False,
               'auditOnly': True, 'candidateOrFitExecuted': False, 'sourceVideosDecoded': False,
               'scope': 'Independent terminal record arithmetic, proposal preservation, saved PNG pixels, bindings, process retirement and saved-product integrity; reused development only.',
               'limitations': ['Historical original-refinement maxDifference is an in-trial asserted parity value; actual replay arrays were not persisted and are not independently reproduced.',
                               'Source video first-frame identity is checked against the bound sequential-decode harness and source hash, not a second video decode.',
                               'Photometric per-sample boundary observations are checked for recorded invariants; candidate photometry is not rerun.',
                               'The original run failed finalization and has no report.json. This audit recovers flushed observations; it does not convert the original execution into a pass.',
                               'Loaded-module before/after inventories and runtime version inventory were not persisted and cannot be verified or reconstructed.',
                               'No independent court accuracy, physical geometry, participant accuracy or production qualification.']}
    before = {}
    try:
        prepared = read(UNIT / 'preparation.json')
        protocol = read(UNIT / 'protocol.json')
        resource = read(RUN / 'resource_receipt.json')
        require(not (RUN / 'report.json').exists(), 'This recovery auditor is scoped to the original missing final report')
        progress = [json.loads(line) for line in (RUN / 'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(progress) == 73 and [row['kind'] for row in progress] == ['existing_seed']*64+['automatic']*9, 'Expected all 73 complete ordered observations')
        report = {'cases': [row['entry'] for row in progress if row['kind'] == 'existing_seed'],
                  'automatic': [row['entry'] for row in progress if row['kind'] == 'automatic'],
                  'spectatorHypothesisControl': read(RUN / 'spectator_control.json')}
        recovery = read(UNIT / 'recovery.json')
        require(recovery['trialTerminalPassed'] is False and recovery['originalRunTerminalStatus'] == 'FAILED'
                and recovery['qualificationEvidence'] is False and recovery['productionPromoted'] is False,
                'Recovery promoted the failed trial')
        require(all(recovery[key] == report[key] for key in ('cases', 'automatic', 'spectatorHypothesisControl')),
                'Root recovery differs from the exact flushed observations')
        before = dict(prepared['bindings'])
        before[name(Path(__file__))] = binding(Path(__file__))
        before[name(UNIT / 'preparation.json')] = binding(UNIT / 'preparation.json')
        before[name(UNIT / 'recovery.json')] = binding(UNIT / 'recovery.json')
        before[name(UNIT / 'recover_records.py')] = binding(UNIT / 'recover_records.py')
        before.update({name(path): binding(path) for path in RUN.iterdir() if path.is_file()})
        verify(before)
        require(binding(UNIT / 'candidate.py')['sha256'] == EXPECTED_CANDIDATE, 'Frozen candidate differs')
        require(prepared['guards'] == GUARDS and prepared['cases'] == 64 and prepared['automaticSourceViews'] == 9, 'Prepared scope or guards changed')
        require(not prepared['modelsOrInference'] and not prepared['productionPromotion'], 'Preparation scope promoted')
        require(protocol['qualificationEvidence'] is False and protocol['productionPromoted'] is False and protocol['physicalSpeedOr3DEnabled'] is False, 'Protocol promotion')
        require(resource['status'] == 'FAILED' and resource['passed'] is False and resource['exitCode'] == 1 and resource['allOwnedIdentitiesRetired'] and not resource['remainingOwnedIdentities'], 'Original failed run has not retired')
        require('safeguardStop' not in resource and "AttributeError: 'numpy.ndarray' object has no attribute 'memory_info'" in (RUN / 'stdout.log').read_text(encoding='utf-8'), 'Recorded failure is not the scoped finalization failure')
        require(resource['sourcesUnchanged'] and resource['savedProductUnchanged'], 'Trial source/product preservation failed')
        require(product() == prepared['productBefore'], 'Saved product rows/files changed independently')
        tests_paths = [resolve(path) for path in prepared['bindings'] if path.endswith('/receipt.json') and '/tests' in path]
        require(len(tests_paths) == 2, 'Expected preserved first test failure and successful revised fixture run')
        test_history = []
        for test_path in tests_paths:
            tests = read(test_path)
            require(tests['sourceBefore'] == tests['sourceAfter'] and tests['sourcesUnchanged'] and tests['allOwnedIdentitiesRetired'], 'Focused test source or retirement changed')
            verify(tests['sourceBefore'])
            for leaf, expected in tests['outputs'].items():
                require(binding(test_path.parent / leaf) == expected, 'Focused test output changed')
            suites = list(ET.parse(test_path.parent / 'junit.xml').getroot().iter('testsuite'))
            counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
            counts['passed'] = counts['tests']-counts['failures']-counts['errors']-counts['skipped']
            require(counts == tests['counts'] and counts['tests'] > 0, 'Focused JUnit count differs')
            if test_path.parent.name == 'tests_v2':
                require(tests['passed'] and tests['exitCode'] == 0 and counts['passed'] == counts['tests'] == 36, 'Revised focused tests did not pass')
            else:
                require(test_path.parent.name == 'tests' and tests['passed'] is False and counts == {'tests': 35, 'failures': 1, 'errors': 0, 'skipped': 0, 'passed': 34}, 'First fixture failure was not preserved')
            test_history.append({'receipt': name(test_path), 'counts': counts, 'passed': tests['passed']})
        frozen_cases = case_inputs()
        require(len(frozen_cases) == len(report['cases']) == 64, 'Full existing case count differs')
        leading = read(ROOT / 'outputs/vision_upgrade_audit/court_ridge_boundary01/report.json')['cases']
        require(len(leading) == 64, 'Historical leading case count differs')
        pooled = {group: {stage: Counter() for stage in STAGES} for group in ('selection', 'external_source_check')}
        valid, invalid, labeled, changed, all_gains, all_losses = 0, 0, 0, 0, [], []
        for index, (entry, frozen, previous) in enumerate(zip(report['cases'], frozen_cases, leading, strict=True)):
            old, group = frozen['old'], frozen['group']
            require((entry['index'], entry['id'], entry['group'], entry['frame']) == (index, frozen['id'], group, frozen['frame']), 'Existing case identity/order changed')
            require((previous['id'], previous['group']) == (frozen['id'], group), 'Leading comparison identity differs')
            require(entry['source'] == name(frozen['path']) and entry['sourceBinding'] == binding(frozen['path']) and entry['sourceBinding']['sha256'] == frozen['sha256'], 'Existing source binding differs')
            initialized = 'original_points' in old if group == 'pilot' else old['initial_calibration']['is_valid']
            require(entry['initialValid'] is initialized and entry['conditionalCueEvaluated'] is initialized, 'Initialization availability differs')
            if not initialized:
                invalid += 1
                require(entry['accepted'] is False and entry['reason'] == 'INITIAL_GEOMETRIC_CALIBRATION_INVALID' and not entry.get('changed', False), 'Invalid initialization gained a court')
            else:
                valid += 1
                candidate_segments, baseline_segments = proposals(entry)
                seed, original, boundary, actual = (points(entry[key]) for key in ('pointsReference', 'previousReference', 'boundaryBaselineReference', 'candidateReference'))
                expected_seed = old['original_points'] if group == 'pilot' else multiply([row['prediction'] for row in old['scores']['projected']['landmarks']], .75)
                expected_original = old['refined_points'] if group == 'pilot' else multiply([row['prediction'] for row in old['scores']['refined']['landmarks']], .75)
                near(seed, expected_seed, 'Original seed changed'); near(original, expected_original, 'Frozen original output changed')
                require(0 <= entry['baselineReplayMaxDifferenceReferencePx'] < 1e-5, 'Recorded original parity assertion failed')
                width, height = entry['nativeSize']
                reference_height = round(height*960/width)
                refinement(seed, boundary, entry['boundaryBaselineEvidence'], baseline_segments, 960, reference_height)
                refinement(seed, actual, entry['evidence'], candidate_segments, 960, reference_height)
                require(entry['accepted'] is entry['evidence']['accepted'] and entry['reason'] == entry['evidence']['reason'], 'Candidate decision differs from refinement')
                actually_changed = entry['evidence']['accepted'] != entry['boundaryBaselineEvidence']['accepted'] or max(abs(a-b) for p, q in zip(actual, boundary, strict=True) for a, b in zip(p, q, strict=True)) > .1
                require(entry['changed'] is actually_changed, 'Same-optimizer changed decision differs')
                changed += actually_changed
                if actually_changed:
                    require(binding(resolve(entry['panel']['path'])) == {key: entry['panel'][key] for key in ('bytes', 'sha256')}, 'Changed court panel differs')
            if group != 'pilot':
                labeled += 1
                labels = [row['label'] for row in old['scores']['projected']['landmarks']]
                stage_hits = {}
                for stage in STAGES:
                    if initialized:
                        field = {'projected': 'pointsReference', 'original': 'previousReference', 'boundary_baseline': 'boundaryBaselineReference', 'candidate': 'candidateReference'}[stage]
                        prediction = multiply(entry[field], 1/.75)
                    else:
                        prediction = [row['prediction'] for row in old['scores']['projected']['landmarks']]
                    counts, stage_hits[stage] = score(entry['scores'][stage], labels, prediction)
                    pooled[group][stage].update(counts)
                require({key: entry['scores']['boundary_baseline'][key] for key in ('tp', 'fp', 'fn')} == {key: previous['scores']['boundary_only'][key] for key in ('tp', 'fp', 'fn')}, 'Complete historical boundary scorer count differs')
                for stage, suffix in (('original', 'Original'), ('boundary_baseline', 'BoundaryBaseline')):
                    gains = [i for i, (p, q) in enumerate(zip(stage_hits['candidate'], stage_hits[stage], strict=True)) if p and not q]
                    losses = [i for i, (p, q) in enumerate(zip(stage_hits['candidate'], stage_hits[stage], strict=True)) if q and not p]
                    require(entry.get('gainedVs'+suffix, []) == gains and entry.get('lostVs'+suffix, []) == losses, 'Gained/lost landmark identities differ')
                    if stage == 'boundary_baseline':
                        all_gains.extend({'case': index, 'identity': i} for i in gains)
                        all_losses.extend({'case': index, 'identity': i} for i in losses)
        require((valid, invalid, labeled) == (47, 17, 40), 'Availability/labeled denominators changed')
        for group in pooled:
            for stage in STAGES:
                require(dict(pooled[group][stage]) == {key: sum(row['scores'][stage][key] for row in report['cases'] if row['group'] == group) for key in ('tp', 'fp', 'fn')}, 'Independent pooled count differs from flushed scores')
                require(recovery['pooled'][group][stage] == dict(pooled[group][stage]), 'Root recovered pooled count differs')
        frozen_auto = auto_inputs()
        require(len(frozen_auto) == len(report['automatic']) == 9 and len({name(row['path']) for row in frozen_auto}) == 9, 'Nine exact automatic inputs differ')
        require(Counter(row['group'] for row in frozen_auto) == {'Commons_development': 3, 'uvy_tennis_videos': 3, 'caltennis_diagnostic': 3}, 'Automatic source group counts differ')
        admissions, baseline_admissions = [], []
        for index, (entry, frozen) in enumerate(zip(report['automatic'], frozen_auto, strict=True)):
            require((entry['index'], entry['id'], entry['group'], entry['source'], entry['exactSequentialFrame']) == (index, frozen['id'], frozen['group'], name(frozen['path']), 0), 'Automatic source/frame identity differs')
            require(entry['sourceBinding'] == binding(frozen['path']) and entry['sourceBinding']['sha256'] == frozen['sha256'] and entry['physicalCourtValidated'] is False, 'Automatic input binding/promotion differs')
            original = entry['original']
            require(binding(resolve(original['path'])) == {key: original[key] for key in ('bytes', 'sha256')}, 'Automatic original PNG binding differs')
            require(binding(resolve(entry['panel']['path'])) == {key: entry['panel'][key] for key in ('bytes', 'sha256')}, 'Automatic source-context panel binding differs')
            shape, pixel_hash = png_pixels(resolve(original['path']))
            require(shape == entry['nativeSize'] and pixel_hash == entry['decodedNativePixelSha256'], 'Saved PNG pixels differ from recorded native decode')
            candidate_segments, baseline_segments = proposals(entry)
            if 'seedReference' in entry:
                seed, actual, baseline = entry['seedReference'], entry['returnedReference'], entry['baselineReturnedReference']
                height = round(shape[1]*960/shape[0])
                refinement(seed, actual, entry['refinement'], candidate_segments, 960, height)
                refinement(seed, baseline, entry['baselineRefinement'], baseline_segments, 960, height)
                compare_support(entry['imageSupport'], support(actual, candidate_segments, 960, height))
                compare_support(entry['baselineImageSupport'], support(baseline, baseline_segments, 960, height))
                for prefix, fit, line_support in (('', entry['refinement'], entry['imageSupport']), ('baseline', entry['baselineRefinement'], entry['baselineImageSupport'])):
                    admitted = fit['accepted'] and line_support['diagnostic_gate_pass'] and sum(row['supported_fraction'] >= .5 for row in line_support['lines']) >= 6
                    key = 'acceptedImageHypothesis' if not prefix else 'baselineAcceptedImageHypothesis'
                    require(entry[key] is admitted, 'Automatic admission arithmetic differs')
                if entry['id'] == 'dimitrov_tiafoe_2024':
                    review = read(ROOT / 'outputs/vision_upgrade_audit/dimitrov_court_image01/review.json')
                    marks = review['corrected_approximate_coordinates_native_px']
                    require(len(marks) == 14 and marks[8] is None and sum(mark is not None for mark in marks) == 13, 'Corrected rough source marks differ')
                    scale = shape[0]/960
                    errors = [None if mark is None else distance([v*scale for v in point], mark) for point, mark in zip(actual, marks, strict=True)]
                    rough = entry['approximateSourceReview']
                    near(rough['distancesNativePx'], errors, 'Approximate source errors differ')
                    near(rough['meanNativeDistancePx'], sum(v for v in errors if v is not None)/13, 'Approximate source mean differs')
                    require(rough['uncertaintyNativePx'] == 8 and rough['qualifiedGroundTruth'] is False, 'Rough marks were promoted')
            else:
                require(entry['acceptedImageHypothesis'] is False and entry['addedProposalCount'] == 0 and entry['reason'] == 'NO_FOUR_CORNER_COMPONENT_AT_FROZEN_EPSILON', 'Absent surface gained automatic admission')
            if entry['acceptedImageHypothesis']:
                admissions.append(entry['id'])
            if entry.get('baselineAcceptedImageHypothesis', False):
                baseline_admissions.append(entry['id'])
        require(len(progress) == 73 and [row['entry'] for row in progress if row['kind'] == 'existing_seed'] == report['cases'] and [row['entry'] for row in progress if row['kind'] == 'automatic'] == report['automatic'], 'Complete append-only progress differs from terminal report')
        spectator = report['spectatorHypothesisControl']
        require(read(RUN / 'spectator_control.json') == spectator, 'Exclusive spectator record differs')
        old_false = read(ROOT / 'outputs/vision_upgrade_audit/court_calibration_support01/report.json')['cases'][166]
        require(old_false['index'] == 166 and old_false['id'] == 'tennis_V02' and old_false['default']['is_valid'] is False and binding(resolve(spectator['source']))['sha256'] == old_false['input_sha256'], 'Spectator source/case differs')
        require(spectator['sourceBinding'] == binding(resolve(spectator['source'])) and spectator['exactSequentialFrame'] == 0
                and spectator['nativeSize'] == old_false['native_size'] and spectator['originalProductionCalibrationRejected'] is True, 'Spectator source/decode semantics differ')
        spectator_source = next(entry for entry in report['automatic'] if entry['source'] == spectator['source'])
        require(spectator['decodedNativePixelSha256'] == spectator_source['decodedNativePixelSha256'], 'Spectator native pixels differ from same source first frame')
        require(binding(resolve(spectator['panel']['path'])) == {key: spectator['panel'][key] for key in ('bytes', 'sha256')}, 'Spectator source-context panel binding differs')
        candidate_segments, baseline_segments = proposals(spectator)
        seed = multiply(old_false['predictions']['alternate'], 960/old_false['native_size'][0])
        near(spectator['seedReference'], seed, 'Spectator seed differs')
        refinement(seed, spectator['returnedReference'], spectator['refinement'], candidate_segments, 960, 540)
        refinement(seed, spectator['baselineReturnedReference'], spectator['baselineRefinement'], baseline_segments, 960, 540)
        compare_support(spectator['imageSupport'], support(spectator['returnedReference'], candidate_segments, 960, 540))
        compare_support(spectator['baselineImageSupport'], support(spectator['baselineReturnedReference'], baseline_segments, 960, 540))
        for field, evidence, supported in (('acceptedImageHypothesis', spectator['refinement'], spectator['imageSupport']),
                                           ('baselineAcceptedImageHypothesis', spectator['baselineRefinement'], spectator['baselineImageSupport'])):
            admitted = evidence['accepted'] and supported['diagnostic_gate_pass'] and sum(row['supported_fraction'] >= .5 for row in supported['lines']) >= 6
            require(spectator[field] is admitted, 'Spectator admission arithmetic differs')
        sampled = [json.loads(line) for line in (RUN / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(sampled) == resource['samples'] > 0, 'Owned sample count differs')
        identities = {(row['pid'], row['createTimeUnixSeconds']): row for row in resource['ownedIdentities']}
        launcher = resource['launcherIdentity']
        launcher_key = launcher['pid'], launcher['createTimeUnixSeconds']
        require(launcher_key in identities and identities[launcher_key]['depth'] == 0, 'Recorded launcher ownership differs')
        actual = read(RUN / 'actual_identity.json')
        acknowledged = read(RUN / 'actual_sampled.json')
        require(acknowledged['identity'] == actual and acknowledged['sampledByExactIdentity'], 'Actual identity handshake differs')
        actual_key = actual['pid'], actual['createTimeUnixSeconds']
        seen, peaks, lows = Counter(), {}, 0
        for sample in sampled:
            rss = sum(row['memoryBytes']['rss'] for row in sample['ownedProcesses'])
            require(rss == sample['ownedTreeRssBytes'] and rss <= GUARDS['maximumOwnedRssBytes'] and sample['elapsedSeconds'] <= GUARDS['wallSeconds'], 'Recorded RSS/wall guard violated')
            lows = lows+1 if sample['availableRamBytes'] < GUARDS['minimumAvailableBytes'] else 0
            require(lows < GUARDS['lowSamples'], 'Low-RAM guard was crossed')
            for row in sample['ownedProcesses']:
                key = row['pid'], row['createTimeUnixSeconds']
                require(key in identities, 'Sample contains an unowned identity')
                seen[key] += 1
                peaks[key] = max(peaks.get(key, 0), row['memoryBytes']['rss'])
        require(seen[actual_key] > 0 and resource['peakOwnedTreeRssBytes'] == max(row['ownedTreeRssBytes'] for row in sampled), 'Actual interpreter was not sampled or peak differs')
        require(resource['minimumAvailableRamBytes'] <= min(row['availableRamBytes'] for row in sampled), 'Recorded minimum RAM differs')
        import psutil
        retired = []
        for key, record in identities.items():
            require(record['sampleCount'] == seen[key] and record['peakSampledRssBytes'] == peaks.get(key, 0), 'Owned identity sample summary differs')
            try:
                process = psutil.Process(key[0])
                live = process.is_running() and process.create_time() == key[1]
            except psutil.NoSuchProcess:
                live = False
            require(not live, 'Exact owned identity remains live')
            retired.append({'pid': key[0], 'createTimeUnixSeconds': key[1], 'retired': True})
        require(not any(module == 'torch' or module.startswith(('torch.', 'ultralytics', 'cv2', 'numpy', 'scipy')) for module in sys.modules), 'Auditor imported a prohibited heavy runtime')
        verify(before)
        require(not (RUN / 'report.json').exists(), 'Missing original final report was replaced during audit')
        require(product() == prepared['productBefore'], 'Saved product changed during independent audit')
        receipt.update(passed=True, verifiedExistingCases=64, validRefinements=valid, initialAbstentions=invalid, labeledCases=labeled,
                       independentlyCheckedNativeErrors=40*14*4, changedExistingCases=changed,
                       pooled={group: {stage: dict(counts) for stage, counts in stages.items()} for group, stages in pooled.items()},
                       gainedVsBoundaryBaseline=all_gains, lostVsBoundaryBaseline=all_losses,
                       automaticSourceViews=9, automaticAdmissions=admissions, baselineAutomaticAdmissions=baseline_admissions,
                       spectatorAccepted=spectator['acceptedImageHypothesis'], allOwnedIdentitiesRetired=True, retirementChecks=retired,
                       loadedModuleInventoryVerified=False, originalFinalReportRecovered=False,
                       focusedTestHistory=test_history,
                       inputsBefore=before, inputsAfter={path: binding(resolve(path)) for path in before}, sourcesUnchanged=True,
                       savedProductUnchanged=True, savedJobs=len(prepared['productBefore']['jobs']), savedFiles=len(prepared['productBefore']['files']))
    except BaseException as error:
        receipt.update(error=type(error).__name__+': '+str(error), traceback=traceback.format_exc(), inputsBefore=before)
    receipt['completedAt'] = datetime.now(timezone.utc).isoformat()
    save(output / 'receipt.json', receipt)
    print(json.dumps({key: receipt.get(key) for key in ('passed', 'error', 'verifiedExistingCases', 'validRefinements', 'initialAbstentions', 'labeledCases', 'automaticSourceViews')}), flush=True)
    return 0 if receipt['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
