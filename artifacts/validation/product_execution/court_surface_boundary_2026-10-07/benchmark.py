"""One frozen, guarded surface-boundary court development comparison."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
UNIT = Path(__file__).resolve().parent
RUN = UNIT / 'run'
BASE = Path('C:/Users/ac-98/AppData/Local/Programs/Python/Python313/python.exe')
MONITOR = ROOT / 'scripts/audit/measure_candidate_storage.py'
MONITOR_SHA256 = '8a6789800d16cc8decb5f6441169ca1a6705bfdd30ba4de2a1a04ad725b18c39'
OLD_FOLDERS = ('court_refinement_selection01', 'court_refinement_raw_labels01', 'court_line_refinement01')
GUARDS = {'preflightAvailableBytes': 750000000, 'maximumOwnedRssBytes': 1000000000,
          'minimumAvailableBytes': 400000000, 'lowSamples': 5, 'sampleSeconds': .25, 'wallSeconds': 600}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        return {'bytes': path.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def name(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def handshake(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.prepared')
    save(temporary, value)
    assert not path.exists()
    temporary.rename(path)


def verify(values):
    for path, expected in values.items():
        target = Path(path) if Path(path).is_absolute() else ROOT / path
        assert binding(target) == expected, 'Frozen input changed: ' + path


def ordinary_sources():
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).is_file():
            path = Path(filename).resolve()
            result[name(path)] = binding(path)
    return result


def module_at(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def cases():
    rows = []
    for group, folder in zip(('selection', 'external_source_check', 'pilot'), OLD_FOLDERS, strict=True):
        report = load(ROOT / 'outputs/vision_upgrade_audit' / folder / 'report.json')
        assert report['complete'] and report['qualification_evidence'] is False
        for old in report['cases'] if group == 'pilot' else report['images']:
            identity = old.get('id', old.get('identity'))
            path = Path(old['video']) if group == 'pilot' else ROOT / 'data/external/court_heatmap_pilot/images' / f'{identity}.png'
            expected = old['video_sha256'] if group == 'pilot' else old['input_sha256']
            rows.append({'group': group, 'old': old, 'id': identity, 'path': path, 'expectedSha256': expected,
                         'frame': old['frame'] if group == 'pilot' else None})
    assert len(rows) == 64
    return rows


def automatic_inputs():
    result = []
    public = load(ROOT / 'data/demos/sources.json')
    for row in public['sources']:
        result.append({'id': row['id'], 'group': 'Commons_development', 'path': ROOT / row['inputPath'],
                       'expectedSha256': row['inputSha256'], 'frame': 0})
    for dataset in ('uvy_tennis_videos', 'caltennis_diagnostic'):
        manifest = load(ROOT / 'data/external' / dataset / 'manifest.json')
        assert manifest['complete']
        records = {row['path']: row for row in manifest['files']}
        sequences = list(manifest['sequences'].items()) if isinstance(manifest['sequences'], dict) else list(enumerate(manifest['sequences'], 1))
        for identity, row in sequences:
            result.append({'id': dataset + '_' + str(identity), 'group': dataset,
                           'path': ROOT / 'data/external' / dataset / row['video'],
                           'expectedSha256': records[row['video']]['sha256'], 'frame': 0})
    assert len(result) == 9
    return result


def product_snapshot():
    database = ROOT / 'outputs/product/analyses.sqlite3'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute('SELECT * FROM analyses ORDER BY id')]
    assert len(rows) == 10 and not any(row['status'] in ('UPLOADED', 'PROCESSING') for row in rows)
    files = {name(path): binding(path) for path in (ROOT / 'outputs/product/analyses').rglob('*') if path.is_file()}
    return {'jobs': rows, 'files': files}


def input_bindings():
    inputs = [Path(__file__), UNIT / 'candidate.py', UNIT / 'test_candidate.py', UNIT / 'test_candidate_v2.py', UNIT / 'protocol.json', UNIT / 'run_tests.py', UNIT / 'run_tests_v2.py', UNIT / 'prepare_test_followup.py', UNIT / 'test_followup_preparation.json', MONITOR, BASE]
    inputs.extend(UNIT / folder / leaf for folder in ('tests', 'tests_v2') for leaf in ('receipt.json', 'junit.xml'))
    scripts = ('probe_court_line_segments.py', 'probe_court_surface_support.py', 'refine_court_line_geometry.py',
               'refine_court_boundary_geometry.py', 'audit_court_line_support.py', 'audit_caltennis_projection.py',
               'benchmark_court_refinement_labels.py')
    inputs.extend(ROOT / 'scripts/evaluate' / leaf for leaf in scripts)
    inputs.extend(ROOT / 'outputs/vision_upgrade_audit' / folder / 'report.json' for folder in OLD_FOLDERS)
    inputs.extend(ROOT / 'outputs/vision_upgrade_audit' / folder / 'report.json' for folder in
                  ('court_ridge_boundary01', 'court_ridge_ablation_no_sides01', 'court_calibration_support01', 'dimitrov_court_image01'))
    inputs.extend(ROOT / 'outputs/vision_upgrade_audit/dimitrov_court_image01' / leaf for leaf in
                  ('review.json', '01_original_frame0.png', 'boundary_evidence.json'))
    inputs.extend(ROOT / 'data/external' / folder / 'manifest.json' for folder in
                  ('court_heatmap_pilot', 'uvy_tennis_videos', 'caltennis_diagnostic'))
    inputs.append(ROOT / 'data/demos/sources.json')
    inputs.extend(path for path in (ROOT / 'src').rglob('*.py'))
    inputs.append(ROOT / 'configs/phase6_analytics/wasb_tiled_residual_pose_small_validation.yaml')
    for row in cases() + automatic_inputs():
        assert binding(row['path'])['sha256'] == row['expectedSha256'], 'Historical image/video changed: ' + row['id']
        inputs.append(row['path'])
    return {name(path): binding(path) for path in inputs}


def prepare():
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    for path in (UNIT / 'candidate.py', UNIT / 'test_candidate.py', UNIT / 'test_candidate_v2.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    tests = load(UNIT / 'tests_v2/receipt.json')
    assert tests['passed'] and tests['sourcesUnchanged'] and tests['allOwnedIdentitiesRetired']
    assert tests['sourceBefore'] == tests['sourceAfter'] and tests['exitCode'] == 0
    assert tests['counts']['passed'] == tests['counts']['tests'] > 0
    assert not any(tests['counts'][key] for key in ('failures', 'errors', 'skipped'))
    verify(tests['sourceBefore'])
    for leaf, expected in tests['outputs'].items():
        assert binding(UNIT / 'tests_v2' / leaf) == expected, 'Focused test output changed'
    assert binding(MONITOR)['sha256'] == MONITOR_SHA256
    assert not RUN.exists()
    values = input_bindings()
    save(UNIT / 'preparation.json', {'preparedAt': stamp(), 'status': 'PREPARED_NOT_EXECUTED',
        'bindings': values, 'productBefore': product_snapshot(), 'guards': GUARDS,
        'cases': 64, 'automaticSourceViews': 9, 'modelsOrInference': False, 'productionPromotion': False})
    print(json.dumps({'prepared': True, 'boundInputs': len(values), 'modelsOrInference': False}), flush=True)


def child():
    import psutil
    before = load(UNIT / 'preparation.json')
    assert RUN.is_dir() and (RUN / 'launch_marker.json').is_file()
    verify(before['bindings'])
    actual = psutil.Process()
    identity = {'pid': actual.pid, 'createTimeUnixSeconds': actual.create_time()}
    handshake(RUN / 'actual_identity.json', identity)
    deadline = time.monotonic() + 10
    while not (RUN / 'actual_sampled.json').exists():
        assert time.monotonic() < deadline, 'Actual interpreter sampling acknowledgement timed out'
        time.sleep(.01)
    assert load(RUN / 'actual_sampled.json')['identity'] == identity
    assert Path(actual.exe()).resolve() == BASE.resolve()
    began = time.perf_counter()
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    from scripts.evaluate.probe_court_line_segments import propose as original_propose
    from scripts.evaluate.refine_court_line_geometry import refine as original_refine
    from scripts.evaluate.refine_court_boundary_geometry import refine
    from scripts.evaluate.audit_court_line_support import support
    from scripts.evaluate.audit_caltennis_projection import LINES
    from src.court.court_geometry import TennisCourtGeometry
    candidate = module_at('_surface_boundary_candidate', UNIT / 'candidate.py')
    cv2.setNumThreads(1)
    import scipy
    ordinary_before = ordinary_sources()
    # The old scorer's whole module imports a model. Extract its unchanged pure
    # function only; no model module is executed in this development replay.
    scorer_path = ROOT / 'scripts/evaluate/benchmark_court_refinement_labels.py'
    scorer_node = next(node for node in ast.parse(scorer_path.read_text(encoding='utf-8')).body
                       if isinstance(node, ast.FunctionDef) and node.name == 'score')
    scorer_scope = {'np': np}
    exec(compile(ast.Module(body=[scorer_node], type_ignores=[]), str(scorer_path), 'exec'), scorer_scope)
    score = scorer_scope['score']
    report = {'startedAt': stamp(), 'complete': False, 'qualificationEvidence': False, 'productionPromoted': False,
              'modelInferenceRun': False, 'actualIdentity': identity, 'cases': [], 'automatic': [],
              'ordinarySourcesBefore': ordinary_before,
              'environment': {'python': sys.version, 'numpy': np.__version__, 'opencv': cv2.__version__, 'scipy': scipy.__version__},
              'scope': 'Reused development label agreement and image-supported initialization; no independent or physical accuracy qualification.'}

    def decode(path, frame=None):
        if frame is None:
            image = cv2.imread(str(path))
            assert image is not None
        else:
            capture = cv2.VideoCapture(str(path))
            try:
                image = None
                for _ in range(frame + 1):
                    ok, image = capture.read()
                    assert ok, 'Exact sequential frame unavailable'
            finally:
                capture.release()
        return image

    def segments(rows):
        return np.asarray([row['segment'] for row in rows if row['support_fraction'] >= .5], dtype=float).reshape(-1, 2, 2)

    def panel(image, points, title):
        output = image.copy()
        if points is not None:
            for a, b in LINES:
                cv2.line(output, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (20, 30, 240), 1, cv2.LINE_AA)
        output = cv2.copyMakeBorder(output, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(output, title, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
        return output

    def selected(image):
        surface = candidate.select_surface(image)
        # Explicit absence avoids selecting a different surface or hull mask.
        mask = None if surface is None else surface['component']['mask']
        rows = candidate.propose(image, mask)
        expected = [row for row in original_propose(image) if row['support_fraction'] >= .5]
        actual = [row['baseline_record'] for row in rows if row['evidence_origin'] == 'BASELINE']
        # Include duplicates; preserve every original passing row exactly.
        assert sorted(json.dumps(row, sort_keys=True) for row in expected) == sorted(json.dumps(row, sort_keys=True) for row in actual)
        for row in rows:
            if row['evidence_origin'] == 'BASELINE':
                assert all(row[key] == value for key, value in row['baseline_record'].items())
        return surface, rows, expected

    leading = load(ROOT / 'outputs/vision_upgrade_audit/court_ridge_boundary01/report.json')
    assert len(leading['cases']) == 64
    try:
        with (RUN / 'progress.jsonl').open('x', encoding='utf-8') as progress:
            for index, row in enumerate(cases()):
                old, group = row['old'], row['group']
                valid = 'original_points' in old if group == 'pilot' else old['initial_calibration']['is_valid']
                entry = {'index': index, 'id': row['id'], 'group': group, 'frame': row['frame'],
                         'source': name(row['path']), 'sourceBinding': binding(row['path']), 'initialValid': valid}
                leading_row = leading['cases'][index]
                assert leading_row['id'] == row['id'] and leading_row['group'] == group
                if not valid:
                    entry.update(accepted=False, reason='INITIAL_GEOMETRIC_CALIBRATION_INVALID',
                                 conditionalCueEvaluated=False)
                    if group != 'pilot':
                        entry['scores'] = {key: old['scores']['projected'] for key in ('projected', 'original', 'boundary_baseline', 'candidate')}
                else:
                    image = decode(row['path'], row['frame'])
                    reference = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))
                    entry['decodedNativePixelSha256'] = hashlib.sha256(image.tobytes()).hexdigest()
                    entry['nativeSize'] = [image.shape[1], image.shape[0]]
                    labels = None
                    if group == 'pilot':
                        points, expected = np.asarray(old['original_points']), np.asarray(old['refined_points'])
                        old_evidence = old
                    else:
                        points = np.asarray([point['prediction'] for point in old['scores']['projected']['landmarks']]) * .75
                        expected = np.asarray([point['prediction'] for point in old['scores']['refined']['landmarks']]) * .75
                        labels = [point['label'] for point in old['scores']['projected']['landmarks']]
                        old_evidence = old['refinement']
                    baseline_rows = original_propose(reference)
                    baseline_segments = segments(baseline_rows)
                    replayed, evidence = original_refine(points, baseline_segments, 960, reference.shape[0])
                    difference = float(np.max(np.abs(replayed - expected)))
                    assert difference < 1e-5 and evidence['accepted'] == old_evidence['accepted'], ('Historical baseline replay differs', index, difference)
                    boundary_baseline, boundary_evidence = refine(points, baseline_segments, 960, reference.shape[0])
                    surface, proposed, baseline_passing = selected(reference)
                    actual, revised = refine(points, segments(proposed), 960, reference.shape[0])
                    changed = bool(revised['accepted'] != boundary_evidence['accepted'] or np.max(np.abs(actual - boundary_baseline)) > .1)
                    entry.update(accepted=revised['accepted'], reason=revised['reason'], conditionalCueEvaluated=True,
                                 baselineReplayMaxDifferenceReferencePx=difference,
                                 baselineProposalCount=len(baseline_segments), baselineProposals=baseline_passing, candidateProposalCount=len(segments(proposed)),
                                 addedProposalCount=sum(item.get('evidence_origin') == 'SURFACE_BOUNDARY_RECOVERY' for item in proposed),
                                 surface=None if surface is None else {key: value for key, value in surface['component'].items() if key != 'mask'},
                                 pointsReference=points.tolist(), previousReference=expected.tolist(), candidateReference=actual.tolist(),
                                 boundaryBaselineReference=boundary_baseline.tolist(), boundaryBaselineEvidence=boundary_evidence,
                                 evidence=revised, proposals=proposed, changed=changed,
                                 baselinePointsUnchangedInProduction=True)
                    if labels is not None:
                        entry['scores'] = {key: score((values / .75).tolist(), labels) for key, values in
                                           (('projected', points), ('original', expected), ('boundary_baseline', boundary_baseline), ('candidate', actual))}
                        assert all(entry['scores']['boundary_baseline'][key] == leading_row['scores']['boundary_only'][key]
                                   for key in ('tp', 'fp', 'fn')), 'Historical boundary-only scorer replay differs'
                        entry['leadingCombinedScores'] = leading_row['scores']['combined']
                        entry['gainedVsOriginal'] = [p['identity'] for p, q in zip(entry['scores']['candidate']['landmarks'], entry['scores']['original']['landmarks'], strict=True) if p['correct'] and not q['correct']]
                        entry['lostVsOriginal'] = [p['identity'] for p, q in zip(entry['scores']['candidate']['landmarks'], entry['scores']['original']['landmarks'], strict=True) if q['correct'] and not p['correct']]
                        entry['gainedVsBoundaryBaseline'] = [p['identity'] for p, q in zip(entry['scores']['candidate']['landmarks'], entry['scores']['boundary_baseline']['landmarks'], strict=True) if p['correct'] and not q['correct']]
                        entry['lostVsBoundaryBaseline'] = [p['identity'] for p, q in zip(entry['scores']['candidate']['landmarks'], entry['scores']['boundary_baseline']['landmarks'], strict=True) if q['correct'] and not p['correct']]
                    if changed:
                        preview_path = RUN / f'changed_{index:02d}.png'
                        preview = np.concatenate((panel(reference, boundary_baseline, f'{index} {row["id"]} same optimizer baseline'),
                                                  panel(reference, actual, 'surface boundary candidate')), axis=1)
                        assert cv2.imwrite(str(preview_path), preview)
                        entry['panel'] = {'path': name(preview_path), **binding(preview_path)}
                if group != 'pilot':
                    assert all(entry['scores']['boundary_baseline'][key] == leading_row['scores']['boundary_only'][key]
                               for key in ('tp', 'fp', 'fn')), 'Complete boundary-only scorer replay differs'
                    entry['leadingCombinedScores'] = leading_row['scores']['combined']
                report['cases'].append(entry)
                progress.write(json.dumps({'kind': 'existing_seed', 'entry': entry}, allow_nan=False) + '\n')
                progress.flush()
                print(json.dumps({'case': index, 'id': row['id'], 'accepted': entry['accepted'], 'changed': entry.get('changed', False)}), flush=True)

            canonical = TennisCourtGeometry.get_canonical_keypoints().astype(float)
            for index, row in enumerate(automatic_inputs()):
                image = decode(row['path'], 0)
                scale = 960 / image.shape[1]
                reference = cv2.resize(image, (960, round(image.shape[0] * scale)))
                pixel = hashlib.sha256(image.tobytes()).hexdigest()
                original_path = RUN / f'automatic_{index:02d}_original.png'
                assert cv2.imwrite(str(original_path), image)
                surface, proposed, baseline_passing = selected(reference)
                entry = {'index': index, 'id': row['id'], 'group': row['group'], 'source': name(row['path']),
                         'sourceBinding': binding(row['path']), 'exactSequentialFrame': 0,
                         'decodedNativePixelSha256': pixel, 'nativeSize': [image.shape[1], image.shape[0]],
                         'original': {'path': name(original_path), **binding(original_path)},
                         'acceptedImageHypothesis': False, 'physicalCourtValidated': False,
                         'proposals': proposed, 'baselineProposals': baseline_passing, 'baselineProposalCount': len(baseline_passing), 'candidateProposalCount': len(segments(proposed)),
                         'addedProposalCount': sum(item.get('evidence_origin') == 'SURFACE_BOUNDARY_RECOVERY' for item in proposed)}
                if surface is None:
                    entry['reason'] = 'NO_FOUR_CORNER_COMPONENT_AT_FROZEN_EPSILON'
                    seed = actual = None
                else:
                    quad = np.asarray(surface['quadrilateral'], dtype=float)
                    center = quad.mean(axis=0)
                    corners = quad[np.argsort(np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0]))]
                    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
                    first = 0 if lengths[0] + lengths[2] <= lengths[1] + lengths[3] else 1
                    edges = [corners[[first, (first + 1) % 4]], corners[[(first + 2) % 4, (first + 3) % 4]]]
                    edges.sort(key=lambda edge: edge[:, 1].mean())
                    ordered = np.vstack([edge[np.argsort(edge[:, 0])] for edge in edges])
                    matrix = cv2.getPerspectiveTransform(canonical[:4].astype(np.float32), ordered.astype(np.float32))
                    seed = cv2.perspectiveTransform(canonical[None], matrix)[0]
                    original_segments = segments(baseline_passing)
                    baseline_actual, baseline_refinement = refine(seed, original_segments, 960, reference.shape[0])
                    baseline_support = support(baseline_actual, original_segments, 960, reference.shape[0])
                    baseline_admitted = bool(baseline_refinement['accepted'] and baseline_support['diagnostic_gate_pass'] and
                                             sum(part['supported_fraction'] >= .5 for part in baseline_support['lines']) >= 6)
                    actual, refinement = refine(seed, segments(proposed), 960, reference.shape[0])
                    evidence = support(actual, segments(proposed), 960, reference.shape[0])
                    admitted = bool(refinement['accepted'] and evidence['diagnostic_gate_pass'] and
                                    sum(part['supported_fraction'] >= .5 for part in evidence['lines']) >= 6)
                    entry.update(acceptedImageHypothesis=admitted, reason=refinement['reason'],
                                 component={key: value for key, value in surface['component'].items() if key != 'mask'},
                                 quadrilateralReference=quad.tolist(), seedReference=seed.tolist(),
                                 baselineReturnedReference=baseline_actual.tolist(), baselineRefinement=baseline_refinement,
                                 baselineImageSupport=baseline_support, baselineAcceptedImageHypothesis=baseline_admitted,
                                 returnedReference=actual.tolist(), refinement=refinement, imageSupport=evidence)
                    if row['id'] == 'dimitrov_tiafoe_2024':
                        frozen = ROOT / 'outputs/vision_upgrade_audit/dimitrov_court_image01'
                        exact = cv2.imread(str(frozen / '01_original_frame0.png'))
                        assert np.array_equal(image, exact), 'Dimitrov source pixels differ from corrected review'
                        prior = load(frozen / 'report.json')
                        assert np.max(np.abs(seed - np.asarray(prior['candidate']['seed_landmarks_native_px']) / 2)) < 1e-6
                        marks = load(frozen / 'review.json')['corrected_approximate_coordinates_native_px']
                        errors = [None if mark is None else float(np.linalg.norm(actual[j] / scale - mark)) for j, mark in enumerate(marks)]
                        entry['approximateSourceReview'] = {'uncertaintyNativePx': 8, 'qualifiedGroundTruth': False,
                            'distancesNativePx': errors, 'meanNativeDistancePx': float(np.mean([value for value in errors if value is not None]))}
                preview_path = RUN / f'automatic_{index:02d}_panel.png'
                preview = np.concatenate((panel(reference, seed, row['id'] + ' automatic seed'),
                                          panel(reference, actual, 'returned candidate; physical mapping unqualified')), axis=1)
                assert cv2.imwrite(str(preview_path), preview)
                entry['panel'] = {'path': name(preview_path), **binding(preview_path)}
                report['automatic'].append(entry)
                progress.write(json.dumps({'kind': 'automatic', 'entry': entry}, allow_nan=False) + '\n')
                progress.flush()
                print(json.dumps({'automatic': row['id'], 'accepted': entry['acceptedImageHypothesis'], 'reason': entry['reason']}), flush=True)

            false_control = load(ROOT / 'outputs/vision_upgrade_audit/court_calibration_support01/report.json')['cases'][166]
            source = ROOT / 'data/external/uvy_tennis_videos/UVY/tennis_V02/vid_maRB027fzxc.mp4'
            assert false_control['index'] == 166 and false_control['id'] == 'tennis_V02'
            assert false_control['group'] == 'camera_control' and false_control['model'] == 'uvy_player_tracking_baseline'
            assert false_control['default']['is_valid'] is False
            assert binding(source)['sha256'] == false_control['input_sha256']
            image = decode(source, 0)
            assert [image.shape[1], image.shape[0]] == false_control['native_size']
            scale = 960 / image.shape[1]
            reference = cv2.resize(image, (960, round(image.shape[0] * scale)))
            _, proposed, baseline_passing = selected(reference)
            seed = np.asarray(false_control['predictions']['alternate']) * scale
            baseline_actual, baseline_evidence = refine(seed, segments(baseline_passing), 960, reference.shape[0])
            baseline_support = support(baseline_actual, segments(baseline_passing), 960, reference.shape[0])
            actual, evidence = refine(seed, segments(proposed), 960, reference.shape[0])
            supported = support(actual, segments(proposed), 960, reference.shape[0])
            report['spectatorHypothesisControl'] = {'originalProductionCalibrationRejected': True,
                'source': name(source), 'sourceBinding': binding(source), 'exactSequentialFrame': 0,
                'nativeSize': [image.shape[1], image.shape[0]], 'decodedNativePixelSha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'seedReference': seed.tolist(), 'returnedReference': actual.tolist(),
                'baselineProposals': baseline_passing, 'baselineProposalCount': len(baseline_passing),
                'proposals': proposed, 'candidateProposalCount': len(segments(proposed)),
                'addedProposalCount': sum(item.get('evidence_origin') == 'SURFACE_BOUNDARY_RECOVERY' for item in proposed),
                'refinement': evidence, 'imageSupport': supported,
                'baselineReturnedReference': baseline_actual.tolist(), 'baselineRefinement': baseline_evidence, 'baselineImageSupport': baseline_support,
                'baselineAcceptedImageHypothesis': bool(baseline_evidence['accepted'] and baseline_support['diagnostic_gate_pass'] and
                    sum(row['supported_fraction'] >= .5 for row in baseline_support['lines']) >= 6),
                'acceptedImageHypothesis': bool(evidence['accepted'] and supported['diagnostic_gate_pass'] and
                    sum(row['supported_fraction'] >= .5 for row in supported['lines']) >= 6),
                'scope': 'Saved incorrect spectator hypothesis; specificity diagnostic only.'}
            control_panel = RUN / 'spectator_control_panel.png'
            assert cv2.imwrite(str(control_panel), np.concatenate((panel(reference, seed, 'saved incorrect spectator seed'), panel(reference, actual, 'returned candidate; negative control')), axis=1))
            report['spectatorHypothesisControl']['panel'] = {'path': name(control_panel), **binding(control_panel)}
            save(RUN / 'spectator_control.json', report['spectatorHypothesisControl'])
        report['pooled'] = {}
        for group in ('selection', 'external_source_check'):
            report['pooled'][group] = {}
            for stage in ('projected', 'original', 'boundary_baseline', 'candidate'):
                scored = [row['scores'][stage] for row in report['cases'] if row['group'] == group]
                counts = {key: sum(row[key] for row in scored) for key in ('tp', 'fp', 'fn')}
                report['pooled'][group][stage] = counts
        assert len(report['cases']) == 64 and len(report['automatic']) == 9
        assert not any(name == 'torch' or name.startswith(('torch.', 'ultralytics')) for name in sys.modules)
        verify(ordinary_before)
        report['ordinarySourcesAfter'] = ordinary_sources()
        verify(report['ordinarySourcesAfter'])
        assert all(report['ordinarySourcesAfter'].get(key) == value for key, value in ordinary_before.items())
        verify(before['bindings'])
        assert product_snapshot() == before['productBefore']
        report['sourcesUnchanged'] = report['savedProductUnchanged'] = True
        report['complete'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['childElapsedSeconds'] = time.perf_counter() - began
        report['actualProcessMemoryBytes'] = actual.memory_info()._asdict()
        report['completedAt'] = stamp()
        save(RUN / 'report.json', report)


def run():
    import psutil
    prepared = load(UNIT / 'preparation.json')
    verify(prepared['bindings'])
    assert product_snapshot() == prepared['productBefore']
    RUN.mkdir(exist_ok=False)
    save(RUN / 'launch_marker.json', {'at': stamp(), 'oneShot': True, 'guards': GUARDS})
    resource = {'startedAt': stamp(), 'status': 'NOT_STARTED', 'passed': False, 'childStarted': False,
                'samples': 0, 'peakOwnedTreeRssBytes': 0, 'minimumAvailableRamBytes': psutil.virtual_memory().available}
    process = owned = None
    began = time.monotonic()
    try:
        assert resource['minimumAvailableRamBytes'] >= GUARDS['preflightAvailableBytes'], 'RAM preflight below750MB'
        monitor = module_at('_surface_owned_monitor', MONITOR)
        arguments = [str(BASE), '-B', str(Path(__file__)), '--child']
        with (RUN / 'stdout.log').open('xb') as log, (RUN / 'samples.jsonl').open('x', encoding='utf-8') as samples:
            process = subprocess.Popen(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            owned = monitor.OwnedProcessTree(process)
            resource['childStarted'] = True
            resource['launcherIdentity'] = {'pid': owned.root_identity[0], 'createTimeUnixSeconds': owned.root_identity[1]}
            low = 0
            while process.poll() is None:
                rows = owned.sample()
                available = psutil.virtual_memory().available
                rss = sum(row['memoryBytes']['rss'] for row in rows)
                elapsed = time.monotonic() - began
                samples.write(json.dumps({'at': stamp(), 'elapsedSeconds': elapsed, 'ownedProcesses': rows,
                                          'ownedTreeRssBytes': rss, 'availableRamBytes': available}) + '\n')
                samples.flush()
                resource['samples'] += 1
                resource['peakOwnedTreeRssBytes'] = max(resource['peakOwnedTreeRssBytes'], rss)
                resource['minimumAvailableRamBytes'] = min(resource['minimumAvailableRamBytes'], available)
                if (RUN / 'actual_identity.json').exists() and not (RUN / 'actual_sampled.json').exists():
                    identity = load(RUN / 'actual_identity.json')
                    if owned.observed_workload(identity):
                        handshake(RUN / 'actual_sampled.json', {'at': stamp(), 'identity': identity, 'sampledByExactIdentity': True})
                low = low + 1 if available < GUARDS['minimumAvailableBytes'] else 0
                reason = 'RSS above1GB' if rss > GUARDS['maximumOwnedRssBytes'] else 'HostRAM below400MB5samples' if low >= 5 else 'Wall above600seconds' if elapsed > 600 else None
                if reason:
                    resource['safeguardStop'] = reason
                    raise RuntimeError(reason)
                time.sleep(.25)
            resource['exitCode'] = process.returncode
        completed = load(RUN / 'report.json')
        assert process.returncode == 0 and completed['complete'], 'Incomplete candidate benchmark; retain partial evidence'
        verify(completed['ordinarySourcesBefore'])
        verify(completed['ordinarySourcesAfter'])
        assert owned.observed_workload(load(RUN / 'actual_identity.json'))
        resource['status'], resource['passed'] = 'COMPLETED', True
    except BaseException as error:
        resource['status'] = 'NOT_STARTED' if not resource['childStarted'] else 'STOPPED' if resource.get('safeguardStop') else 'FAILED'
        resource['error'] = type(error).__name__ + ': ' + str(error)
        if process is not None:
            resource['terminationActions'] = monitor.stop_owned(process, owned)
    finally:
        if owned is not None:
            resource['ownedIdentities'] = owned.identities()
            remaining = owned.wait_for_retirement()
            resource['remainingOwnedIdentities'] = remaining
            resource['allOwnedIdentitiesRetired'] = not remaining
            if remaining:
                resource['passed'] = False
        try:
            verify(prepared['bindings'])
            resource['sourcesUnchanged'] = True
            resource['savedProductUnchanged'] = product_snapshot() == prepared['productBefore']
            assert resource['savedProductUnchanged']
        except BaseException as error:
            resource['bindingError'] = repr(error)
            resource['passed'] = False
        resource['elapsedSeconds'] = time.monotonic() - began
        resource['completedAt'] = stamp()
        save(RUN / 'resource_receipt.json', resource)
    print(json.dumps({key: resource.get(key) for key in ('passed', 'status', 'error', 'elapsedSeconds')}), flush=True)
    return 0 if resource['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--child', action='store_true')
    arguments = parser.parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.child:
        child()
    else:
        raise SystemExit(run())
