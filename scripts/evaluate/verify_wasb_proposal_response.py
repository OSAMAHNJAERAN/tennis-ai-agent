"""Independently recompute frozen response quantities from saved arrays, without inference."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/vision_upgrade_audit/wasb_proposal_response01'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def points(heatmap, inverse, box):
    """Reconstruct the frozen mass-ranked component decoder directly from arrays."""
    count, components = cv2.connectedComponents(np.asarray(heatmap > .2, dtype=np.uint8))
    left, top, right, bottom = box
    found = []
    for component in range(1, count):
        yy, xx = np.nonzero(components == component)
        weights = heatmap[yy, xx]
        mass = float(weights.sum())
        # The frozen upstream affine_transform casts the homogeneous point to
        # float32 before applying its float64 inverse. Preserve that arithmetic.
        center = np.array([float(xx @ weights / mass), float(yy @ weights / mass), 1.], dtype=np.float32)
        native = inverse @ center
        if 0 <= native[0] < right - left and 0 <= native[1] < bottom - top:
            found.append((mass, float(native[0] + left), float(native[1] + top), float(weights.max())))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is expected
    else:
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-7), (actual, expected)


def main():
    destination = RUN / 'independent_verification.json'
    if destination.exists():
        raise FileExistsError('Preserve the existing independent verification.')
    report_path, selection_path = RUN / 'report.json', RUN / 'selection.json'
    protected = {path.name: digest(path) for path in (report_path, selection_path)}
    report, selection = read(report_path), read(selection_path)
    assert report['complete'] and report['qualification_evidence'] is False
    assert report['selection_sha256'] == digest(selection_path)
    assert selection['protocol_sha256'] == digest(ROOT / 'docs/experiments/WASB_PROPOSAL_RESPONSE_PROTOCOL.md')
    source = ROOT / 'artifacts/training/vision_upgrade/wasb_spaced_head_pilot05/manifest.json'
    assert selection['source_sha256'] == digest(source)
    selected = {(entry['clip'], entry['row']['frame']): entry for entry in selection['entries']}
    assert len(selected) == len(report['entries']) == 44
    counts, slot_controls = Counter(), Counter()
    entries = []
    for entry in report['entries']:
        original = selected[(entry['clip'], entry['frame'])]
        assert entry['role'] == original['role'] and entry['target_xy'] == original['row']['target_xy']
        path = RUN / entry['heatmaps']
        assert path.parent.resolve() == RUN.resolve() and digest(path) == entry['heatmaps_sha256']
        target = entry['target_xy']
        width, height = original['width'], original['height']
        def error(x, y):
            return math.hypot((x - target[0]) * 512 / width, (y - target[1]) * 288 / height)
        averaged_active = any_individual_active = any_individual_correct = False
        proposals = []
        with np.load(path, allow_pickle=False) as arrays:
            assert sorted(arrays.files) == [f'view{i}' for i in range(5)]
            for view in entry['views']:
                number, box = view['view'], view['box']
                maps = arrays[f'view{number}']
                assert maps.shape == (len(original['row']['windows']), 288, 512)
                assert np.isfinite(maps).all() and maps.min() >= 0 and maps.max() <= 1
                inverse = np.asarray(view['inverse'])
                yy, xx = np.indices((288, 512))
                homogeneous = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size)))
                native = inverse @ homogeneous
                native[0] += box[0]
                native[1] += box[1]
                inside = ((native[0] >= box[0]) & (native[0] < box[2])
                          & (native[1] >= box[1]) & (native[1] < box[3]))
                averaged = np.mean(maps, axis=0)
                close(float(averaged.ravel()[inside].max()), view['average_global_peak'])
                for i, heatmap in enumerate(maps):
                    close(float(heatmap.ravel()[inside].max()), view['slot_global_peaks'][i])
                if target is not None:
                    squared_error = ((native[0] - target[0]) * 512 / width) ** 2 + ((native[1] - target[1]) * 288 / height) ** 2
                    eligible = inside & (squared_error <= 16)
                    assert int(eligible.sum()) == view['tolerance_pixel_count']
                    active = int(np.count_nonzero((averaged.ravel() > .2) & eligible))
                    assert active == view['average_target_active_pixels']
                    averaged_active |= active > 0
                    close(float(averaged.ravel()[eligible].max()) if eligible.any() else None, view['average_target_peak'])
                    for i, heatmap in enumerate(maps):
                        peak = float(heatmap.ravel()[eligible].max()) if eligible.any() else None
                        if peak is not None:
                            close(peak, view['slot_target_peaks'][i])
                            any_individual_active |= peak > .2
                        decoded_correct = any(error(x, y) <= 4 for _, x, y, _ in points(heatmap, inverse, box))
                        assert decoded_correct == view['slot_correct_proposals'][i]
                        any_individual_correct |= decoded_correct
                else:
                    assert 'average_target_peak' not in view and not any(view['slot_correct_proposals'])
                for rank, (_, x, y, confidence) in enumerate(points(averaged, inverse, box)):
                    proposals.append({'view': number, 'box': box, 'mass_rank': rank, 'x': x, 'y': y, 'confidence': confidence})
        assert len(proposals) == len(original['row']['candidates'])
        for actual, expected in zip(proposals, original['row']['candidates']):
            assert all(actual[key] == expected[key] for key in ('view', 'box', 'mass_rank'))
            assert all(abs(actual[key] - expected[key]) <= 1e-5 for key in ('x', 'y', 'confidence')), (entry['clip'], entry['frame'], actual, expected)
        result = ('COMPONENT_DISPLACEMENT' if averaged_active else 'TEMPORAL_ATTENUATION' if any_individual_active else 'WEAK_RESPONSE') if entry['role'] == 'PROPOSAL_MISS' else entry['role']
        assert result == entry['category'] and entry['prediction_reproduced'] is True
        counts[result] += 1
        if entry['role'] == 'PROPOSAL_MISS' and any_individual_correct:
            slot_controls[result] += 1
        entries.append({'clip': entry['clip'], 'frame': entry['frame'], 'category': result,
                        'individualCorrectProposal': any_individual_correct, 'savedHeatmapSha256': digest(path)})
    assert dict(counts) == report['counts']
    assert protected == {path.name: digest(path) for path in (report_path, selection_path)}
    # Freeze the visual subset before source-image inspection, as required by protocol.
    visual = []
    for group in sorted({entry['category'] for entry in entries if entry['category'] not in ('CORRECT_CONTROL', 'ABSENT_CONTROL')}):
        visual.extend(sorted((entry for entry in entries if entry['category'] == group), key=lambda item: (item['clip'], item['frame']))[:2])
    for role in ('CORRECT_CONTROL', 'ABSENT_CONTROL'):
        visual.append(sorted((entry for entry in entries if entry['category'] == role), key=lambda item: (item['clip'], item['frame']))[0])
    receipt = {'verifiedAt': datetime.now(timezone.utc).isoformat(), 'verificationScriptSha256': digest(Path(__file__)),
               'sourceReportSha256': protected['report.json'], 'selectionSha256': protected['selection.json'],
               'modelInferenceRun': False, 'qualificationEvidence': False, 'all44SavedArraysVerified': True,
               'averagedCandidatesReproduced': True, 'counts': dict(counts),
               'proposalMissesWithIndividualCorrectCandidate': dict(slot_controls),
               'entries': entries, 'predeclaredVisualSubset': visual}
    receipt['verificationImplementationCorrection'] = {
        'initialIssue': 'Independent decoder omitted the upstream float32 point cast before its inverse affine transform.',
        'firstDifferencePx': 0.0000260920741,
        'correction': 'Reproduce the inspected frozen upstream arithmetic; retain the original 1e-5 native-pixel candidate tolerance.',
        'modelOrSavedEvidenceChanged': False}
    destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('counts', 'proposalMissesWithIndividualCorrectCandidate', 'all44SavedArraysVerified', 'predeclaredVisualSubset')}))


if __name__ == '__main__':
    main()
