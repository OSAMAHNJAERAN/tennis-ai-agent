"""Compare frozen source-only visual estimates with saved image observations."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'artifacts/validation/product_execution/ball_render_original_review'
RESULT = ROOT / 'outputs/product/analyses/match_1bb12f2aa52a438ebce8a9895fa2bd4f/result.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    labels_path = REVIEW / 'labels.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    assert labels['frozen'] and labels['prediction_blind']
    result = json.loads(RESULT.read_text(encoding='utf-8'))
    points = result['trajectories']['ballTrajectory']
    comparisons = []
    for label in labels['frames']:
        assert digest(ROOT / label['path']) == label['original_png_sha256']
        point = points[label['decoder_frame_index']]
        reference = label['center_pixels']
        observed = [point['x_px'], point['y_px']]
        error = None
        if label['visibility'] == 'visible' and reference is not None and all(value is not None for value in observed):
            error = math.dist(observed, [reference['x'], reference['y']])
        comparisons.append({'frame_index': label['decoder_frame_index'], 'review_visibility': label['visibility'],
                            'review_center': reference, 'review_error_interval_pixels': label['error_interval_pixels'],
                            'observation_state': point['state'], 'observation_center': observed,
                            'center_separation_px': error,
                            'included_in_accuracy_metric': False})
    report = {'analysis_id': result['summary']['id'], 'labels_sha256': digest(labels_path),
              'result_sha256': digest(RESULT), 'comparisons': comparisons,
              'visible_estimates': sum(label['visibility'] == 'visible' for label in labels['frames']),
              'uncertain_frames': sum(label['visibility'] == 'uncertain' for label in labels['frames']),
              'confirmed_absent_frames': sum(label['visibility'] == 'invisible_or_occluded' for label in labels['frames']),
              'metric_status': 'NOT_QUALIFICATION; APPROXIMATE_SINGLE_AI_DEVELOPMENT_REVIEW',
              'notes': ['No detector rerun, label revision, threshold change or production promotion.',
                        'Four uncertain frames are excluded from localization/absence claims.',
                        'There are no confirmed absent labels; an absent-frame false-positive rate cannot be computed.',
                        'Long trails can reflect actual observed travel; existing renderer clears unavailable coordinates.',
                        'The adjacent extreme image jumps remain a coherence diagnostic, not adjudicated false positives.']}
    destination = REVIEW / 'saved_observation_comparison.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'report': str(destination.relative_to(ROOT)),
                      'visible_center_separations_px': [row['center_separation_px'] for row in comparisons
                                                         if row['review_visibility'] == 'visible'],
                      'uncertain_frames_excluded': report['uncertain_frames'],
                      'accuracy_claim': False}))


if __name__ == '__main__':
    main()
