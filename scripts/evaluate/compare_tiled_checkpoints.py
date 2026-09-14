"""Compare checkpoint changes with identical continuous inference and labels."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.evaluate.audit_ball_candidate_coverage import digest
from scripts.evaluate.compare_ball_resolution import compare
from scripts.evaluate.summarize_ball_validation import summarize


def compare_reports(baseline, candidate):
    if not baseline['complete'] or not candidate['complete']:
        raise ValueError('Both continuous reports must be complete')
    for key in ('dataset_manifest_sha256', 'configuration'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'Checkpoint comparison changed {key}')
    baseline_code = {name: value for name, value in baseline['code_hashes'].items()
                     if name != 'scripts/evaluate/benchmark_tiled_ball_stream.py'}
    candidate_code = {name: value for name, value in candidate['code_hashes'].items()
                      if name != 'scripts/evaluate/benchmark_tiled_ball_stream.py'}
    if baseline_code != candidate_code:
        raise ValueError('Inference, filter or metric code differs')
    if len(baseline['clips']) != len(candidate['clips']):
        raise ValueError('Clip count changed')
    seen = set()
    for first, second in zip(baseline['clips'], candidate['clips'], strict=True):
        for key in ('clip', 'frames', 'fps', 'size'):
            if first[key] != second[key]:
                raise ValueError(f'Clip alignment changed: {key}')
        if first['clip'] in seen:
            raise ValueError('Duplicate clip')
        seen.add(first['clip'])
        for clip in (first, second):
            if any(len(clip['predictions'][name]) != clip['frames']
                   for name in ('raw', 'stationary', 'pixel_motion')):
                raise ValueError('Incomplete frame predictions')
    results = {}
    for stage in ('raw', 'stationary', 'pixel_motion'):
        first = [row for clip in baseline['clips'] for row in clip['labeled_rows'][stage]]
        second = [row for clip in candidate['clips'] for row in clip['labeled_rows'][stage]]
        results[stage] = {'baseline': summarize(first), 'candidate': summarize(second),
                          'paired': compare(first, second)}
    return {'qualification_evidence': False, 'complete': True,
            'scope': 'PAIRED_CHECKPOINT_COMPARISON_ON_REUSED_DEVELOPMENT_DATA',
            'baseline_checkpoint_sha256': baseline['checkpoint_sha256'],
            'candidate_checkpoint_sha256': candidate['checkpoint_sha256'],
            'dataset_manifest_sha256': baseline['dataset_manifest_sha256'],
            'compared_clips': len(seen), 'compared_frames': sum(c['frames'] for c in baseline['clips']),
            'benchmark_script_hash_equal': baseline['code_hashes'].get('scripts/evaluate/benchmark_tiled_ball_stream.py') ==
                                           candidate['code_hashes'].get('scripts/evaluate/benchmark_tiled_ball_stream.py'),
            'script_difference_scope': 'Benchmark checkpoint argument added; inference/filter/metric hashes must match',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    result['source_reports'] = {str(p): digest(p) for p in (args.baseline, args.candidate)}
    result['evaluator_sha256'] = digest(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    for stage, value in result['results'].items():
        print(json.dumps({'stage': stage, 'candidate': {k: value['candidate']['pooled'][k]
                                                       for k in ('precision', 'recall', 'f1')},
                          'paired': {k: v for k, v in value['paired'].items() if k != 'changes'}}))


if __name__ == '__main__':
    main()
