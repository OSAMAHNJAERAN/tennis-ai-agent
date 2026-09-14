"""Controlled trajectory/noise benchmark for player ground-motion estimates.

Noise levels are declared stress conditions, not measured detector uncertainty.
Only synthetic trajectories have ground truth here; no real speed claim follows.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.evaluate.replay_wasb_thresholds import digest
from src.tracking.player_motion_tracking import summarize_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for fps in (25, 30, 60):
        timestamps = np.arange(fps * 5 + 1, dtype=float) / fps
        for scenario in ('stationary', 'constant_velocity', 'smooth_turn'):
            if scenario == 'stationary':
                truth = np.column_stack((np.full_like(timestamps, 5.), np.full_like(timestamps, 10.)))
                speed = np.zeros_like(timestamps)
            elif scenario == 'constant_velocity':
                truth = np.column_stack((3 * timestamps, np.zeros_like(timestamps)))
                speed = np.full_like(timestamps, 3.)
            else:
                omega = .8
                truth = np.column_stack((4 * np.cos(omega * timestamps), 4 * np.sin(omega * timestamps)))
                speed = np.full_like(timestamps, 4 * omega)
            for noise_sigma in (0., .03, .08):
                observed = truth + np.random.default_rng(7).normal(0, noise_sigma, truth.shape)
                summary = summarize_motion(observed.tolist(), timestamps.tolist())
                values = [(index, sample['speed_kmh'] / 3.6) for index, sample in enumerate(summary['samples'])
                          if sample['speed_kmh'] is not None and timestamps[index] >= .3]
                expected_distance = float(np.trapezoid(speed, timestamps))
                measured = summary['observed_distance_m']
                results.append({'fps': fps, 'scenario': scenario, 'noise_sigma_m': noise_sigma,
                                'expected_distance_m': expected_distance, 'estimated_distance_m': measured,
                                'distance_error_m': measured - expected_distance if measured is not None else None,
                                'mean_speed_kmh': summary['mean_speed_kmh'],
                                'speed_rmse_mps_after_warmup': float(np.sqrt(np.mean([(value - speed[index]) ** 2
                                                                                    for index, value in values]))),
                                'measured_duration_s': summary['measured_duration_s'],
                                'accepted_intervals': summary['accepted_intervals']})
    result = {'schema_version': '1.0', 'qualification_evidence': False,
              'scope': 'SYNTHETIC_KNOWN_TRAJECTORIES; DECLARED_NOISE_STRESS_NOT_REAL_DETECTOR_CALIBRATION',
              'motion_code_sha256': digest(ROOT / 'src/tracking/player_motion_tracking.py'),
              'evaluator_sha256': digest(__file__), 'seed': 7, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps([row for row in results if row['fps'] == 60 and row['noise_sigma_m'] == .03]), flush=True)


if __name__ == '__main__':
    main()
