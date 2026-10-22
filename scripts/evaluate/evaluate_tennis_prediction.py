"""Evaluate causal tennis forecasts from independently reviewed chronological point matches."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.prediction.evaluation import evaluate_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", default=1000, type=int)
    parser.add_argument("--bins", default=10, type=int)
    parser.add_argument("--seed", default=20261008, type=int)
    args = parser.parse_args()
    report = evaluate_file(args.manifest, args.output, bins=args.bins,
                           bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    print(f"Wrote evaluation for {len(report['matches'])} matches; calibration remains unqualified: {args.output}")


if __name__ == "__main__":
    main()
