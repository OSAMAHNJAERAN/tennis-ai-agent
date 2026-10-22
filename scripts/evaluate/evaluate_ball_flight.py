"""Evaluate physical ball-speed estimates against synchronized independent references."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analytics.flight_evaluation import evaluate_flight_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_flight_file(args.manifest, args.output)
    print(f"Evaluated {report['all_stroke_count']} declared strokes; physical qualification still requires reference audit: {args.output}")


if __name__ == '__main__':
    main()
