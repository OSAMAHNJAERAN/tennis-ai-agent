import argparse
import sys
import os
import json

from src.pipeline.phase4_pipeline import Phase4Pipeline

def main():
    parser = argparse.ArgumentParser(description="T88J709 Phase 4 Assisted Tennis IN/OUT Line-Calling Engine Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to input tennis video file")
    parser.add_argument("--output", type=str, default="outputs/phase4_line_calls_1", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/phase4_line_calls/pipeline.yaml", help="Path to config YAML")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    pipeline = Phase4Pipeline(config_path=args.config)
    summary = pipeline.run(input_video_path=args.input, output_dir=args.output)
    
    print("\nPhase 4 Execution Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
