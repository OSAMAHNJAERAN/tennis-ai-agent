import argparse
import yaml
import os
import shutil
from src.pipeline.phase2_pipeline import Phase2Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Tennis Vision Phase 2 Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, required=True, help="Output directory path")
    parser.add_argument("--config", type=str, default="configs/phase2_ball/pipeline.yaml", help="Config YAML path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device")
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['device'] = args.device
    
    pipeline = Phase2Pipeline(config)
    summary = pipeline.run(args.input, args.output)
    
    print("Phase 2 Execution Summary:")
    print(summary)
    
    os.makedirs(args.output, exist_ok=True)
    shutil.copy(args.config, os.path.join(args.output, "run_config.yaml"))

if __name__ == "__main__":
    main()
