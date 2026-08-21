import os
import json
import argparse
from ultralytics import YOLO

def evaluate(config):
    print(f"Evaluating Ball Detector Model: {config.model_path}")
    print(f"Data config: {config.data_yaml}")
    
    model = YOLO(config.model_path)
    
    # Run evaluation
    metrics = model.val(data=config.data_yaml, split='val')
    
    results = {
        'mAP50': float(metrics.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(metrics.results_dict.get('metrics/mAP50-95(B)', 0)),
        'precision': float(metrics.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(metrics.results_dict.get('metrics/recall(B)', 0))
    }
    
    # Calculate F1
    p = results['precision']
    r = results['recall']
    if p + r > 0:
        results['f1'] = 2 * (p * r) / (p + r)
    else:
        results['f1'] = 0.0
        
    # YOLO val doesn't inherently give "per-frame detection rate" (frames with >=1 ball / total frames)
    # without parsing per-image results, but we record standard COCO metrics.
    # Note: For exact per-frame rate, custom parsing of predictions would be needed.
        
    print("\nEvaluation Results:")
    print(f"mAP50: {results['mAP50']:.4f}")
    print(f"mAP50-95: {results['mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, 'ball_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {os.path.join(config.output_dir, 'ball_evaluation.json')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Ball Detector Model using YOLO")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained YOLO model")
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to data.yaml file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args)
