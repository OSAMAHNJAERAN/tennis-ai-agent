import os
import argparse
import shutil
from ultralytics import YOLO

def train(config):
    print(f"Starting YOLO training for ball detection...")
    print(f"Data config: {config.data_yaml}")
    print(f"Output dir: {config.output_dir}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load a pretrained model
    model = YOLO('yolo11m.pt')
    
    # Train the model
    results = model.train(
        data=config.data_yaml,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        patience=config.patience,
        device=config.device,
        seed=config.seed,
        project=config.output_dir,
        name="ball_detector",
        exist_ok=True
    )
    
    print("\nTraining completed.")
    print("Evaluating on validation and test splits...")
    
    # Val evaluation is automatically done at the end of training
    # Output metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print("Validation Results:")
        print(f"mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
    
    # Copy the best model to the root of output_dir
    best_model_path = os.path.join(config.output_dir, "ball_detector", "weights", "best.pt")
    if os.path.exists(best_model_path):
        target_path = os.path.join(config.output_dir, "best_ball_detector.pt")
        shutil.copy2(best_model_path, target_path)
        print(f"Copied best model to {target_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Ball Detector Model using YOLO")
    parser.add_argument('--data_yaml', type=str, required=True, help="Path to data.yaml file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output models and logs")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training")
    parser.add_argument('--batch', type=int, default=16, help="Batch size")
    parser.add_argument('--patience', type=int, default=50, help="Early stopping patience")
    parser.add_argument('--device', type=str, default='', help="Device to use (e.g. '0' or 'cpu')")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    train(args)
