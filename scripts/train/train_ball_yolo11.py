import argparse
import os
import shutil
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 Tennis Ball Detector")
    parser.add_argument("--config", type=str, default="configs/phase2_yolo11/ball_yolo11m.yaml", help="Path to training config")
    parser.add_argument("--target_weight_name", type=str, default="yolo11m_tennis_ball_best.pt", help="Filename for saved artifact")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"\n=======================================================")
    print(f"Starting YOLO11 Ball Detector Training")
    print(f"Base Model: {cfg['model']} | Image Size: {cfg['imgsz']} | Epochs: {cfg['epochs']}")
    print(f"Dataset: {cfg['data']}")
    print(f"=======================================================\n")

    model = YOLO(cfg['model'])
    
    results = model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        patience=cfg['patience'],
        batch=cfg['batch'],
        imgsz=cfg['imgsz'],
        device=cfg['device'],
        optimizer=cfg.get('optimizer', 'AdamW'),
        lr0=cfg.get('lr0', 0.001),
        lrf=cfg.get('lrf', 0.01),
        weight_decay=cfg.get('weight_decay', 0.0005),
        warmup_epochs=cfg.get('warmup_epochs', 3.0),
        plots=cfg.get('plots', True),
        save=cfg.get('save', True),
        workers=0,
        project=cfg.get('project', 'experiments/phase2_yolo11/train'),
        name=cfg.get('name', 'yolo11_ball')
    )

    print("\nTraining complete! Evaluating on test set...")
    val_results = model.val(data=cfg['data'], split='test')
    
    # Save to artifacts/models/ball/
    artifacts_dir = "artifacts/models/ball"
    os.makedirs(artifacts_dir, exist_ok=True)
    best_weight_src = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    target_path = os.path.join(artifacts_dir, args.target_weight_name)
    
    if os.path.exists(best_weight_src):
        shutil.copy(best_weight_src, target_path)
        print(f"\nSaved best model weights to: {target_path}")
    else:
        print(f"\nWarning: {best_weight_src} not found, saving model directly")
        model.save(target_path)

    print("\nEvaluation Metrics (Test Set):")
    print(f"  Precision: {val_results.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall:    {val_results.results_dict.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP50:     {val_results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP50-95:  {val_results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

if __name__ == "__main__":
    main()
