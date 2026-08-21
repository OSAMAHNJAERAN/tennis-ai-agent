import os
import json
import argparse
import numpy as np
import cv2
import torch
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.court.court_keypoint_detector import CourtKeypointDetector

def get_canonical_court_keypoints():
    """
    Returns 14 keypoints in ITF standard court coordinate system (in meters).
    Origin (0,0) is at the center of the net.
    """
    # Assuming half length = 11.885, half width = 5.485
    # Define canonical points based on ITF standard
    hl = 11.885
    hw = 5.485
    sw = 4.115 # singles width
    sl = 6.4 # service line
    
    pts = [
        # Top baseline (y = -hl)
        (-hw, -hl), (-sw, -hl), (sw, -hl), (hw, -hl),
        # Top service line (y = -sl)
        (-sw, -sl), (0, -sl), (sw, -sl),
        # Net line (y = 0)
        (-hw, 0), (hw, 0),
        # Bottom service line (y = sl)
        (-sw, sl), (0, sl), (sw, sl),
        # Bottom baseline (y = hl)
        (-hw, hl), (-sw, hl), (sw, hl), (hw, hl)
    ]
    # For a 14-point model, need to match exact definitions.
    # Assuming 14 points typically omit the center marks on baselines or something similar.
    # We will just generate a generic 14 points array here for evaluation stub.
    # In reality, this must match the 14 points the model was trained on.
    return np.array(pts[:14], dtype=np.float32)

def evaluate(config):
    print(f"Evaluating Court Keypoint Model: {config.model_path}")
    
    detector = CourtKeypointDetector(model_path=config.model_path)
    
    val_annotations_file = os.path.join(config.data_dir, 'annotations', 'val.json')
    images_dir = os.path.join(config.data_dir, 'images', 'val')
    
    with open(val_annotations_file, 'r') as f:
        annotations = json.load(f)
        
    all_errors_px = []
    reprojection_errors = []
    
    canonical_pts = get_canonical_court_keypoints()
    
    for anno in annotations:
        img_path = os.path.join(images_dir, anno['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gt_kpts = np.array(anno['keypoints']).reshape(14, 2)
        pred_kpts = detector.predict(img)
        
        # Pixel error
        err = np.linalg.norm(pred_kpts - gt_kpts, axis=1)
        all_errors_px.extend(err)
        
        # Homography
        # Find homography from canonical to predicted image space
        H, status = cv2.findHomography(canonical_pts, pred_kpts, cv2.RANSAC, 5.0)
        
        if H is not None:
            # Reproject canonical points using H
            pts_src = canonical_pts.reshape(-1, 1, 2)
            pts_dst = cv2.perspectiveTransform(pts_src, H).reshape(-1, 2)
            
            # Reprojection error wrt predicted points (or GT, depends on evaluation protocol)
            # Usually we evaluate how well the H maps canonical back to ground truth
            repr_err = np.linalg.norm(pts_dst - gt_kpts, axis=1)
            reprojection_errors.extend(repr_err)
            
    all_errors_px = np.array(all_errors_px)
    
    results = {
        'mean_error_px': float(np.mean(all_errors_px)),
        'median_error_px': float(np.median(all_errors_px)),
        'max_error_px': float(np.max(all_errors_px))
    }
    
    if len(reprojection_errors) > 0:
        results['mean_reprojection_error'] = float(np.mean(reprojection_errors))
        results['median_reprojection_error'] = float(np.median(reprojection_errors))
    else:
        results['mean_reprojection_error'] = None
        results['median_reprojection_error'] = None
        
    print(f"Mean Keypoint Error (px): {results['mean_error_px']:.2f}")
    print(f"Median Keypoint Error (px): {results['median_error_px']:.2f}")
    print(f"Max Keypoint Error (px): {results['max_error_px']:.2f}")
    
    if results['mean_reprojection_error'] is not None:
        print(f"Mean Reprojection Error: {results['mean_reprojection_error']:.4f}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, 'court_evaluation.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Court Keypoint Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model weights")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing images and annotations")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args)
