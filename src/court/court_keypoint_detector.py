import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from typing import Optional, List

class CourtKeypointDetector:
    """
    Court keypoint detection model using a fine-tuned ResNet-50.
    Detects 14 keypoints on the tennis court.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the CourtKeypointDetector.
        
        Args:
            model_path: Path to the model weights. If None, uses an untrained model.
            device: 'cpu', 'cuda', or 'auto'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model = self.build_model()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        
        self.model.to(self.device)
        self.model.eval()
        
        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def build_model(self) -> nn.Module:
        """
        Create a ResNet-50 model with a modified final FC layer for 14 keypoints (28 outputs).
        """
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 14 keypoints, each has (x, y) = 28 outputs
        model.fc = nn.Linear(model.fc.in_features, 14 * 2)
        return model

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess the frame for model input.
        - Resize to 224x224
        - BGR to RGB
        - Normalize using ImageNet parameters
        """
        # OpenCV reads in BGR, convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet mean and std
        img_normalized = (img_normalized - self.mean) / self.std
        
        # Convert to CHW format for PyTorch
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        
        return img_chw

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict 14 court keypoints for a single frame.
        
        Args:
            frame: Input image (BGR, numpy array)
            
        Returns:
            np.ndarray: 14x2 array of (x, y) keypoints in original image coordinates
        """
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = torch.from_numpy(self._preprocess(frame)).unsqueeze(0).float().to(self.device)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Reshape to 14x2
        keypoints = output.squeeze().cpu().numpy().reshape(14, 2)
        
        # The model predicts normalized coordinates (usually in the range [0, 1] relative to the 224x224 image).
        # Assuming the model outputs coordinates in the 224x224 space:
        # We need to scale them back to the original image dimensions.
        scale_x = w / 224.0
        scale_y = h / 224.0
        
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        return keypoints

    def predict_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Predict court keypoints for a batch of frames.
        
        Args:
            frames: List of input images (BGR, numpy arrays)
            
        Returns:
            List[np.ndarray]: List of 14x2 arrays of (x, y) keypoints in original image coordinates
        """
        if not frames:
            return []
            
        preprocessed_frames = [self._preprocess(frame) for frame in frames]
        input_tensor = torch.tensor(np.array(preprocessed_frames)).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        results = []
        for i, output in enumerate(outputs):
            h, w = frames[i].shape[:2]
            keypoints = output.cpu().numpy().reshape(14, 2)
            
            scale_x = w / 224.0
            scale_y = h / 224.0
            
            keypoints[:, 0] *= scale_x
            keypoints[:, 1] *= scale_y
            
            results.append(keypoints)
            
        return results
