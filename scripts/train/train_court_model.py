import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple

class CourtKeypointDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Dataset for court keypoints.
        Loads images and keypoint annotations.
        """
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images', split)
        self.annotations_file = os.path.join(data_dir, 'annotations', f'{split}.json')
        
        with open(self.annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anno = self.annotations[idx]
        img_path = os.path.join(self.images_dir, anno['file_name'])
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        h, w = img.shape[:2]
        
        # BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        img_resized = cv2.resize(img_rgb, (224, 224))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_normalized = (img_normalized - self.mean) / self.std
        
        # Convert to CHW
        img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))).float()
        
        # Process keypoints (14x2)
        # We expect annotations to provide 14 points (x, y)
        kpts = np.array(anno['keypoints'], dtype=np.float32).reshape(-1, 2)
        
        # Scale keypoints to 224x224 space
        kpts[:, 0] = kpts[:, 0] * (224.0 / w)
        kpts[:, 1] = kpts[:, 1] * (224.0 / h)
        
        # Flatten to (28,)
        kpts_tensor = torch.from_numpy(kpts.flatten()).float()
        
        return img_tensor, kpts_tensor

def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_errors = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate pixel error in 224x224 space
            # outputs shape: (B, 28)
            preds = outputs.cpu().numpy().reshape(-1, 14, 2)
            gts = targets.cpu().numpy().reshape(-1, 14, 2)
            
            # Euclidean distance per keypoint
            errors = np.linalg.norm(preds - gts, axis=2) # shape: (B, 14)
            all_errors.extend(errors.flatten())
            
    mean_loss = total_loss / len(val_loader)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    
    return mean_loss, float(mean_error), float(std_error)

def train(config):
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    train_dataset = CourtKeypointDataset(config.data_dir, split='train')
    val_dataset = CourtKeypointDataset(config.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 14 * 2)
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'mean_keypoint_error': []}
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        val_loss, mean_error, std_error = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mean_keypoint_error'].append(mean_error)
        
        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Mean Error (px in 224): {mean_error:.2f} ± {std_error:.2f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(config.output_dir, 'best_court_model.pt')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
            
    # Save history
    with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Court Keypoint Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing images and annotations")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output models and logs")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for training")
    
    args = parser.parse_args()
    train(args)
