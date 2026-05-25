# Model utilities and helper functions

import os
import time
from typing import Dict, Tuple, Optional, Any

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    # Stub so the rest of the module doesn't fail at import time
    nn = None
    torch = None


if _TORCH_AVAILABLE:
    class DeepfakeDetector(nn.Module):
        """Enhanced Xception-based deepfake detection model"""
        def __init__(self, pretrained=False):
            super(DeepfakeDetector, self).__init__()
            # Create backbone sequential model with enhanced architecture
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 0),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 512, 3, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            
            # Enhanced classification head with extra FC layer
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout1 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(512, 256)
            self.relu = nn.ReLU(inplace=True)
            self.dropout2 = nn.Dropout(0.3)
            self.fc2 = nn.Linear(256, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.backbone(x)
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout1(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            return self.sigmoid(x)
else:
    class DeepfakeDetector:  # type: ignore[no-redef]
        """Stub when PyTorch is unavailable"""
        pass


class ModelUtils:
    """Utility class for model operations"""

    @staticmethod
    def load_model(model_path: str, device: Optional[Any] = None) -> Tuple[Any, Any]:
        """Load a trained model with error handling"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = DeepfakeDetector()
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        return model, device

    @staticmethod
    def preprocess_image(image_path: str, image_size: int = 299) -> Tuple[Any, float]:
        """Preprocess image for model input and return preprocessing time"""
        start_time = time.time()
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        
        preprocessing_time = time.time() - start_time
        return tensor, preprocessing_time

    @staticmethod
    def predict_image(model: Any, image_tensor: Any, device: Any) -> Dict[str, Any]:
        """Make prediction with detailed information"""
        start_time = time.time()
        
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output = model(image_tensor)
            confidence_raw = output.item()
            
        inference_time = time.time() - start_time
        
        # Determine prediction
        prediction = "Fake" if confidence_raw > 0.5 else "Real"
        
        # Calculate confidence percentage (0-100%)
        if prediction == "Fake":
            confidence_percent = confidence_raw * 100
        else:
            confidence_percent = (1 - confidence_raw) * 100
        
        # Determine threat level
        if confidence_raw > 0.7:
            threat_level = "high"
        elif confidence_raw > 0.4:
            threat_level = "medium"
        else:
            threat_level = "low"
        
        return {
            "prediction": prediction,
            "confidence": confidence_percent,
            "confidence_raw": confidence_raw,
            "threat_level": threat_level,
            "inference_time_ms": round(inference_time * 1000, 2)
        }

    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """Get comprehensive information about the model"""
        info = {
            "model_name": "Verifixia AI Xception",
            "version": "2.4.1",
            "architecture": "Xception-based CNN",
            "input_size": "299x299",
            "framework": "PyTorch",
        }
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            info.update({
                "exists": True,
                "size_mb": round(file_size, 2),
                "path": model_path,
                "status": "loaded"
            })
        else:
            info.update({
                "exists": False,
                "path": model_path,
                "status": "not_found"
            })
        
        return info

    @staticmethod
    def get_model_metadata(model: Any, device: Any) -> Dict[str, Any]:
        """Get detailed model metadata including parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "layers": {
                "convolutional": 5,
                "batch_norm": 5,
                "fully_connected": 1,
                "dropout": 1
            },
            "architecture_details": {
                "entry_flow": "Conv2d(3→32→64)",
                "middle_flow": "Conv2d(64→128→256)",
                "exit_flow": "Conv2d(256→512)",
                "classifier": "FC(512→1) + Sigmoid"
            }
        }

    @staticmethod
    def interpret_confidence(confidence_raw: float) -> Dict[str, str]:
        """Interpret confidence score and provide detailed analysis"""
        if confidence_raw > 0.9:
            return {
                "level": "Very High",
                "description": "Strong indicators of deepfake manipulation detected",
                "recommendation": "Content should be flagged and reviewed"
            }
        elif confidence_raw > 0.7:
            return {
                "level": "High",
                "description": "Multiple deepfake artifacts identified",
                "recommendation": "Content likely manipulated, further analysis recommended"
            }
        elif confidence_raw > 0.5:
            return {
                "level": "Moderate",
                "description": "Some suspicious patterns detected",
                "recommendation": "Content may be manipulated, manual review suggested"
            }
        elif confidence_raw > 0.3:
            return {
                "level": "Low",
                "description": "Minimal deepfake indicators found",
                "recommendation": "Content appears mostly authentic"
            }
        else:
            return {
                "level": "Very Low",
                "description": "No significant manipulation detected",
                "recommendation": "Content appears authentic"
            }
